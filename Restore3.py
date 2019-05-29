#! /usr/bin/env python
# version 1.0 16/4/16
# 9/8/16 error message on <256 colours

from gimpfu import *
from math import *
from array import array

def get_colourmapX(image):
# make a copy for trial restoration
    local_copy=pdb.gimp_image_duplicate(image)
    colourmap=palette(local_copy)
    if colourmap==None: return None
    pdb.gimp_image_delete(local_copy)
# unpack colourmap and get parameters for extrapolation
    colmap=[0,0,0]; C_hi=[0,0,0]
    for C in [R,G,B]:
        colmap[C]=colourmap[C]
        s0=sum(colmap[C])
        s1=sum([i*colmap[C][i] for i in range(0,256)])
        C_hi[C]=int(0.5*(max(colmap[C]) + 3.04e-5*(3.0*s1 - 254.0*s0)))
        C_hi[C]=min([C_hi[C], 255])
    return (colmap, C_hi)

#-------------------------------------------------------------------------------------}

def Y(r,g,b):
    rr=expand[r]; gg=expand[g]; bb=expand[b]
    Y = rr*0.2126 + gg*0.7152 + bb*0.0722 + 1e-10
    return Y

#-------------------------------------------------------------------------------------}

def rgb2n(r, g, b):
# computes box number at level 6 from (truncated) rgb values
    return ((ctab[r] << 2) | (ctab[g] << 1) | ctab[b] ) >> 6

#-------------------------------------------------------------------------------------}

def interpolate(c):
# interpolation in ctab is used to convert colour number to rgb
    i=0; j=128
    while j:
        if c >= ctab[i+j]: i+=j
        j=j>>1
    return i

#-------------------------------------------------------------------------------------}

def n2rgb(n, level):
# computes rgb values at centre of box
# shift to adjust for level
    nn=n << (24 - 3*level)
# unpack in r and g parts
    nr=(nn & 0x924924) >> 2
    ng=(nn & 0x492492) >> 1
    nb=(nn & 0x249249)
# get lower value in box
    r=interpolate(nr); g=interpolate(ng); b=interpolate(nb)
# add half box length
    mid=0x80 >> level
    return (r | mid, g | mid, b | mid)

#-------------------------------------------------------------------------------------}

def colourfit(x, l):
# computes measure of optimality
    (c, n, sub_boxes)=x
    s=n/2**l
    for box in sub_boxes: s+=colourfit(box, l+1)
    return s

#-------------------------------------------------------------------------------------}

def get_colourlist(x, l):
# converts colour tree to list of colours
    colourlist=[]
    (c, n, sub_boxes)=x
    if n>0: colourlist.append(n2rgb(c,l))
    for box in sub_boxes: colourlist+=get_colourlist(box, l+1)
    return colourlist

#-------------------------------------------------------------------------------------}

def loss(x, lo, parent, l):
# calculates node that can be moved up tree with least penalty
    (c, n, sub_boxes)=x
    z=n>>l
    if l>0 and n>0 and z<lo[0]: lo=[z, l, x, parent]
    for box in sub_boxes: lo=loss(box, lo, x, l+1)
    return lo

#-------------------------------------------------------------------------------------}

def gain(x, hi, l):
# calculates node that can be moved down tree with greatest gain
    if l==6: return hi
    (c, n, sub_boxes)=x
    z=n>>l
    if z>hi[0]: hi=[z, l, x]
    for box in sub_boxes: hi=gain(box, hi, l+1)
    return hi

#-------------------------------------------------------------------------------------}

def move_down(l, box):
# move colours down level
    global coltree, numcols
    (c, n, sub_boxes)=box
# colour root; sub boxes are coloured 8*c + j
    cc=c << 3
    threshold=n/8
# make list of unused subboxes
    z=[0,1,2,3,4,5,6,7,]
    for sub in sub_boxes: z.remove(sub[0] - cc)
# get pixels at lower level
    q=p[l+1][cc:cc+8]
    for j in z:
# don't make small numbers of pixels into new colour
        if q[j]<=threshold: continue
        newcol=cc+j
# add entry in list of subboxes and increase count of colours
        box[2].append([newcol, q[j], []])
        numcols+=1
        box[1]-=q[j]
# if all pixels moved down original colour not used
    if box[1]==0: numcols-=1
    return box

#-------------------------------------------------------------------------------------}

def move_up(l, box, parent):
# moves node up a level
    global coltree, numcols
    (c, n, sub_boxes)=box
    newcol=c >> 3
    i=parent[2].index(box)
    sub=parent[2][i]
# if the parent box had no pixels we create new colour
    if parent[1]==0: numcols+=1
# move pixels from box to parent
    parent[1]+=n
    parent[2][i][1]=0
    numcols-=1
# if there are no sub boxes delete box
    if not box[2]:
        del parent[2][i]
    return

#-------------------------------------------------------------------------------------}

def palette(small):
# computes optimum colour map
    global p, coltree, numcols
    draw_in=pdb.gimp_image_get_active_drawable(small)
    width = draw_in.width; height = draw_in.height
    pr = draw_in.get_pixel_rgn(0, 0, width, height, True, False)

    px=array("B")
    px.fromstring(pr[0:width, 0:height])

    rr=px[0:len(px):3]; gg=px[1:len(px):3]; bb=px[2:len(px):3]
    n=len(rr)
# create lists to contain counts
    p0=[]
    Nlevels=6
    for level in range(0, Nlevels + 1):
        p0.append(array('i', [0 for i in range(0, 8**level)]))
# count pixels in levels
    p=p0
    it=iter(px)
    try:
        while True:
            r=next(it)
            g=next(it)
            b=next(it)

            i=rgb2n(r, g, b)
            p[6][i]+=1
            p[5][i>>3]+=1
            p[4][i>>6]+=1
            p[3][i>>9]+=1
            p[2][i>>12]+=1
            p[1][i>>15]+=1
            p[0][i>>18]+=1

    except StopIteration: pass

# initial colour tree
    numcols=0
    L2=[[] for i in range(0,8)]
    for i in range(0,8):
        for j in range(0,8):
            c=8*i+j
            if p[2][c]>0:
                L2[i].append([c, p[2][c], []])
                numcols+=1
    L1=[]
    for i in range(0,8):
        if L2[i]: L1.append([i, 0, L2[i]])
    coltree=[0, 0, L1]

# target number of colurs
    Ntarg=256
    lastfit=1e10
# count of moves
    k=0
    while True:
        if numcols<Ntarg:
            G=gain(coltree, [0, None, None], 0)
            if G[0]==0:
                pdb.gimp_message("Image has <256 distinct colours, impossible to restore")
                return None
            move_down(G[1], G[2])
# note fit before moving up
        s=colourfit(coltree, 0)
        if numcols>=Ntarg:
            L=loss(coltree, [1e10, None, None], [coltree], 0)
            move_up(L[1], L[2], L[3])
# if we have the right number of colours exit if fit getting worse
        if numcols==Ntarg:
            nowfit=s
            if nowfit >= lastfit: break
            lastfit=nowfit
# count moves up and down
        k=k+1
# force exit in exceptional circumstances
        if k>200: break

# unpack colour tree and sort
    colours=get_colourlist(coltree, 0)
    colours.sort(lambda C1, C2: cmp( \
                Y(C1[0],C1[1],C1[2]),  Y(C2[0],C2[1],C2[2])))

    return zip(*colours)

#-------------------------------------------------------------------------------------}

def RGB2LUV(r,g,b):
    rr=expand[r]; gg=expand[g]; bb=expand[b]
    X = rr*0.4124 + gg*0.3576 + bb*0.1805
    Y = rr*0.2126 + gg*0.7152 + bb*0.0722
    d=rr*3.6593 + gg*11.4432 + bb*4.115 + 1e-10
    U=4.0*X/d
    V=9.0*Y/d 
    Y = Y/100.0
    if Y > 0.008856: Y = Y**0.333333
    else:            Y = 7.787*Y + 0.1379
    Lstar = 116.0*Y - 16.0
    ustar = 13.0*Lstar*(U - 0.1978398)
    vstar = 13.0*Lstar*(V - 0.4683363)
    return (Lstar, ustar, vstar)

#-------------------------------------------------------------------------------------}

def simplex(x0, scale, F, ftol, debug):
    ok=True
    n=len(x0)
    p=[[x0[j]+scale[j]*(1-abs(cmp(i,j))) for j in range(0,n)] for i in range(0, n+1)]
    psum=[sum([p[i][j] for i in range(0,n+1)]) for j in range(0,n)]
    nfunc=0
    y=[F(p[i]) for i in range(0,n+1)]
    while True:
# get highest
        hi=y.index(max(y))
# set this value very low and get next highest
        (y[hi], ysave)=(-1e10, y[hi])
        nh=y.index(max(y))
        y[hi]=ysave
# get lowest
        lo=y.index(min(y))
        rtol=2*abs(y[hi]-y[lo])/(abs(y[hi]) + abs(y[lo]))
        if rtol<ftol: break
        if nfunc>5000:
            ok=False; break
        nfunc=nfunc+2
        (ytry, p, y, psum)=Try(p, y, psum, n, F, hi, -1.0)
        if ytry<=y[lo]:
            (ytry, p, y, psum)=Try(p, y, psum, n, F, hi, 2.0)
        elif ytry>=y[nh]:
            ysave=y[hi]
            (ytry, p, y, psum)=Try(p, y, psum, n, F, hi, 0.5)
            if ytry>=ysave:
                for i in range(0,n+1):
                    if i!=lo:
                        for j in range(0,n):
                            psum[j]=0.5*(p[i][j]+p[lo][j])
                            p[i][j]=psum[j]
                        y[i]=F(psum)
            nfunc= nfunc + n
            psum=[sum([p[i][j] for i in range(0,n+1)]) for j in range(0,n)]
        else: nfunc=nfunc-1
    return (p[lo], ok)

def Try(p, y, psum, n, F, hi, fac):
    fac1=(1.0-fac)/n
    fac2=fac1-fac
    ptry=[psum[j]*fac1 - p[hi][j]*fac2 for j in range(0,n)]
    ytry=F(ptry)
    if ytry<y[hi]:
        y[hi]=ytry
        for j in range(0,n):
            psum[j]+=ptry[j] - p[hi][j]
            p[hi][j]=ptry[j]
    return (ytry, p, y, psum)

#-------------------------------------------------------------------------------------}

def first_restore():
    global CH
    CH=R; scale=[0.02, 0.02]; x0=[1.0, 1.0]
    ((lamr, sigr), ok_r)=simplex(x0, scale, ideal_colour, 1e-4, True)
    CH=G; scale=[0.02, 0.02]; x0=[1.0, 1.0]
    ((lamg, sigg), ok_g)=simplex(x0, scale, ideal_colour, 1e-4, True)
    CH=B; scale=[0.02, 0.02]; x0=[1.0, 1.0]
    ((lamb, sigb), ok_b)=simplex(x0, scale, ideal_colour, 1e-4, True)
    if not (ok_r and ok_g and ok_b):
        pdb.gimp_message("The plug-in has failed to obtain a satisfactory restoration; the result shown may be poor.")
    return (lamr, lamg, lamb, sigr, sigg, sigb)

#-------------------------------------------------------------------------------------}

def ideal_colour(p):
    lamc, sigc = p
    measure=0.0
    for i in range(0,256):
        c=int(255*sigc*(colmap[CH][i]/255.0)**lamc)
        measure+=(c - ideal_col[i])**2
    return measure

#-------------------------------------------------------------------------------------}

def colour_balance(p):
    lamr, lamg, lamb, sigr, sigg, sigb  = p
    usum=0.0; vsum=0.0; wsum=0.0
    for i in range(0,256):
        r=int(255*sigr*(colmap[R][i]/255.0)**lamr)
        g=int(255*sigg*(colmap[G][i]/255.0)**lamg)
        b=int(255*sigb*(colmap[B][i]/255.0)**lamb)
        (Lstar, ustar, vstar)=RGB2LUV(r,g,b)
        s=ustar*ustar + vstar*vstar
        w=grey/(grey + s)
        usum=usum + w*ustar
        vsum=vsum + w*vstar
        wsum=wsum + w
    dist=(usum/wsum - u_off)**2 + (vsum/wsum - v_off)**2
    return dist

#-------------------------------------------------------------------------------------}

def levels_params(Lambda, Sigma, c_hi):
# gets parameters to the used in gimp levels command
    alpha=[0,0,0]
    s=[0,0,0]
    for C in [R,G,B]: s[C]=Sigma[C]*(c_hi[C]/255.0)**Lambda[C]
    smax=max(s)
    for C in [R,G,B]:
        alpha[C]=1.0/Lambda[C]
        s[C]=int(255.0*(s[C]/smax))
    return (alpha, c_hi, s)

###############################################################################

def restore(image, drawable, gammatarg, sat_choice, save):
    global R, G, B, ideal_col, expand, colmap, ctab
    global u_off, v_off, grey

# place where gimp stores levels data and debbug file
    gimp_levels_location="/home/gjd/.gimp-2.8/levels/restore"
    debug_file_location="/home/gjd/gimp/NewRestore/debug.txt"

    R=0; G=1; B=2
# set parameters here for colour balance, these seem about optimum
    grey=50.0; u_off=2.0;  v_off=2.0
# define colours of 'ideal' image'
    ideal_col=[int(255.0*(i/255.0)**gammatarg) for i in range(0,256)]
# make look up table for conversion to LUV, needs to be bigger than 256
    expand=[0.0 for c in range(0,320)]
    for c in range(0,256):
        C=c/255.0
        if C > 0.04045: C = ((C + 0.055)/1.055)**2.4
        else: C=C/12.92
        expand[c]=100.0*C

# look-up table for constructing list index
    ctab=[]
    for c in range(0,256):
        i=0
        mask=0x80
        for j in range(0,8):
            i=(i << 2) | (c & mask)
            mask = mask >> 1
        ctab.append(i)

# message on attempt to restore greyscale image
    if pdb.gimp_drawable_is_gray(drawable):
        pdb.gimp_message("NewRestore does not work on greyscale images.  Either use the levels command from the colors menu, or scan the image in colour (not convert from greyscale to RGB), restore it and then convert the result to greyscale.")
        return

# open debug file 
    debugfile=open(debug_file_location, "w")

# make small image for speed
    small_image=pdb.gimp_image_duplicate(image)
    width=pdb.gimp_image_width(image)
    height=pdb.gimp_image_height(image)
    pdb.gimp_image_scale(small_image, width/10, height/10)

# get colourmap of small image. colmap needs to be global
    cols=get_colourmapX(small_image)
    if cols==None: return
    (colmap, C_hi)=cols

# write original colourmap to debug file
    debugfile.write("#unrestored colmap\n")
    for i in range(0,256):
        debugfile.write(str(colmap[R][i])+","+str(colmap[G][i])+","+str(colmap[B][i])+"\n")

# first estimate of restoration parameters
    (lamr, lamg, lamb, sigr, sigg, sigb) = first_restore()
    LAMBDA1=[lamr, lamg, lamb]
    SIGMA1=[sigr, sigg, sigb]

# write parameters to debug file
    debugfile.write("#FIRST RESTORATION\n")
    debugfile.write("#LAMBDAS" + str(LAMBDA1)+"\n")
    debugfile.write("#SIGMAS" + str(SIGMA1)+"\n")

# get parameters for levels command
    (alpha1, m1, s1) = levels_params(LAMBDA1, SIGMA1, C_hi)

# write levels parameters to debug file
    debugfile.write("#alpha1" + str(alpha1) + "\n")
    debugfile.write("#m1" + str(m1) + "\n")
    debugfile.write("#s1" + str(s1) + "\n")

# restore the small image to get colourmap
    layer=pdb.gimp_image_get_active_layer(small_image)
    for C in [R,G,B]:
        pdb.gimp_levels(layer, C+1, 0, m1[C], alpha1[C], 0, s1[C])
# get colmap of restored image
    (colmap, junk)=get_colourmapX(small_image)
# delete small image since no longer required
    pdb.gimp_image_delete(small_image)

# write colour map of restored image to debug file
#    debugfile.write("#first restored colmap\n")
#    for i in range(0,256):
#        debugfile.write(str(colmap[R][i])+","+str(colmap[G][i])+","+str(colmap[B][i])+"\n")

# second stage to adjust colour balance
    scale=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    x0=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    fit=simplex(x0, scale, colour_balance, 1e-4, True)[0]
    lamr, lamg, lamb, sigr, sigg, sigb, = fit
    LAMBDA2=[lamr, lamg, lamb]; SIGMA2=[sigr, sigg, sigb]

# write balancing parameters to debug file
#    debugfile.write("#COLOUR BALANCING\n")
#    debugfile.write("#LAMBDA2"+str(LAMBDA2)+"\n")
#    debugfile.write("#SIGMA2"+str(SIGMA2)+"\n")

# combine parameters
    for C in [R,G,B]:
        SIGMA2[C]=SIGMA2[C]*SIGMA1[C]**LAMBDA2[C]
        LAMBDA2[C]=LAMBDA2[C]*LAMBDA1[C]

# write combined parameters to debug file
#    debugfile.write("#Combined parameters\n")
#    debugfile.write("#LAMBDAS"+str(LAMBDA2)+"\n")
#    debugfile.write("#SIGMAS"+str(SIGMA2)+"\n")

# get parameters for levels command
    (alpha2, m2, s2) = levels_params(LAMBDA2, SIGMA2, C_hi)

# write final levels parameters to debug file
#    debugfile.write("#final parameters\n")
#    debugfile.write("#alpha2 " + str(alpha2) + "\n")
#    debugfile.write("#m2     " + str(m2) + "\n")
#    debugfile.write("#s2     " + str(s2) + "\n")

# create restored image to display
    image_out=pdb.gimp_image_duplicate(image)
    layer=pdb.gimp_image_get_active_layer(image_out)

# restore new image and display it
    for C in [R,G,B]:
        pdb.gimp_levels(layer, C+1, 0, m2[C], alpha2[C], 0, s2[C])

    if sat_choice==0:
        pdb.gimp_image_set_filename(image_out, "Less_Saturated_Option")
        pdb.gimp_display_new(image_out)

# generate more saturate doption
    if sat_choice==1:
# make a duplicate and decompose into HSV space
        sat_image=pdb.gimp_image_duplicate(image_out)
        draw=pdb.gimp_image_get_active_drawable(sat_image)
        (hue, sat, value, junk)=pdb.plug_in_decompose(sat_image, draw, "HSV", 0)
        sat_draw=pdb.gimp_image_get_active_drawable(sat)
# get statistics to compute new saturation
        (mean, std_dev, median, pixels, count, percentile) \
            =pdb.gimp_histogram(sat_draw, 0, 0, 255)
# get an estimate of high saturation values
        maxsat=mean + 2.0*std_dev
# new layer to contain scaling values, needs to be division mode to boost sat
        sat_scale=pdb.gimp_layer_new(sat, width, height, 2, "sat", 100, 15) 
# factor by which to scale saturation
        fac=int(min(255, 255/min(1.5, 150.0/maxsat)))
        pdb.gimp_context_set_foreground((fac,fac,fac))
        pdb.gimp_drawable_fill(sat_scale, 0)
        pdb.gimp_image_insert_layer(sat, sat_scale, None, 0)
        pdb.gimp_image_merge_down(sat, sat_scale, 2)
        hsv=pdb.plug_in_compose(hue, draw, sat, value, junk, "HSV")
# delete unwanted images
        pdb.gimp_image_delete(image_out)
        pdb.gimp_image_delete(sat_image)
        pdb.gimp_image_delete(hue)
        pdb.gimp_image_delete(sat)
        pdb.gimp_image_delete(value)
# label and display of the more saturated option
        pdb.gimp_image_set_filename(hsv, "More_Saturated_Option")
        pdb.gimp_display_new(hsv)

#    debugfile.close()

# save parameters for future use
    if save==True:
        File=open(gimp_levels_location, "w")
        File.write("# GIMP levels tool settings generated by Restore2\n\n")
        File.write("(time 0)\n(time 0)\n")
#
        File.write("(channel value)\n")
        File.write("(gamma 1.0)\n")
        File.write("(low-input 0.0)\n")
        File.write("(high-input 1.0)\n")
        File.write("(low-output 0.0)\n")
        File.write("(high-output 1.0)\n")
        File.write("(time 0)\n")
#
        for (col,C) in [("red",R), ("green",G), ("blue",B)]:
            File.write("(channel " + col + ")\n")
            File.write("(gamma " + str(alpha2[C]) + ")\n")
            File.write("(low-input 0.0)\n")
            File.write("(high-input " + str(m2[C]/255.0) + ")\n")
            File.write("(low-output 0.0)\n")
            File.write("(high-output " + str(s2[C]/255.0) + ")\n")
            File.write("(time 0)\n")

        File.write("(channel alpha)\n")
        File.write("(gamma 1.0)\n")
        File.write("(low-input 0.0)\n")
        File.write("(high-input 1.0)\n")
        File.write("(low-output 0.0)\n")
        File.write("(high-output 1.0)\n")
        File.close()

register(
  "python-fu_Restore3",
  "Restore faded slide",
  "Restores scanned photos that have deteriorated with age.  Estimates the loss of dyes in the emulsion and restores the original values.  The Light-Dark parameter allows some choice about the result.  Some pictures are improved by having the saturation increased but this calculation is VERY slow and hence is optional", 
  "Geoff Daniell, gjd@lionhouse.plus.com",
  "Geoff Daniell",
  "2016",
  "<Image>/Restore/Restore3", "",
  [ (PF_SLIDER, "gammatarg", "Light-Dark",  1.0, (0.5, 2.0, 0.25) ),
    (PF_RADIO,  "sat_choice",  "Saturation\nadjustment", 1, (("Off",0), ("On",1)) ),
    (PF_TOGGLE, "save", "Save parameters", FALSE),
  ],
  [],
  restore
  )

main()
