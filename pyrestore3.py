import os
import sys
import PIL
from PIL import Image, ImageStat
from array import array

# 9/8/16 error message on <256 colours

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

def get_colourmapX(image):
# make a copy for trial restoration
    local_copy=image.copy()
    colourmap=palette(local_copy)
    if colourmap==None: return None
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
    pxls=list(small.getdata())
# create lists to contain counts
    p0=[]
    Nlevels=6
    for level in range(0, Nlevels + 1):
        p0.append(array('i', [0 for i in range(0, 8**level)]))
# count pixels in levels
    p=p0
    it=iter(pxls)
    try:
        while True:
            (r,g,b)=next(it)
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
                print "Less than 256 distinct colours, impossible to restore"
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

#def get_colourmap(small_image):
#
#    copy_small=small_image.copy()
#    copy_small=copy_small.quantize(colors=256, method=0)
#    colourmap=copy_small.getpalette()
# unpack colourmap and get parameters for extrapolation
#    colmap=[0,0,0]; C_hi=[0,0,0]
#    for C in [R,G,B]:
#        colmap[C]=colourmap[slice(C,768,3)]
# PIL colour list is in reverse order to gimp
#        colmap[C]=colmap[C][::-1]
#        s0=sum(colmap[C])
#        s1=sum([i*colmap[C][i] for i in range(0,256)])
#        C_hi[C]=int(0.5*(max(colmap[C]) + 3.04e-5*(3.0*s1 - 254.0*s0)))
#            C_hi[C]=min([C_hi[C], 255])
#    return (colmap, C_hi)

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

#-------------------------------------------------------------------------------------}

def adjust_levels(image, hi_in, gamma, hi_out):
    RGB=list(image.split())
    for C in [R,G,B]:
        alpha=1.0/float(gamma[C])
        float_hi_in=float(hi_in[C])
        RGB[C]=RGB[C].point(lambda I: int(hi_out[C]*(I/float_hi_in)**alpha+0.5))
    return Image.merge("RGB", tuple(RGB))

###############################################################################

def restore(im):
    global colmap
    (width, height)=im.size
    small_image=im.resize((width/8, height/8), resample=PIL.Image.NEAREST)
# colourmap of small image
    cols=get_colourmapX(small_image)
    if cols==None: return None
    (colmap, C_hi)=cols
#    print colmap
# first estimate of restoration parameters
    (lamr, lamg, lamb, sigr, sigg, sigb) = first_restore()
    LAMBDA1=[lamr, lamg, lamb]; SIGMA1=[sigr, sigg, sigb]
#
#    print C_hi
#    print "LAMBDA1, SIGMA1", LAMBDA1, SIGMA1
#
# get parameters for levels command
    (alpha1, m1, s1) = levels_params(LAMBDA1, SIGMA1, C_hi)
# restore the small image to get colourmap
    restored_small=adjust_levels(small_image, m1, alpha1, s1)
#
#    print "m1, alpha1, s1", m1, alpha1, s1
#
# get colmap of restored image
    (colmap, junk)=get_colourmapX(restored_small)
# second stage to adjust colour balance
    scale=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    x0=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    fit=simplex(x0, scale, colour_balance, 1e-4, True)[0]
    lamr, lamg, lamb, sigr, sigg, sigb, = fit
    LAMBDA2=[lamr, lamg, lamb]; SIGMA2=[sigr, sigg, sigb]
#
#    print "LAMBDA2, SIGMA2", LAMBDA2, SIGMA2
#
# combine parameters
    for C in [R,G,B]:
        SIGMA2[C]=SIGMA2[C]*SIGMA1[C]**LAMBDA2[C]
        LAMBDA2[C]=LAMBDA2[C]*LAMBDA1[C]

# get parameters for levels command
    (alpha2, m2, s2) = levels_params(LAMBDA2, SIGMA2, C_hi)
#
#    print "m2, alpha2, s2", m2, alpha2, s2
#
# restore main image
    restored_image=adjust_levels(im, m2, alpha2, s2)

# generate more saturated option
    if sat_choice==1:
        HSVimage=restored_image.convert("HSV")
        HSV=list(HSVimage.split())
# get statistics to compute new saturation
        stats=ImageStat.Stat(HSV[1])
        mean=stats.mean[0]
        std_dev=stats.stddev[0]
# get an estimate of high saturation values
        maxsat=mean + 2.0*std_dev
        fac=1.0/min(1.0, 1.0/min(1.5, 150.0/maxsat))
# increase the saturation
        HSV[1]=HSV[1].point(lambda I: int(fac*I+0.5))
        more_saturated=Image.merge("HSV", tuple(HSV))
        more_saturated=more_saturated.convert("RGB")
        return more_saturated
    else: return restored_image

#############################################################################

#default parameters
directory=os.getcwd()
# save current directory
savedir=directory

gammatarg=1.0
sat_choice=1
#parse parameters
for param in sys.argv:
    Dir=param.find("dir=")
    if Dir>=0: directory=param[Dir+4:]
    gam=param.find("Light-Dark=")
    if gam>=0: gammatarg=float(param[gam+11:])
    sat=param.find("saturate=")
    if sat>=0: sat_choice=eval(param[sat+9:])

#print directory, gammatarg, sat_choice

R=0; G=1; B=2
# set parameters here for colour balance, these seem about optimum
grey=50.0; u_off=2.0;  v_off=2.0

# define colours of 'ideal' image'
ideal_col=[int(255.0*(i/255.0)**gammatarg) for i in range(0,256)]

# make look up table for conversion to LUV, needs to be bigger than 256
expand=[0.0 for c in range(0,360)]
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

os.chdir(directory)
filelist=os.listdir(directory)
jpegs=[]; JPEGS=[]; tiffs=[]; TIFFS=[]
for File in filelist:
    if File.find(".jpg")>0:  jpegs.append(File)
    if File.find(".JPG")>0:  JPGES.append(File)
    if File.find(".tiff")>0: tiffs.append(File)
    if File.find(".TIFF")>0: TIFFS.append(File)

# in windows the file searching is NOT case sensitive
if JPEGS!=jpegs: jpegs+=JPEGS
if TIFFS!=tiffs: tiffs+=TIFFS

for photo in jpegs+tiffs:
# Strip off directory name and .jpg to get file name
    photoname=os.path.split(photo)[1]
    print photoname
# Open photo
    im=Image.open(photoname)
#    print str(im.info.get('icc_profile'))
# restore the image
    restored_image=restore(im)
    if restored_image==None: continue
# Save file
    newfilename=os.path.join(directory, "restored", photoname)
    restored_image.save(newfilename, icc_profile=im.info.get('icc_profile'))

# return to saved directory
os.chdir(savedir)

