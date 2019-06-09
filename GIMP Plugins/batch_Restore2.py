#! /usr/bin/env python
# version 1.0 16/4/16
from gimpfu import *
from math import *
from os import path

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

def get_colourmap(image):
# make a copy for trial restoration
    local_copy=pdb.gimp_image_duplicate(image)
# convert to indexed, get colour map and delete copy
    pdb.gimp_image_convert_indexed(local_copy,0,0,256,FALSE,TRUE,"mypal")
    (num_bytes, colourmap)=pdb.gimp_image_get_colormap(local_copy)
    pdb.gimp_image_delete(local_copy)
# unpack colourmap and get parameters for extrapolation
    colmap=[0,0,0]; C_hi=[0,0,0]
    for C in [R,G,B]:
        colmap[C]=colourmap[slice(C,768,3)]
        s0=sum(colmap[C])
        s1=sum([i*colmap[C][i] for i in range(0,256)])
        C_hi[C]=int(0.5*(max(colmap[C]) + 3.04e-5*(3.0*s1 - 254.0*s0)))
        C_hi[C]=min([C_hi[C], 255])
    return (colmap, C_hi)

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
    global R, G, B, ideal_col, expand, colmap
    global u_off, v_off, grey
# place where gimp stores levels data and debug file
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

# message on attempt to restore greyscale image
    if pdb.gimp_drawable_is_gray(drawable):
        pdb.gimp_message("NewRestore does not work on greyscale images.  Either use the levels command from the colors menu, or scan the image in colour (not convert from greyscale to RGB), restore it and then convert the result to greyscale.")
        return

# open debug file 
#    debugfile=open(debug_file_location, "w")

# make small image for speed
    small_image=pdb.gimp_image_duplicate(image)
    width=pdb.gimp_image_width(image)
    height=pdb.gimp_image_height(image)
    pdb.gimp_image_scale(small_image, width/10, height/10)

# get colourmap of small image. colmap needs to be global
    (colmap, C_hi)=get_colourmap(small_image)

# write original colourmap to debug file
#    debugfile.write("#unrestored colmap\n")
#    for i in range(0,256):
#        debugfile.write(str(colmap[R][i])+","+str(colmap[G][i])+","+str(colmap[B][i])+"\n")

# first estimate of restoration parameters
    (lamr, lamg, lamb, sigr, sigg, sigb) = first_restore()
    LAMBDA1=[lamr, lamg, lamb]
    SIGMA1=[sigr, sigg, sigb]

# write parameters to debug file
#    debugfile.write("#FIRST RESTORATION\n")
#    debugfile.write("#LAMBDAS" + str(LAMBDA1)+"\n")
#    debugfile.write("#SIGMAS" + str(SIGMA1)+"\n")

# get parameters for levels command
    (alpha1, m1, s1) = levels_params(LAMBDA1, SIGMA1, C_hi)

# write levels parameters to debug file
#    debugfile.write("#alpha1" + str(alpha1) + "\n")
#    debugfile.write("#m1" + str(m1) + "\n")
#    debugfile.write("#s1" + str(s1) + "\n")

# restore the small image to get colourmap
    layer=pdb.gimp_image_get_active_layer(small_image)
    for C in [R,G,B]:
        pdb.gimp_levels(layer, C+1, 0, m1[C], alpha1[C], 0, s1[C])
# get colmap of restored image
    (colmap, junk)=get_colourmap(small_image)
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
#        pdb.gimp_display_new(image_out)
        return image_out
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
#        pdb.gimp_display_new(hsv)
        return hsv

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


def batch_Restore2(directory, gammatarg, sat_choice):
    global debugfile, save
    save=False
#    debugfile=open(debug_file_location, "w")
# Get list of jpeg and JPEGS files in directory
    (num,jpegs)=pdb.file_glob(path.join(directory, "*.jpg"), 0)
    (num,JPEGS)=pdb.file_glob(path.join(directory, "*.JPG"), 0)
    (num,tiffs)=pdb.file_glob(path.join(directory, "*.tiff"), 0)
    (num,TIFFS)=pdb.file_glob(path.join(directory, "*.TIFF"), 0)
#    debugfile.write(str(jpegs)+"\n")
#    debugfile.write(str(JPEGS)+"\n")
# in windows the file searching is NOT case sensitive
    if JPEGS!=jpegs: jpegs+=JPEGS
    if TIFFS!=tiffs: tiffs+=TIFFS
    for photo in jpegs+tiffs:
        photoname=path.split(photo)[1]
# Strip off directory name and .jpg to get file name	
#        debugfile.write(str(photo)+"\n")
# Open photo
        image=pdb.gimp_file_load(photo,photo)
        drawable=pdb.gimp_image_get_active_drawable(image)
        new_image=restore(image, drawable, gammatarg, sat_choice, False)
# Save file
        newfilename=path.join(directory, "restored", photoname)
        new_drawable=pdb.gimp_image_get_active_drawable(new_image)
        pdb.gimp_file_save(new_image, new_drawable, newfilename, newfilename)
        pdb.gimp_image_delete(new_image)
        pdb.gimp_image_delete(image)

#    debugfile.close()

register(
  "python-fu_Batch_Restore2",
  "Restore faded slide",
  "Restores scanned photos that have deteriorated with age.  Estimates the loss of dyes in the emulsion and restores the original values.  The Light-Dark parameter allows some choice about the result.  Some pictures are improved by having the saturation increased", 
  "Geoff Daniell, gjd@lionhouse.plus.com",
  "Geoff Daniell",
  "2016",
  "<Toolbox>/Xtns/Batch Restore2", "",
  [ (PF_STRING, "arg0", "directory",          R"/home/gjd/restore"),
    (PF_SLIDER, "gammatarg", "Light-Dark",  1.0, (0.5, 2.0, 0.25) ),
    (PF_RADIO,  "sat_choice",  "Saturation\nadjustment", 1, (("Off",0), ("On",1)) ),
  ],
  [],
  batch_Restore2
  )

main()
