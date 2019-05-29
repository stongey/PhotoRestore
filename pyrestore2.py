import os

import sys
import PIL
from PIL import Image, ImageStat

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

def Y(r,g,b):
    rr=expand[r]; gg=expand[g]; bb=expand[b]
    Y = rr*0.2126 + gg*0.7152 + bb*0.0722 + 1e-10
    return Y

#-------------------------------------------------------------------------------------}

def get_colourmap(small_image):
    copy_small=small_image.copy()
    copy_small=copy_small.quantize(colors=256, method=0)
    colourmap=copy_small.getpalette()
# unpack colourmap and get parameters for extrapolation
    colmap=[0,0,0]; C_hi=[0,0,0]
    for C in [R,G,B]:
        colmap[C]=colourmap[slice(C,768,3)]
# PIL colour list is in reverse order to gimp
        colmap[C]=colmap[C][::-1]
        s0=sum(colmap[C])
        s1=sum([i*colmap[C][i] for i in range(0,256)])
        C_hi[C]=int(0.5*(max(colmap[C]) + 3.04e-5*(3.0*s1 - 254.0*s0)))
        C_hi[C]=min([C_hi[C], 255])
    return (colmap, C_hi)

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
    (colmap, C_hi)=get_colourmap(small_image)
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
    (colmap, junk)=get_colourmap(restored_small)
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
    if sat_choice:
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

print directory, gammatarg, sat_choice

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

# save current directory
savedir=directory
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
# restore the image
    restored_image=restore(im)
# Save file
    newfilename=os.path.join(directory, "restored", photoname)
    restored_image.save(newfilename, icc_profile=im.info.get('icc_profile'))

# return to saved directory
os.chdir(savedir)

