#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Restore faded slides and photos.

Restores scanned slides that have deteriorated with age.
Estimates the loss of dyes in the emulsion and restores the original values.
The Degree of Restoration parameter adjusts the contrast and a yellow shift seems to improve the results.
The side-absorption correction can be left as additional layers.

################################################################################
# Note pyrestore4.py seems to perform  better than any earlier versions in the #
# vast majority of cases.                                                      #
################################################################################

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Geoff Daniell"
__contact__ = "geoff@lionhouse.plus.com"
__copyright__ = "Copyright 2008, Geoff Daniell"
__credits__ = ["Geoff Daniell", "Marc St. Onge"]
__deprecated__ = False
__email__ = "stongey@gmail.com"
__license__ = "GPLv3"
__maintainer__ = "Marc St. Onge"
__status__ = "Development"
__version__ = "0.0.1"


# standard library imports
import os
import sys

# 3rd party package imports
from numpy import *
import PIL
from PIL import Image


# GLOBAL CONSTANTS FIXING THE PROPERTIES OF THE RESTORED IMAGE

# Target average lightness. It is unwise to adjust this.
L0 = 50.0

# Initial values for extreme lightness in restored image.
L_top = 95.0
L_bot = 10.0

# Target for saturation, it is increased to this value if below.
sat_targ = 0.6

# Set debug value to get diagnostic information.
# debug = 0: no information; debug = 1: numerical values of restoration
# parameters are printed; debug = 2: intermediate images saved; debug = 3: both.
debug = 0

# White point in LUV colour space, these values are extremely critical
# for achieving a good white colour in the restored images

# easy rgb values D65 2deg
#    U0 = 0.1978398; V0 = 0.4683363

# easy rgb values D65 10 deg
#    U0 = 0.19786; V0 = 0.46955

# CIE values
#    U0 = 0.2009; V0 = 0.4610

# empirical values
# U0 = 0.2003; V0 = 0.4655

# new empirical values used in pyrestore4.py
U0 = 0.2005
V0 = 0.4665

# experimental values for pyrestore4.py
# U0 = 0.2010; V0 = 0.4670

# still more shifted to yellow?
# U0 = 0.2015; V0 = 0.4680

# further adjustments to improve pyrestore4 poorer than pyrestore3
# U0 = 0.2030; V0 = 0.4700

# -------------------------------------------------------------------------------------}


def Y(r, g, b):
    """Computes the colourXYZ Y parameter which is used to order the colours found."""
    rr = expand[r]
    gg = expand[g]
    bb = expand[b]
    Y = rr*0.2126 + gg*0.7152 + bb*0.0722 + 1e-10
    return Y

# -------------------------------------------------------------------------------------}


def enhance(im, p, q, uvfac):
    """Adjusts the lightness of the final image using the p and q values and the
    saturation using the uvfac parameter. See notes for explanation."""

    (w, h) = im.size

    # Decompress the image using the gamma value.
    RGB = im.point(xpand)

    # Split the image into R, G, and B monochrome images.
    RGB = list(RGB.split())

    # Put red, green and blue components into numpy arrays. From now on the
    # computation is done using numpy arrays.
    r = array(list(RGB[0].getdata()))
    g = array(list(RGB[1].getdata()))
    b = array(list(RGB[2].getdata()))

    # Get the XYZ colour components of the image as whole arrays. Z is not used
    X = r*0.4124 + g*0.3576 + b*0.1805
    Y = r*0.2126 + g*0.7152 + b*0.0722
    d = r*3.6593 + g*11.4432 + b*4.115 + 1e-10

    # Convert these to L*, U* and V*
    Lstar = where(Y > 0.8856, 24.9914*(Y**0.333333) - 16.0, 9.03296*Y)
    U = 4.0*X/d
    V = 9.0*Y/d 

    # Scale the lightness L* using searate parameters for L*<L0 and L*>L0
    newLstar = where(Lstar > L0, p[0]*Lstar + q[0], p[1]*Lstar + q[1])

    # Convert scaled L* to Y parameter.
    newY = where(newLstar <= 8.0, newLstar*0.110705, 6.40657e-5*(newLstar + 16.0)**3)

    # Apply saturation enhancement to U and V
    newU = uvfac*Lstar*(U - U0)/newLstar + U0
    newV = uvfac*Lstar*(V - V0)/newLstar + V0

    # Convert scaled U and V to X,Y,Z.
    X = 2.25*newY*newU/newV
    Y = newY
    Z = 0.25*newY*(12.0 - 3.0*newU - 20.0*newV)/newV

    # Convert to r,g,b
    r = X*3.2406 - Y*1.5372 - Z*0.4986
    g = -X*0.9689 + Y*1.8758 + Z*0.0415
    b = X*0.0557 - Y*0.2040 + Z*1.0570

    # Make new empty monochrome images
    newr = Image.new("L", (w, h))
    newg = Image.new("L", (w, h))
    newb = Image.new("L", (w, h))

    # Insert the pixels from the numpy arrays
    newr.putdata(r)
    newg.putdata(g)
    newb.putdata(b)

    # Merge the monochrome images into a RGB image
    newim = Image.merge("RGB", (newr, newg, newb))

    # Perform the gamma compression on the enhanced image
    newim = newim.point(cmpress)
    return newim


def xpand(c):
    """Function used in converting the compressed RGB values in the image to
    actual intensity values."""

    C = c / 255.0
    if C > 0.04045:
        C = ((C + 0.055)/1.055)**2.4
    else:
        C = C/12.92
    return int(100.0*C + 0.5)


def cmpress(c):
    """The reverse of xpand(c) above."""
    C = c/100.0
    if C > 0.0031308:
        C = 1.055*(C**0.4166666) - 0.055
    else:
        C = 12.92*C
    return int(C*255.0 + 0.5)


def RGB2LUV(r, g, b):
    """Converts rgb colour values to L* u* v*."""
    rr = expand[r]
    gg = expand[g]
    bb = expand[b]
    X = rr*0.4124 + gg*0.3576 + bb*0.1805
    Y = rr*0.2126 + gg*0.7152 + bb*0.0722
    # Z = rr*0.0193 + gg*0.1192 + bb*0.9505
    d = rr*3.6593 + gg*11.4432 + bb*4.115 + 1e-10
    U = 4.0*X/d
    V = 9.0*Y/d 
    Lstar = 24.9914*(Y**0.333333) - 16.0 if Y > 0.8856 else 9.03296*Y
    ustar = 13.0*Lstar*(U - U0)
    vstar = 13.0*Lstar*(V - V0)
    return (Lstar, ustar, vstar)


def LUV2RGB(Lstar, ustar, vstar):
    """Converts L* u* v* couour values to rgb."""
    u = ustar/(13.0*Lstar) + U0
    v = vstar/(13.0*Lstar) + V0
    Y = Lstar*0.110705 if Lstar <= 8.0 else 6.40657e-5*(Lstar + 16.0)**3
    X = 2.25*Y*u/v
    Z = 0.25*Y*(12.0 - 3.0*u - 20.0*v)/v
    r = cmpress(X*3.2406 - Y*1.5372 - Z*0.4986)
    g = cmpress(-X*0.9689 + Y*1.8758 + Z*0.0415)
    b = cmpress(X*0.0557 - Y*0.2040 + Z*1.0570)

    return (r, g, b)

# -------------------------------------------------------------------------------------}


def simplex(x0, scale, F, eps, debug):
    """Minimises function F in n dimensions using Nelder-Meade method starting
    from vector x0.  Parameter debug is not used.
    Exits if has found set of points with (|high| - |low|)/(|high| + |low|) < eps
    or number of function evaluations exceeds 5000.
    On exit returns coordinates of minimum and True or False, depending on
    number of function evaluations used."""

    ok = False

    # Get number of dimensions.
    n = len(x0)

    # Set up initial simplex.
    p = [x0[:] for i in range(0, n+1)]
    for i in range(0, n):
        p[i][i] += scale[i]
    psum = [sum([p[i][j] for i in range(0, n+1)]) for j in range(0, n)]
    nfunc = 0

    # Get function value at vertices
    y = [F(p[i]) for i in range(0, n+1)]
    while True:
        # Get highest
        hi = y.index(max(y))

        # Set this value very low and get next highest
        (y[hi], ysave) = (-1e10, y[hi])
        next_hi = y.index(max(y))
        y[hi] = ysave

        # Get lowest
        lo = y.index(min(y))

        # Test for convergence
        if 2.0*abs(y[hi] - y[lo])/(abs(y[hi]) + abs(y[lo])) < eps:
            ok = True
            break

        # Exit if failed to converge
        if nfunc > 5000:
            break

        nfunc += 2
        (ynew, p, y, psum) = trial(p, y, psum, n, F, hi, -1.0)

        # If new point better try going further
        if ynew <= y[lo]:
            (ynew, p, y, psum) = trial(p, y, psum, n, F, hi, 2.0)
        # If the new point is worse than the next highest...
        elif ynew >= y[next_hi]:
            ysave = y[hi]
            (ynew, p, y, psum) = trial(p, y, psum, n, F, hi, 0.5)

            # If getting nowhere shrink the simplex
            if ynew >= ysave:

                # Loop over vertices keeping the lowest point unchanged
                for i in range(0, n+1):
                    if i == lo:
                        continue
                    pnew = [0.5*(p[i][j] + p[lo][j]) for j in range(0, n)]
                    p[i] = pnew
                    y[i] = F(pnew)
            nfunc += n
            psum = [sum([p[i][j] for i in range(0, n+1)]) for j in range(0, n)]
        else:
            nfunc -= 1
    return (p[lo], ok)


def trial(p, y, psum, n, F, hi, dist):
    """Compute point pnew along line from p[hi] to centroid excluding p[h1]."""
    a = (1.0 - dist) / n
    b = a - dist
    pnew = [a*psum[j] - b*p[hi][j] for j in range(0, n)]
    ynew = F(pnew)

    # If improvement accept and adjust psum
    if ynew < y[hi]:
        y[hi] = ynew
        psum = [psum[j] + (pnew[j] - p[hi][j]) for j in range(0, n)]
        p[hi] = pnew
    return (ynew, p, y, psum)

# -------------------------------------------------------------------------------------}


def get_colours(im):
    """Makes list of distict colours that occur in image."""
    pixels = list(im.getdata())
    colours = []

    # Allow space for 4096 separate colours
    N = 4096
    rgb = zeros(N, int)
    for (r, g, b) in pixels:

        # Construct n using 4 most significant bits of r, g and b.
        n = ((r << 4) & 0xf00) | (g & 0xf0) | (b >> 4)

        # If this colour has not yet occurred flag that it has and add colour to list
        if not rgb[n]:
            rgb[n] = 1
            colours.append((r, g, b))

    # The array rgb is no longer needed
    del rgb
    return colours

# -------------------------------------------------------------------------------------}


def first_restore():
    """Optimises lambda and sigma separately for each colour channel."""
    global CH

    CH = R
    scale = [0.02, 0.02]
    x0 = [1.0, 1.0]
    ((lamr, sigr), ok_r) = simplex(x0, scale, ideal_colour, 1e-4, False)

    CH = G
    scale = [0.02, 0.02]
    x0 = [1.0, 1.0]
    ((lamg, sigg), ok_g) = simplex(x0, scale, ideal_colour, 1e-4, False)

    CH = B
    scale = [0.02, 0.02]
    x0 = [1.0, 1.0]
    ((lamb, sigb), ok_b) = simplex(x0, scale, ideal_colour, 1e-4, False)

    if not (ok_r and ok_g and ok_b):
        print "The program has failed to obtain a satisfactory restoration; the result shown may be poor."

    return (lamr, lamg, lamb, sigr, sigg, sigb)

# -------------------------------------------------------------------------------------}


def ideal_colour(p):
    """Calculates measure of misfit between actual colours and ideal colours.
    This is to be minimised in the first stage of restoration."""

    lamc, sigc = p
    measure = 0.0
    for i in range(0, numcols):
        c = int(255*sigc*(colmap[CH][i]/255.0)**lamc)
        measure += (c - ideal_col[i])**2
    return measure

# -------------------------------------------------------------------------------------}


def func_restore(p):
    """Calculates the qunatity to be minimised in the final restored image."""
    global minimum, checkL

    (lamr, lamg, lamb, sigr, sigg, sigb) = p
    L = []
    U = []
    V = []
    for C in colours:
        (r, g, b) = C

        # 'Restores'rgb values with lambda and sigma, see notes.
        r = int(255*sigr*(r/255.0)**lamr)
        g = int(255*sigg*(g/255.0)**lamg)
        b = int(255*sigb*(b/255.0)**lamb)

        # Make sure that no r, g or b values are outside permitted range.
        r = min(r, 255)
        g = min(g, 255)
        b = min(b, 255)

        r = max(r, 0)
        g = max(g, 0)
        b = max(b, 0)

        # Convert the rgb values to L*, U* and V* and append to lists.
        (Lstar, ustar, vstar) = RGB2LUV(r, g, b)
        U.append(ustar)
        V.append(vstar)
        L.append(Lstar)

    # Convert the lists to numpy arrays for fast processing.
    L = array(L)
    U = array(U)
    V = array(V)

    # Set nLUV to the length of the lists.
    nLUV = float(len(U))

    # Compute the average L and its offset from specified value L0.
    Lbar = sum(L) / nLUV
    Loff = Lbar - L0

    # Make an array of brightness values for diagnostic purposes.
    Ldist = zeros(14, int)
    for lightness in L:
        l = int(lightness) >> 3
        Ldist[l] += 1

    # Export Ldist for diagnostic purposes.
    checkL = Ldist

    # Compute parameters for function to me minimised, see notes.
    Ldash = L/L0 - 1.0
    UL0 = sum(U)/nLUV
    UL1 = sum(U*Ldash)/nLUV
    UL2 = sum(U*Ldash*Ldash)/nLUV

    u0 = UL0
    u1 = 3.0*UL1
    u2 = 2.5*(3.0*UL2 - UL0)

    VL0 = sum(V)/nLUV
    VL1 = sum(V*Ldash)/nLUV
    VL2 = sum(V*Ldash*Ldash)/nLUV

    v0 = VL0
    v1 = 3.0*VL1
    v2 = 2.5*(3.0*VL2 - VL0)

    minimum = [Loff**2, u0**2, 0.3333333*u1**2, 0.2*u2**2, v0**2, 0.3333333*v1**2, 0.2*v2**2]

    return sum(minimum)

# -------------------------------------------------------------------------------------}


def combine_params(fit1, fit2):
    """Given two sets of lambda and sigma parameters, compute a single set that
    will produce the same outcome."""

    lam1 = fit1[0:3]
    sig1 = fit1[3:6]

    lam2 = fit2[0:3]
    sig2 = fit2[3:6]

    lam = [0, 0, 0]
    sig = [0, 0, 0]

    for C in [R, G, B]:
        sig[C] = sig2[C]*sig1[C]**lam2[C]
        lam[C] = lam1[C]*lam2[C]

    return [lam[0], lam[1], lam[2], sig[0], sig[1], sig[2]]

# -------------------------------------------------------------------------------------}


def get_enhance_params(im):
    """From a given 'restored' image compute the average saturation and a
    lightness range that includes most of the pixels.

    This code repeats the computation of the list of colours but with additional
    code to get the saturation."""

    sum_sat = 0.0
    nLUV = 0
    light = zeros(100, int)
    N = 4096
    rgb = zeros(N, int)
    pixels = list(im.getdata())
    for (r, g, b) in pixels:
        (Lstar, ustar, vstar) = RGB2LUV(r, g, b)
        light[int(Lstar)] += 1
        n = ((r << 4) & 0xf00) | (g & 0xf0) | (b >> 4)
        if not rgb[n]:
            rgb[n] = 1
            sum_sat += sqrt(ustar**2 + vstar**2) / Lstar
            nLUV += 1
    del rgb

    # Get average saturation over whole image.
    sat = sum_sat/nLUV

    # Upper percentile of lightness.
    Ltail = int(0.025*sum(light))
    Lmax = len(light)
    while sum(light[Lmax:]) < Ltail:
        Lmax -= 1

    # Lower percentile of lightness
    Lmin = 0
    while sum(light[0:Lmin]) < Ltail:
        Lmin += 1

    return (Lmax, Lmin, sat)

# -------------------------------------------------------------------------------------}


def restore_image(image, fit):
    """Restores image using lambda and sigma values in fit."""
    (lamr, lamg, lamb, sigr, sigg, sigb) = fit
    Lambda = [lamr, lamg, lamb]; Sigma = [sigr, sigg, sigb]

    # Split the RGB image into three monochrome images.
    RGB = list(image.split())

    # Apply restoration operation to each channel.
    for C in [R, G, B]:
        RGB[C] = RGB[C].point(lambda I: int(255*Sigma[C]*(I/255.0)**Lambda[C]+0.5))

    # Recombine the monochrome images.
    return Image.merge("RGB", (RGB[0], RGB[1], RGB[2]))

###############################################################################


def restore(im, photoname):
    """Restores image, photoname is simply used for constructing directory names."""
    global colours, colmap, ideal_col, numcols

    # Create a small image for speed.
    (width, height) = im.size
    small_image = im.resize((width/8, height/8), resample=PIL.Image.NEAREST)

    if debug & 2:
        newfilename = os.path.join(directory, "restored", "before" + photoname)
        small_image.save(newfilename, icc_profile=im.info.get('icc_profile'))

    # Get colourmap of small image.
    colours = get_colours(small_image)

    # Sort the colours according to their Y paramtere or lightness
    colours.sort(lambda C1, C2: cmp(Y(C1[0], C1[1], C1[2]),  Y(C2[0], C2[1], C2[2])))
    numcols = len(colours)

    # Rearrange the colourmap into three, one for each of R, G and B.
    colmap = zip(*colours)

    # Define colours of 'ideal' image.
    gammatarg = 1.0
    ideal_col = [int(255.0*(i/float(numcols))**gammatarg) for i in range(0, numcols)]

    # First restoration of small image.
    fit = first_restore()
    first_fit = fit
    if debug & 1: 
        print "first restore"
        print 'lambda parameters %6.3g%6.3g%6.3g' % (fit[0], fit[1], fit[2])
        print 'sigma parameters %6.3g%6.3g%6.3g' % (fit[3], fit[4], fit[5])

    # Restore the small image.
    restored_small = restore_image(small_image, first_fit)

    # Save the intermediate result.
    if debug & 2:
        newfilename = os.path.join(directory, "restored", "restored" + photoname)
        restored_small.save(newfilename, icc_profile=im.info.get('icc_profile'))

    # Get the colourmap of the restored image.
    colours = get_colours(restored_small)
    colours.sort(lambda C1, C2: cmp(Y(C1[0], C1[1], C1[2]),  Y(C2[0], C2[1], C2[2])))
    numcols = len(colours)
    colmap = zip(*colours)

    # Initial values of lambda and sigma parameters for second restoration.
    scale = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    x0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    fit = simplex(x0, scale, func_restore, 1e-4, False)[0]
    second_fit = fit

    if debug & 1:
        print "second_restore"
        print 'lambda parameters %6.3g%6.3g%6.3g' % (fit[0], fit[1], fit[2])
        print 'sigma parameters %6.3g%6.3g%6.3g' % (fit[3], fit[4], fit[5])

    # Compute a crude measure of the success of the restoration.
    m = minimum
    if sum(minimum) < 2.0:
        outcome = "good"
    if sum(minimum) >= 2.0:
        outcome = "poor"
    if sum(minimum) > 5.0:
        outcome = "bad"

    # Check for impossible values.
    if min(fit) < 0.0:
        outcome = "fail"

    if debug & 1:
        print 'restoration:', outcome
        print 'u^2 values %8.3g  %8.3g  %8.3g' % (m[1], m[2], m[3])
        print 'v^2 values %8.3g  %8.3g  %8.3g' % (m[4], m[5], m[6])
        print 'lightness offset %6.3g' % m[0]
        print 'lightness distribution', checkL

    # Perform second restoration on small image
    second_restored_small = restore_image(restored_small, fit)

    # Save the intermediate result
    if debug & 2:
        newfilename = os.path.join(directory, "restored", "second_restored" + photoname)
        second_restored_small.save(newfilename, icc_profile=im.info.get('icc_profile'))

    # Combine the parameters of the two restorations
    combined_fit = combine_params(first_fit, second_fit)

    fit = combined_fit
    if debug & 1:
        print "combined parameters"
        print 'lambda parameters %6.3g%6.3g%6.3g' % (fit[0], fit[1], fit[2])
        print 'sigma parameters %6.3g%6.3g%6.3g' % (fit[3], fit[4], fit[5])

    # Restore the small image with the combined parameters
    combined_restored_small = restore_image(small_image, combined_fit)

    # Save the intermediate result
    if debug & 2:
        newfilename = os.path.join(directory, "restored", "combined_restored" + photoname)
        combined_restored_small.save(newfilename, icc_profile=im.info.get('icc_profile'))

    # Get the saturation and lightness range to enhance the small and main image.
    (Lmax, Lmin, sat) = get_enhance_params(combined_restored_small)

    if debug & 1:
        print 'Max, Min L and saturation %3d%3d%6.3g' % (Lmax, Lmin, sat)

    # Construct values to be used in enhance operation.
    L_hi = 0.5*(L_top + Lmax)
    L_lo = max(0.5*(L_bot + Lmin), L_bot)
    p = ((L_hi - L0)/(Lmax - L0), (L_lo - L0)/(Lmin - L0))
    q = (L0*(Lmax - L_hi)/(Lmax - L0), L0*(Lmin - L_lo)/(Lmin - L0) + 0.1)
    if sat < sat_targ:
        uvfac = sat_targ / sat
    else:
        uvfac = 1.0

    # Enhance the small image.
    enhanced_small = enhance(combined_restored_small, p, q, uvfac)

    # Save the intermediate result.
    if debug & 2:
        newfilename = os.path.join(directory, "restored", "enhanced" + photoname)
        enhanced_small.save(newfilename, icc_profile=im.info.get('icc_profile'))

    # Restore and enhance the main image.
    restored_image = restore_image(im, combined_fit)
    enhanced_image = enhance(restored_image, p, q, uvfac)
    return enhanced_image
#    return None

#############################################################################


# default parameters
directory = os.getcwd()

# save current directory
savedir = directory

# The original parsing of arguments has been left unchanged although not used
# gammatarg = 1.0
# sat_choice = 1
# parse parameters
# for param in sys.argv:
#    Dir = param.find("dir=")
#    if Dir >= 0: directory=param[Dir+4:]
#    gam = param.find("Light-Dark=")
#    if gam >= 0: gammatarg=float(param[gam+11:])
#    sat = param.find("saturate=")
#    if sat >= 0: sat_choice=eval(param[sat+9:])

# print directory, gammatarg, sat_choice

R = 0
G = 1
B = 2

# set parameters here for colour balance, these seem about optimum
# grey = 50.0; u_off = 2.0;  v_off = 2.0

# Make look up table for conversion to LUV, needs to be bigger than 256
expand = [0.0 for c in range(0, 360)]
for c in range(0, 256):
    C = c/255.0
    if C > 0.04045:
        C = ((C + 0.055)/1.055)**2.4
    else:
        C = C/12.92
    expand[c] = 100.0*C

# Make a list of files to process
os.chdir(directory)
filelist = os.listdir(directory)
jpegs = []
JPEGS = []
tiffs = []
TIFFS = []

for File in filelist:
    if File.find(".jpg") > 0:
        jpegs.append(File)
    if File.find(".JPG") > 0:
        JPEGS.append(File)
    if File.find(".tiff") > 0:
        tiffs.append(File)
    if File.find(".TIFF") > 0:
        TIFFS.append(File)

# In windows the file searching is NOT case sensitive.
if JPEGS != jpegs:
    jpegs += JPEGS
if TIFFS != tiffs:
    tiffs += TIFFS

for photo in jpegs + tiffs:
    # Strip off directory name and .jpg to get file name.
    photoname = os.path.split(photo)[1]
    print "\n"
    print photoname

    # Open photo
    im = Image.open(photoname)

    # Restore the image.
    restored_image = restore(im, photoname)

    # Ignore the result if restoration failed
    if restored_image is None:
        continue

    # Save file to subdirectory 'restored'..
    newfilename = os.path.join(directory, "restored", photoname)
    restored_image.save(newfilename, icc_profile=im.info.get('icc_profile'))

# Return to saved directory.
os.chdir(savedir)
