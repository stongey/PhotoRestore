import os
import sys
import PIL
from PIL import Image, ImageStat
from array import array

# Code was originally written by Geoff Daniell
# 25/5/17 improvements to code, no change in output.

###############################################################################
# Note that pyrestore2.py is preferred over pyrestore3.py since the former    #
# uses the colour quantisation algorithm built into the PIL library.  This    #
# does not lead to identical results to the gimp plug-in Restore2.py. If, for #
# any reason, it is desirable to have the same results using the stand-alone  #
# code and the gimp plug-in one can use pyrestore3.py and Restore3.py.  The   #
# latter is also provided to to keep the plug-in working if in the future the #
# colour indexed mode is removed from GIMP.                                   #
###############################################################################


def RGB2LUV(r, g, b):
    """Converts rgb colour values to L* u* v*."""
    rr = expand[r]; gg = expand[g]; bb = expand[b]
    X = rr*0.4124 + gg*0.3576 + bb*0.1805
    Y = rr*0.2126 + gg*0.7152 + bb*0.0722
    d = rr*3.6593 + gg*11.4432 + bb*4.115 + 1e-10
    U = 4.0*X/d
    V = 9.0*Y/d 
    Y = Y/100.0
    if Y > 0.008856: Y = Y**0.333333
    else:            Y = 7.787*Y + 0.1379
    Lstar = 116.0*Y - 16.0
    ustar = 13.0*Lstar*(U - 0.1978398)
    vstar = 13.0*Lstar*(V - 0.4683363)
    return (Lstar, ustar, vstar)

# -------------------------------------------------------------------------------------}


def get_my_colourmap(image):
    """Gets best 256 colours in image using own algorithm described in the appendix
    to restore2.pdf.  See this document for details of the extrapolation."""

    colmap = my_palette(image)
    # Exit with None if palette fails.
    if colmap == None: return None
    # Get parameters for extrapolation.
    C_hi = [0, 0, 0]
    for C in [R, G, B]:
        s0 = sum(colmap[C])
        s1 = sum([i*colmap[C][i] for i in range(0, 256)])
        C_hi[C] = int(0.5*(max(colmap[C]) + 3.04e-5*(3.0*s1 - 254.0*s0)))
        C_hi[C] = min([C_hi[C], 255])

    return (colmap, C_hi)

##############################################################################
# The following group of functions are used by the function my_palette which
# computes an optimum palette of 256 colours using an octree method.
##############################################################################


def Y(r, g, b):
    """Computes the Y value from rgb values, used for sorting colours by brightness."""
    rr = expand[r]; gg = expand[g]; bb = expand[b]
    Y = rr*0.2126 + gg*0.7152 + bb*0.0722 + 1e-10
    return Y

# -------------------------------------------------------------------------------------}


def rgb2n(r, g, b):
    """A colour is represented by a single integer obtained by interleaving the
    bits from the rgb values so that similar colours map onto close numbers.
    It is the  box number at level 6 computed from (truncated) rgb values."""
    return ((ctab[r] << 2) | (ctab[g] << 1) | ctab[b]) >> 6

# -------------------------------------------------------------------------------------}


def interpolate(c):
    """Interpolation in ctab is used to convert colour number to rgb."""
    i = 0; j = 128
    while j:
        if c >= ctab[i+j]: i += j
        j = j >> 1
    return i

# -------------------------------------------------------------------------------------}


def n2rgb(n, level):
    """Computes rgb values at centre of box from colour number. Returns tuple of rgb values."""

    # Shift colour number to adjust for level in tree.
    nn = n << (24 - 3*level)
    # Unpack in r and g parts.
    nr = (nn & 0x924924) >> 2
    ng = (nn & 0x492492) >> 1
    nb = (nn & 0x249249)
    # Get lower corner value in box.  The function interpolate finds entry in ctab.
    r = interpolate(nr); g = interpolate(ng); b = interpolate(nb)
    # Add half box length to get value at centre.
    mid = 0x80 >> level
    return (r | mid, g | mid, b | mid)

# -------------------------------------------------------------------------------------}


def colourfit(x, l):
    """Computes measure of optimality of the fit.  Note the recursion."""
    (c, n, sub_boxes) = x
    s = n/2**l
    for box in sub_boxes: s += colourfit(box, l+1)
    return s

# -------------------------------------------------------------------------------------}


def get_colourlist(x, l):
    """Converts colour tree to list of colours for output, note the recursion."""
    colourlist = []
    (c, n, sub_boxes) = x
    if n > 0: colourlist.append(n2rgb(c, l))
    for box in sub_boxes: colourlist += get_colourlist(box, l+1)
    return colourlist

# -------------------------------------------------------------------------------------}


def loss(x, lo, parent, l):
    """Finds node that can be moved up tree with least penalty, note the recursion."""
    (c, n, sub_boxes) = x
    z = n >> l
    if l > 0 and n > 0 and z < lo[0]: lo = [z, l, x, parent]
    for box in sub_boxes: lo = loss(box, lo, x, l+1)
    return lo

# -------------------------------------------------------------------------------------}


def gain(x, hi, l):
    """Finds node that can be moved down tree with greatest gain, note the recursion."""
    if l == 6: return hi
    (c, n, sub_boxes) = x
    z = n >> l
    if z > hi[0]: hi = [z, l, x]
    for box in sub_boxes: hi = gain(box, hi, l+1)
    return hi

# -------------------------------------------------------------------------------------}


def move_down(l, box):
    """Move colours down a level."""
    global coltree, numcols
    (c, n, sub_boxes) = box
    # Colour root; sub boxes are coloured 8*c + j.
    cc = c << 3
    threshold = n/8
    # Make list of unused subboxes.
    z = [0, 1, 2, 3, 4, 5, 6, 7, ]
    for sub in sub_boxes: z.remove(sub[0] - cc)
    # Get pixels at lower level
    q = p[l+1][cc:cc+8]
    for j in z:
        # Don't make small numbers of pixels into new colour.
        if q[j] <= threshold: continue
        newcol = cc + j
        # Add entry in list of subboxes and increase count of colours.
        box[2].append([newcol, q[j], []])
        numcols += 1
        box[1] -= q[j]
    # If all pixels moved down original colour not used.
    if box[1] == 0: numcols -= 1
    return box

# -------------------------------------------------------------------------------------}


def move_up(l, box, parent):
    """Moves node up a level."""
    global coltree, numcols
    (c, n, sub_boxes) = box
    newcol = c >> 3
    i = parent[2].index(box)
    sub = parent[2][i]
    # If the parent box had no pixels we create new colour.
    if parent[1] == 0: numcols += 1
    # Move pixels from box to parent
    parent[1] += n
    parent[2][i][1] = 0
    numcols -= 1
    # If there are no sub boxes delete box
    if not box[2]: del parent[2][i]
    return

# -------------------------------------------------------------------------------------}


def my_palette(small):
    """Computes optimum colour map using algorithm described in appendix to restore2.pdf."""
    global p, coltree, numcols
    pxls = list(small.getdata())

    # Note that the PIL module method .getdata() produces a list of tuples whereas
    # the gimp procedure produces a simple list of colour values in order r g b.

    # Create lists to contain counts of numbers of pixels of particular colour.
    p0 = []
    num_levels = 6
    for level in range(0, num_levels + 1):
        p0.append(array('i', [0 for i in range(0, 8**level)]))
    # Count pixels in different levels.
    p = p0
    it = iter(pxls)
    try:
        while True:
            (r, g, b) = next(it)
            # The function rgb2n converts rgb triplet into integer used to index boxes.
            i = rgb2n(r, g, b)
            p[6][i]       += 1
            p[5][i >> 3]  += 1
            p[4][i >> 6]  += 1
            p[3][i >> 9]  += 1
            p[2][i >> 12] += 1
            p[1][i >> 15] += 1
            p[0][i >> 18] += 1

    except StopIteration: pass

    # Construct colour tree.  A node is a tuple (colour number, number of pixels
    # of that colour, list of nodes in the next level of the tree).  Colour
    # numbers are defined as follows.  Let a  colour value c 0<=c<256
    # have binary representation c7 c6 c5 c4 c3 c2 c1 c0;  here c can be r,g or b.
    # A level 6 colour number has the binary representation
    # r7 g7 b7 r6 g6 b6 r5 g5 b5 r4 g4 b4 r3 g3 b3 r2 g2 b2.  The corresponding
    # colours at each higher level are obtained by right shifting this 3 places.

    # The initial colour tree contains the 64 colours at level 2
    numcols = 0
    # level2 is list of boxes at level 2.
    level2 = [[] for i in range(0, 8)]
    for i in range(0, 8):
        for j in range(0, 8):
            c = 8*i+j
            if p[2][c] > 0:
                level2[i].append([c, p[2][c], []])
                numcols += 1
    # level1 is list of boxes at level 1
    level1 = []
    for i in range(0, 8):
        if level2[i]: level1.append([i, 0, level2[i]])
    coltree = [0, 0, level1]

    # Set target number of colours.
    col_targ = 256
    # Start with a very bad fit
    lastfit = 1e10
    # k counts colour moves in case of failure.
    k = 0
    while True:
        # If the number of colours is less than required find the box which, if split,
        # produces the greatest improvement in the fit.
        if numcols < col_targ:
            best_gain = gain(coltree, [0, None, None], 0)
            if best_gain[0] == 0:
                print "Less than 256 distinct colours, impossible to restore"
                return None
            move_down(best_gain[1], best_gain[2])
        # note fit before moving colours up the tree in case we need to exit.
        s = colourfit(coltree, 0)
        # If the number of colours is too large find the box which, if the colours are
        # moved up a level causes the least deterioration in the fit.
        if numcols >= col_targ:
            least_loss = loss(coltree, [1e10, None, None], [coltree], 0)
            move_up(least_loss[1], least_loss[2], least_loss[3])
        # If we have the right number of colours exit if fit getting worse.
        if numcols == col_targ:
            nowfit = s
            if nowfit >= lastfit: break
            lastfit = nowfit
        # Count moves up and down.
        k = k + 1
        # Force exit in exceptional circumstances.
        if k > 200: break

        # Unpack colour tree into a list of colours and sort these according to their brightness.
    colours = get_colourlist(coltree, 0)
    colours.sort(lambda C1, C2: cmp( \
                Y(C1[0],C1[1],C1[2]),  Y(C2[0],C2[1],C2[2])))

    return zip(*colours)

###############################################################################
# End of routines for getting optimum colours
###############################################################################


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
    for i in range(0, n): p[i][i] += scale[i]
    psum = [sum([p[i][j] for i in range(0, n+1)]) for j in range(0, n)]
    nfunc = 0
    # Get function value at vertices.
    y = [F(p[i]) for i in range(0, n+1)]
    while True:
        # Get highest.
        hi = y.index(max(y))
        # Set this value very low and get next highest.
        (y[hi], ysave) = (-1e10, y[hi])
        next_hi = y.index(max(y))
        y[hi] = ysave
        # Get lowest.
        lo = y.index(min(y))
        # Test for convergence.
        if 2.0*abs(y[hi] - y[lo])/(abs(y[hi]) + abs(y[lo])) < eps:
            ok = True
            break
        # Exit if failed to converge.
        if nfunc > 5000: break

        nfunc += 2
        (ynew, p, y, psum) = trial(p, y, psum, n, F, hi, -1.0)
        # If new point better try going further.
        if ynew <= y[lo]:
            (ynew, p, y, psum) = trial(p, y, psum, n, F, hi, 2.0)
        # If the new point is worse than the next highest ...
        elif ynew >= y[next_hi]:
            ysave = y[hi]
            (ynew, p, y, psum) = trial(p, y, psum, n, F, hi, 0.5)
            # If getting nowhere shrink the simplex.
            if ynew >= ysave:
                # Loop over vertices keeping the lowest point unchanged.
                for i in range(0, n+1):
                    if i == lo: continue
                    pnew = [0.5*(p[i][j] + p[lo][j]) for j in range(0, n)]
                    p[i] = pnew
                    y[i] = F(pnew)
            nfunc += n
            psum = [sum([p[i][j] for i in range(0, n+1)]) for j in range(0, n)]
        else: nfunc -= 1
    return (p[lo], ok)


def trial(p, y, psum, n, F, hi, dist):
    """Compute point pnew along line from p[hi] to centroid excluding p[h1]."""
    a = (1.0 - dist)/n
    b = a - dist
    pnew = [a*psum[j] - b*p[hi][j] for j in range(0, n)]
    ynew = F(pnew)
    # If improvement accept and adjust psum.
    if ynew < y[hi]:
        y[hi] = ynew
        psum = [psum[j] + (pnew[j] - p[hi][j]) for j in range(0, n)]
        p[hi] = pnew
    return (ynew, p, y, psum)

# -------------------------------------------------------------------------------------}


def first_restore():
    """Optimises lambda and sigma separately for each colour channel."""
    global CH
    CH = R; scale = [0.02, 0.02]; x0 = [1.0, 1.0]
    ((lamr, sigr), ok_r) = simplex(x0, scale, ideal_colour, 1e-4, True)
    CH = G; scale = [0.02, 0.02]; x0 = [1.0, 1.0]
    ((lamg, sigg), ok_g) = simplex(x0, scale, ideal_colour, 1e-4, True)
    CH = B; scale = [0.02, 0.02]; x0 = [1.0, 1.0]
    ((lamb, sigb), ok_b) = simplex(x0, scale, ideal_colour, 1e-4, True)
    if not (ok_r and ok_g and ok_b):
        pdb.gimp_message("The program has failed to obtain a satisfactory"
                          "restoration; the result shown may be poor.")
    return (lamr, lamg, lamb, sigr, sigg, sigb)

# -------------------------------------------------------------------------------------}


def ideal_colour(p):
    """Calculates measure of misfit between actual colours and ideal colours.
    This is to be minimised in the first stage of restoration."""
    lamc, sigc = p
    measure = 0.0
    for i in range(0,256):
        c = int(255 * sigc * (colmap[CH][i]/255.0)**lamc)
        measure += (c - ideal_col[i])**2
    return measure

# -------------------------------------------------------------------------------------}


def colour_balance(p):
    """Calculates weighted average distance in U* v* space from u_off, v_off.
    This is minimised in the second stage of restoration"""
    lamr, lamg, lamb, sigr, sigg, sigb = p
    usum = 0.0; vsum = 0.0; wsum = 0.0
    for i in range(0, 256):
        # Compute the colour as modified by the trial restoration parameters.
        r = int(255 * sigr * (colmap[R][i]/255.0)**lamr)
        g = int(255 * sigg * (colmap[G][i]/255.0)**lamg)
        b = int(255 * sigb * (colmap[B][i]/255.0)**lamb)
        (Lstar, ustar, vstar) = RGB2LUV(r, g, b)
        s = ustar*ustar + vstar*vstar
        # Weight so that only pale colours are considered.
        w = grey/(grey + s)
        usum = usum + w*ustar
        vsum = vsum + w*vstar
        wsum = wsum + w
    dist = (usum/wsum - u_off)**2 + (vsum/wsum - v_off)**2
    return dist

# -------------------------------------------------------------------------------------}


def levels_params(Lambda, Sigma, c_hi):
    """Converts restoration parameters Lambda and Sigma to those used in
    the gimp levels command."""
    alpha = [0, 0, 0]
    s = [0, 0, 0]
    for C in [R, G, B]: s[C] = Sigma[C] * (c_hi[C]/255.0)**Lambda[C]
    smax = max(s)
    for C in [R, G, B]:
        alpha[C] = 1.0/Lambda[C]
        s[C] = int(255.0*(s[C]/smax))
    return (alpha, c_hi, s)

# -------------------------------------------------------------------------------------}


def adjust_levels(image, hi_in, gamma, hi_out):
    """Function is a replacement for the gimp levels command.
    Split image into R ,G and B images and process separately."""
    RGB = list(image.split())
    for C in [R, G, B]:
        alpha = 1.0/float(gamma[C])
        float_hi_in = float(hi_in[C])
        RGB[C] = RGB[C].point(lambda I: int(hi_out[C]*(I/float_hi_in)**alpha+0.5))
    # Return merged image.
    return Image.merge("RGB", tuple(RGB))

###############################################################################


def restore(im):
    global colmap
    # Create a small image to speed determination of restoration parameters.
    (width, height) = im.size
    small_image = im.resize((width/8, height/8), resample=PIL.Image.NEAREST)
    # Get colourmap of small image, using local code.
    cols = get_my_colourmap(small_image)
    # Exit if failed to get colourmap.
    if cols == None: return None

    # if debug: print cols
    (colmap, C_hi) = cols
    # Get first estimate of restoration parameters.
    (lamr, lamg, lamb, sigr, sigg, sigb) = first_restore()
    Lambda1 = [lamr, lamg, lamb]; Sigma1 = [sigr, sigg, sigb]

#    if debug:
#        print C_hi
#        print "Lambda1, Sigma1", Lambda1, Sigma1
#
    # Convert restoration parameters Lambda1 and Sigma1 to parameters for
    # gimp levels command.
    (alpha1, m1, s1) = levels_params(Lambda1, Sigma1, C_hi)
    # Restore the small image using replacement for gimp levels command and
    # get its colourmap.
    restored_small = adjust_levels(small_image, m1, alpha1, s1)
    # if debug: print "m1, alpha1, s1", m1, alpha1, s1
    (colmap, junk) = get_my_colourmap(restored_small)

    # Do second stage of restoration to adjust colour balance.
    scale = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    x0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    fit = simplex(x0, scale, colour_balance, 1e-4, True)[0]
    lamr, lamg, lamb, sigr, sigg, sigb, = fit
    Lambda2 = [lamr, lamg, lamb]; Sigma2 = [sigr, sigg, sigb]

    # if debug: print "Lambda2, Sigma2", Lambda2, Sigma2

    # Combine the parameters for both stages of restoration.
    Lambda3 = [0, 0, 0]; Sigma3 = [0, 0, 0]
    for C in [R, G, B]:
        Sigma3[C] = Sigma2[C] * Sigma1[C]**Lambda2[C]
        Lambda3[C] = Lambda2[C] * Lambda1[C]

    # Get parameters for gimp levels command.
    (alpha2, m2, s2) = levels_params(Lambda3, Sigma3, C_hi)

    # if debug: print "m2, alpha2, s2", m2, alpha2, s2

    # Restore main full size image.
    restored_image = adjust_levels(im, m2, alpha2, s2)

    # Generate a more saturated option if requested.
    if sat_choice:
        # Convert image to HSV format and split into separate images.
        HSVimage = restored_image.convert("HSV")
        HSV = list(HSVimage.split())
        # Get statistics of the S channel to compute new saturation.
        stats = ImageStat.Stat(HSV[1])
        mean = stats.mean[0]
        std_dev = stats.stddev[0]
        # Compute an estimate of high saturation values and factor by which to scale.
        maxsat = mean + 2.0*std_dev
        fac = 1.0/min(1.0, 1.0/min(1.5, 150.0/maxsat))
        # Increase the values in the saturation channel, merge HSV and convert to RGB.
        HSV[1] = HSV[1].point(lambda I: int(fac*I+0.5))
        more_saturated = Image.merge("HSV", tuple(HSV))
        more_saturated = more_saturated.convert("RGB")
        return more_saturated
    else: return restored_image

#############################################################################
# MAIN PROGRAM
#############################################################################


# Set default parameters.
debug = False
directory = os.getcwd()
gamma_target = 1.0
sat_choice = 1
# Parse the command line parameters.
for param in sys.argv:
    Dir = param.find("dir=")
    if Dir >= 0: directory = param[Dir+4:]
    gam = param.find("Light-Dark=")
    if gam >= 0: gammatarg = float(param[gam+11:])
    sat = param.find("saturate=")
    if sat >= 0: sat_choice = eval(param[sat+9:])

# if debug: print directory, gammatarg, sat_choice

# The letters RGB are used throughout for referring to colours.
R = 0; G = 1; B = 2
# Set parameters here used in colour balance, these seem about optimum.
grey = 50.0; u_off = 2.0; v_off = 2.0

# Define colours of 'ideal' image
ideal_col = [int(255.0 * (i/255.0)**gamma_target) for i in range(0, 256)]

# Make look up table for conversion of RGB to LUV, needs to be bigger 
# than 256 in case search explores here.
expand = [0.0 for c in range(0, 360)]
for c in range(0, 256):
    C = c/255.0
    if C > 0.04045: C = ((C + 0.055)/1.055)**2.4
    else: C = C/12.92
    expand[c] = 100.0*C

# Make look-up table for constructing list index.  Let a colour value 0<=i<256
# have binary representation c7 c6 c5 c4 c3 c2 c1 c0 then
# ctab[i] =  c7 0 0 c6 0 0 c5 0 0 c4 0 0 c3 0 0 c2 0 0 c1 0 0 c0.
ctab = []
for c in range(0, 256):
    i = 0
    mask = 0x80
    for j in range(0,8):
        i = (i << 2) | (c & mask)
        mask = mask >> 1
    ctab.append(i)

# Save the current directory.
savedir = directory
# Make list of files to process.
os.chdir(directory)
filelist = os.listdir(directory)
jpegs = []; JPEGS = []; tiffs = []; TIFFS = []
for File in filelist:
    if File.find(".jpg") > 0:  jpegs.append(File)
    if File.find(".JPG") > 0:  JPEGS.append(File)
    if File.find(".tiff") > 0: tiffs.append(File)
    if File.find(".TIFF") > 0: TIFFS.append(File)

# In windows the file searching is NOT case sensitive, so merge.
if JPEGS != jpegs: jpegs += JPEGS
if TIFFS != tiffs: tiffs += TIFFS

# Loop over the photos to be processed.
for photo in jpegs+tiffs:
    # Strip off directory name and .jpg to get file name.
    photoname = os.path.split(photo)[1]
    print photoname
    # Open photo.
    im = Image.open(photoname)
    # Restore the image.
    restored_image = restore(im)
    # Ignore the result if restoration failed.
    if restored_image == None: continue
    # Save file in subdirectory "restored".
    newfilename = os.path.join(directory, "restored", photoname)
    restored_image.save(newfilename, icc_profile=im.info.get('icc_profile'))

# Return to saved directory at end.
os.chdir(savedir)
