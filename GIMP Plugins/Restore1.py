#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Restore faded slides.

Restores scanned slides that have deteriorated with age.
Estimates the loss of dyes in the emulsion and restores the original values.
The Degree of Restoration parameter adjusts the contrast and a yellow shift seems to improve the results.
The side-absorption correction can be left as additional layers

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
from math import *

# 3rd party package imports
from gimpfu import *


def colourmap(image, alpha, m, top, bot):
    R = 0
    G = 1
    B = 2

    # make a copy for trial restoration
    small_copy = pdb.gimp_image_duplicate(image)

    # restore it
    slayer = pdb.gimp_image_get_active_layer(small_copy)
    for C in [R, G, B]:
        pdb.gimp_levels(slayer, C + 1, 0, m[C], alpha[C], 0, 255)

    # get modified colourmap
    pdb.gimp_image_convert_indexed(small_copy, 0, 0, 256, False, True, "mypal")
    (num_bytes, scolourmap) = pdb.gimp_image_get_colormap(small_copy)
    pdb.gimp_image_delete(small_copy)

    # unpack colourmap
    colrange = [0, 0, 0]
    for C in [R, G, B]:
        c = scolourmap[slice(C, 768, 3)]
        colrange[C] = (percentile(c, top), percentile(c, bot))
    return colrange


def percentile(c, f):
    n = len(c)
    N = 0
    for i in range(0, n):
        if c[i] > 0:
            N += 1
    nf = int(N * f)
    m = -1
    k = 0
    while k < nf:
        m = m + 1
        k = 0
        for i in range(0, n):
            if (c[i] > 0) and (c[i] <= m):
                k = k + 1
    return float(m)


def restore(image, drawable, shift, contrast, flatten):
    # message on attempt to restore greyscale image
    if pdb.gimp_drawable_is_gray(drawable):
        pdb.gimp_message("Restore does not work on greyscale images. Either use the levels command from the "
                         "colors menu, or scan the image in colour (not convert from greyscale to RGB), restore it "
                         "and then convert the result to greyscale.")
        return

    R = 0
    G = 1
    B = 2
    width = pdb.gimp_image_width(image)
    height = pdb.gimp_image_height(image)
    # debugfile = open("/home/gjd/gimp/debug.txt", "w")

    # make small image for speed
    small_image = pdb.gimp_image_new(width, height, 0)
    layer = pdb.gimp_layer_new_from_drawable(drawable, small_image)
    pdb.gimp_image_add_layer(small_image, layer, 0)
    pdb.gimp_image_scale(small_image, width / 10, height / 10)

    # get initial lower and upper percentiles
    top = 1.0
    bot = 0.1
    colrange = colourmap(small_image, [1.0, 1.0, 1.0], [255, 255, 255], top, bot)
    # debugfile.write("colrange" + str(colrange) + "\n")

    # get initial m and alpha
    initalpha = [0.0, 0.0, 0.0]
    initm = [0.0, 0.0, 0.0]
    for C in [R, G, B]:
        (hiC, loC) = colrange[C]
        initm[C] = exp((log(top) * log(loC) - log(bot) * log(hiC)) / (log(top) - log(bot)))
        initalpha[C] = (log(loC) - log(initm[C])) / log(bot)
    # debugfile.write("initial alphas" + str(initalpha) + "\n")
    # debugfile.write("initial " + str(initm) + "\n")

    # iterate to get correct m and alpha
    alpha = initalpha[:]
    m = initm[:]
    dalpha = [1.0, 1.0, 1.0]
    dm = [0.0, 0.0, 0.0]
    iters = 0
    while max(abs(dalpha[R]), abs(dalpha[G]), abs(dalpha[B])) > 0.02:
        iters += 1
        if iters == 10:
            break
        # debugfile.write("during iter" + str(iter) + str(colrange) + "\n")
        # debugfile.write("during iter" + str(iter) + str(alpha) + "\n")
        for C in [R, G, B]:
            if (alpha[C] < 0.1) or (alpha[C] > 10.0):
                pdb.gimp_message("The image has deteriorated too far to restore. "
                                 "You may be able to improve it by using the 'levels' operation first to improve "
                                 "the colour and then trying 'Restore'.")
                return
        colrange = colourmap(small_image, alpha, m, top, bot)
        for C in [R, G, B]:
            (hiC, loC) = colrange[C]
            dalpha[C] = alpha[C] * alpha[C] * (loC / (255 * bot) - 1) / log(bot)
            dm[C] = alpha[C] * (hiC - 255 * top)
            if abs(dalpha[C]) > 0.2 * alpha[C]:
                dalpha[C] = 0.2 * alpha[C] * dalpha[C] / abs(dalpha[C])
            alpha[C] += dalpha[C]
            m[C] += dm[C]

    # if loop failed to converge use original values
    if iters == 10:
        # debugfile.write("alpha iteration failed to converge\n")
        pdb.gimp_message("The image has deteriorated badly; the restoration is probably poor "
                         "but may be an improvement on the original.")
        alpha = initalpha[:]
        m = initm[:]

    # debugfile.write("numberof initial iterations " + str(iter) + "\n")
    # debugfile.write("final alphas" + str(alpha) + "\n")
    # debugfile.write("final m" + str(m) + "\n")

    # create restored image
    new_image = pdb.gimp_image_new(width, height, 0)
    layer0 = pdb.gimp_layer_new_from_drawable(drawable, new_image)
    pdb.gimp_image_add_layer(new_image, layer0, 0)
    pdb.gimp_layer_set_offsets(layer0, 0, 0)
    pdb.gimp_layer_set_mode(layer0, 0)

    # get colourmap of restored small image
    slayer = pdb.gimp_image_get_active_layer(small_image)
    for C in [R, G, B]:
        pdb.gimp_levels(slayer, C + 1, 0, m[C], alpha[C], 0, 255)
    pdb.gimp_image_convert_indexed(small_image, 0, 0, 256, False, True, "mypal")
    (num_bytes, scolourmap) = pdb.gimp_image_get_colormap(small_image)
    # debugfile.write("numbytes  " + str(num_bytes) + "\n")
    # debugfile.write(str(scolourmap))
    newc = [0, 0, 0]
    for C in [R, G, B]:
        newc[C] = scolourmap[slice(C, 768, 3)]
    pdb.gimp_image_delete(small_image)

    # for i in range(0,256):
    #    debugfile.write(str(newc[R][i]) + "," + str(newc[G][i]) + "," + str(newc[B][i]) + "\n")

    # average colour
    av = [(newc[R][i] + newc[G][i] + newc[B][i]) / (3.0 * 255) for i in range(0, 256)]

    # weights for colour balance, included for experiment
    w = [1.0 for i in range(0, 256)]

    # tweak restoration
    LAMBDA = [1.0, 1.0, 1.0]
    SIGMA = [1.0, 1.0, 1.0]
    ok = True
    for C in [R, G, B]:
        lambdaC = 1.0
        dlambdaC = 1.0
        l = [log((newc[C][i] + 1) / 255.0) for i in range(0, 256)]
        iters = 0
        while abs(dlambdaC) > 1e-4:
            iters += 1
            if iters >= 20:
                break
            pc = [pow(newc[C][i] / 255.0, lambdaC) for i in range(0, 256)]
            ppl = sum([w[i] * pc[i] * pc[i] * l[i] for i in range(0, 256)])
            ppll = sum([w[i] * pc[i] * pc[i] * l[i] * l[i] for i in range(0, 256)])
            pp = sum([w[i] * pc[i] * pc[i] for i in range(0, 256)])
            ap = sum([w[i] * av[i] * pc[i] for i in range(0, 256)])
            apl = sum([w[i] * av[i] * pc[i] * l[i] for i in range(0, 256)])
            apll = sum([w[i] * av[i] * pc[i] * l[i] * l[i] for i in range(0, 256)])
            sigma = ap / pp
            dlambdaC = -(apl - sigma * ppl) / (apll - 2 * sigma * ppll)
            lambdaC += dlambdaC
            # debugfile.write("sigma  " + str(C) + "  " + str(sigma) + "\n")
            # debugfile.write("lambda " + str(lambdaC) + "\n")

        LAMBDA[C] = lambdaC
        SIGMA[C] = sigma
        # debugfile.write("number of refinement iterations for colour " + str(C) + "  " + str(iter) + "\n")
        if iter == 20:
            # debugfile.write("lambda iteration failed to converge\n")
            ok = False

    # if loop fails to converge use default values
    if not ok:
        LAMBDA = [1.0, 1.0, 1.0]
        SIGMA = [1.0, 1.0, 1.0]

    # debugfile.write("LAMBDAS" + str(LAMBDA) + "\n")
    # debugfile.write("SIGMAS" + str(SIGMA) + "\n")
    smin = min(SIGMA)

    # adjust parameters and restore
    for C in [R, G, B]:
        alpha[C] = alpha[C] / LAMBDA[C]
        SIGMA[C] = SIGMA[C] / smin
        m[C] = m[C] / pow(SIGMA[C], alpha[C])

    # implement degree of restoration and restore
    for C in [R, G, B]:
        alpha[C] = 1 - contrast * (1 - alpha[C])
        m[C] = 255 - contrast * (255 - m[C])
        pdb.gimp_levels(layer0, C + 1, 0, m[C], alpha[C], 0, 255)

    # debugfile.write("final parameters\n")
    # debugfile.write("alpha " + str(alpha) + "\n")
    # debugfile.write("m     " + str(m) + "\n")

    # fudge colour balance correction
    if shift is True:
        pdb.gimp_color_balance(layer0, 1, True, 0.0, 0.0, -20.0)

    # side absorption corrections
    # approximate values of side absorption coefficient
    deltaRG = 0.15
    deltaRB = 0.07
    deltaGR = 0.05
    deltaGB = 0.18
    deltaBR = 0.05
    deltaBG = 0.03

    # decompose not used for linear approximation to side corrections
    # (rgb_image, junk, junk, junk) = pdb.plug_in_decompose(new_image, drawable, "RGB", 1)
    # (layerR, layerG, layerB) = (rgb_image.layers[0], rgb_image.layers[1], rgb_image.layers[2])

    gr = deltaGR * alpha[G] / alpha[R]
    br = deltaBR * alpha[B] / alpha[R]
    rg = deltaRG * alpha[R] / alpha[G]
    bg = deltaBG * alpha[B] / alpha[G]
    rb = deltaRB * alpha[R] / alpha[B]
    gb = deltaGB * alpha[G] / alpha[B]
    pdb.gimp_invert(layer0)

    # multiplicative correction layer (not currently used)
    #     layer_m = pdb.gimp_layer_new_from_drawable(layer0, new_image)
    #     pdb.gimp_image_add_layer(new_image, layer_m, 0)
    #     pdb.gimp_layer_set_offsets(layer_m, 0, 0)
    #     pdb.gimp_layer_set_mode(layer_m, 3)
    #     gr = deltaGR
    #     br = deltaBR
    #     rg = deltaRG
    #     bg = deltaBG
    #     rb = deltaRB
    #     gb = deltaGB
    #     pdb.plug_in_colors_channel_mixer(new_image, layer_m, FALSE, 0.0, gr, br, rg, 0.0, bg, rb, gb, 0.0)
    #     pdb.gimp_invert(layer_m)

    # division correction layer
    layer_d = pdb.gimp_layer_new_from_drawable(layer0, new_image)
    pdb.gimp_image_add_layer(new_image, layer_d, 0)
    pdb.gimp_layer_set_offsets(layer_d, 0, 0)
    pdb.gimp_layer_set_mode(layer_d, 15)
    pdb.plug_in_colors_channel_mixer(new_image, layer_d, False, 0.0, gr, br, rg, 0.0, bg, rb, gb, 0.0)
    pdb.gimp_invert(layer_d)
    pdb.gimp_invert(layer0)

    if flatten is True:
        drawable = pdb.gimp_image_flatten(new_image)

    pdb.gimp_display_new(new_image)
    # debugfile.close()


register(
    "python-fu_Restore1",
    "Restore faded slides",
    "Restores scanned slides that have deteriorated with age. "
    "Estimates the loss of dyes in the emulsion and restores the original values. "
    "The Degree of Restoration parameter adjusts the contrast and a yellow shift seems to improve the results. "
    "The side-absorption correction can be left as additional layers.",
    "Geoff Daniell, geoff@lionhouse.plus.com",
    "Geoff Daniell",
    "2008",
    "<Image>/Restore/Restore1",
    "",
    [
        (PF_TOGGLE, "shift", "Make less blue", True),
        (PF_SLIDER, "contrast", "Degree of Restoration",  1.0, (0.0, 1.2, 0.05)),
        (PF_TOGGLE, "flatten", "Combine all Layers", True)
    ],
    [],
    restore
    )

main()
