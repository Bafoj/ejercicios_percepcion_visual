# !/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import glob
import sys

from skimage import feature
# from skimage.transform import hough_line, hough_line_peaks  # , probabilistic_hough_line
from skimage.morphology import disk, square, closing, opening, erosion, dilation
from skimage import measure

from scipy import ndimage as ndi
from copy import deepcopy

sys.path.append("../p1-4students")
import visualPercepUtils as vpu

bStudentVersion=True
if not bStudentVersion:
    import p5e

def testOtsu(im, params=None):
    nbins = 256
    th = threshold_otsu(im)
    hist = np.histogram(im.flatten(), bins=nbins, range=[0, 255])[0]
    return [th, im > th, hist]  # threshold, binarized image (using such threshold), and image histogram


def fillGaps(im, params=None):
    binIm = im > threshold_otsu(im)
    sElem = disk(params['closing-radius'])
    return [binIm, closing(binIm, sElem)]

def removeSmallRegions(im, params=None):
    binIm = im > threshold_otsu(im)
    sElem = disk(params['opening-radius'])
    return [binIm, opening(binIm, sElem)]

def fillGapsThenRemoveSmallRegions(im, params=None):
    binIm, closeIm = fillGaps(im, params)  # first, fill gaps
    sElem = disk(params['opening-radius'])
    return [binIm, opening(closeIm, sElem)]


def segmenta_test(im, params=None):
    binIm, closeIm = fillGaps(im, {'closing-radius':3})  # first, fill gaps
    sElem = square(5)
    closeIm = erosion(closeIm,sElem)
    sElem = disk(9)
    closeIm = dilation(erosion(closeIm,sElem),sElem)
    
    sElem = disk(5)
    return [binIm, opening(closeIm, sElem)]

def labelConnectedComponents(im, params=None):
    binIm = fillGapsThenRemoveSmallRegions(im, params)[1]
    return [binIm, measure.label(binIm, background=0)]

def reportPropertiesRegions(labelIm):
    regions = measure.regionprops(labelIm)
    for r, region in enumerate(regions):  # enumerate() is often handy: it provides both the index and the element
        print("Region", r + 1, "(label", str(region.label) + ")")
        print("\t area: ", region.area)
        print("\t perimeter: ", round(region.perimeter, 1))  # show only one decimal place

# -----------------
# Test image files
# -----------------
path_input = './imgs-P5/'
path_output = './imgs-out-P5/'
bAllFiles = True
if bAllFiles:
    files = glob.glob(path_input + "*.p??")
else:
    files = [path_input + 'monedas.pgm']

# --------------------
# Tests to perform
# --------------------
bAllTests = False
if bAllTests:
    tests = ['testOtsu', 'fillGaps', 'removeSmallRegions',
             'fillGapsThenRemoveSmallRegions','labelConnectedComponents']
else:
    # tests = ['testOtsu']
    # tests = ['fillGaps']
    tests = ['segmenta_test']
    # tests = ['removeSmallRegions']
    # tests = ['fillGapsThenRemoveSmallRegions']
    # tests = ['labelConnectedComponents']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testOtsu': "thresholding with Otsu's method",
             'fillGaps': 'Filling gaps inside regions',
             'removeSmallRegions': 'Removing small regions',
             'fillGapsThenRemoveSmallRegions': 'Removing small regions AFTER filling gaps',
             'labelConnectedComponents': 'Labelling conected components',
             'segmenta_test': 'Labelling de pruebas'}

myThresh = 180  # use your own value here
diskSizeForClosing = 2  # try other different values
diskSizeForOpening = 5  # try other different values

def doTests():
    print("Testing ", tests, "on", files)
    nFiles = len(files)
    nFig = None
    for i, imfile in enumerate(files):
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array

        for test in tests:
            title = nameTests[test]
            print(imfile, test)
            params = {}
            m = n = 2
            if test is "testOtsu":
                params = {}
            elif test is "fillGaps":
                params = {}
                params['closing-radius'] = diskSizeForClosing
                m, n = 1, 3
                subtitles = ["original image", "binarized image",
                             "Closing with disc of radius = " + str(diskSizeForClosing)]
            elif test is "segmenta_test":
                params = {}
                params['closing-radius'] = diskSizeForClosing
                m, n = 1, 3
                subtitles = ["original image", "binarized image",
                             "test = " + str(diskSizeForClosing)]
            elif test is "removeSmallRegions":
                params = {}
                params['opening-radius'] = diskSizeForOpening
                m, n = 1, 3
                subtitles = ["original image", "binarized image",
                             "Opening with disc of radius = " + str(diskSizeForOpening)]
            elif test is "fillGapsThenRemoveSmallRegions":
                params = {}
                params['closing-radius'] = diskSizeForClosing
                params['opening-radius'] = diskSizeForOpening
                m, n = 1, 3
                subtitles = ["original image", "binarized image",
                             "Opening and closing with discs of radii  "
                             + str(diskSizeForOpening) + " and " + str(diskSizeForClosing)]
            elif test is 'labelConnectedComponents':
                params = {}
                params['closing-radius'] = diskSizeForClosing
                params['opening-radius'] = diskSizeForOpening
                m, n = 1, 3
                subtitles = ["original image", "processed binarized image", "Connected components"]

            outs_np = eval(test)(im, params)

            if test is "testOtsu":
                outs_np_plot = [outs_np[2]] + [outs_np[1]] + [im > myThresh]
                subtitles = ["original image", "Histogram", "Otsu with threshold=" + str(outs_np[0]),
                             "My threshold: " + str(myThresh)]
                m = n = 2
            else:
                outs_np_plot = outs_np
            print(len(outs_np_plot))
            vpu.showInGrid([im] + outs_np_plot, m=m, n=n, title=title, subtitles=subtitles)
            if test is 'labelConnectedComponents':
                plt.figure()
                vpu.showImWithColorMap(outs_np_plot[1],'jet') # the default color map, 'spectral', does not work in lab computers



                reportPropertiesRegions(labelIm=outs_np_plot[1])

                if not bStudentVersion:
                    p5e.displayImageWithCoins(im,labelIm=outs_np_plot[1])

    plt.show(block=True)  # show pending plots


if __name__ == "__main__":
    doTests()
