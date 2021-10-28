#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from scipy.ndimage import filters
from scipy.signal import medfilt2d
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys
from scipy import signal

sys.path.append("../p1-4students")
import visualPercepUtils as vpu

def applyInEachDim(fun):
    def wrapper(*args,**kargs):
        if len(args[0].shape) == 2:
            return fun(*args,**kargs)
        else:
            res = args[0].copy()
            for i in range(args[0].shape[2]):
                res[:,:,i] = fun(args[0][:,:,i],*args[1:],**kargs)
            return res
    return wrapper


def evaluate_time(fun):
    import time
    def wrapper(*args,**kwargs):
        start = time.time()
        res = fun(*args,**kwargs)
        end = time.time()
        print(f"{fun.__name__} - Total time: {end-start}")
        return res
    return wrapper


# -----------------------
# Salt & pepper noise
# -----------------------

def addSPNoise(im, percent):
    # im is a PIL image
    # percent is in range 0-100 (%)

    # convert image it to numpy 2D array and flatten it
    im_np = np.array(im)
    im_shape = im_np.shape  # keep shape for later use
    im_vec = im_np.flatten()  # this is a 1D array # https://www.geeksforgeeks.org/differences-flatten-ravel-numpy/

    # generate random locations
    N = im_vec.shape[0]  # num pixels
    m = int(math.floor(percent * N/10)) # number of pixels corresponding to the given percentage
    locs = np.random.randint(0, N, m)  # m positions in the 1D array (index 0 to N-1)

    # generate random S/P values (in same proportion)
    s_or_p = np.random.randint(0, 2, m)  # 2 random values (salt and pepper)

    # set the S/P values in the random locations
    im_vec[locs] = 255 * s_or_p  # values after the multiplication will be either 0 or 255

    # turn the 1D array into the original 2D image
    im2 = im_vec.reshape(im_shape)

    # convert Numpy array im2 back to a PIL Image and return it
    return Image.fromarray(im2)


def testSandPNoise(im, percents):
    imgs = []
    for percent in percents:
        imgs.append(addSPNoise(im, percent))
    return imgs


# -----------------
# Gaussian noise
# -----------------
@applyInEachDim
def addGaussianNoise(im, sd=5):
    return im + np.random.normal(loc=0, scale=sd, size=im.shape)


def testGaussianNoise(im, sigmas):
    imgs = []
    for sigma in sigmas:
        print('testing sigma:', sigma)
        imgs.append(addGaussianNoise(im, sigma))
        print(len(imgs))
    return imgs


# -------------------------
# Average (or mean) filter
# -------------------------
@evaluate_time
def averageFilter(im, filterSize):
    mask = np.ones((filterSize, filterSize))
    mask = np.divide(mask, np.sum(mask))
    return filters.convolve(im, mask)
@evaluate_time
def averageFilterSep(im, filterSize):
    mask = np.ones((1, filterSize))
    mask =mask / filterSize
    s= filters.convolve(im, mask)
    print(mask.shape,mask.T.shape)
    return filters.convolve(s,mask.T)


def testAverageFilter(im_clean, params):
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg) # salt and pepper noise
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            a = averageFilter(im_dirty, filterSize)
            b = averageFilterSep(im_dirty, filterSize)
            imgs.append(a)
            imgs.append(b)
            print(a)
            print(b)
            print(np.all(a == b))

    return imgs


# -----------------
# Gaussian filter
# -----------------
@evaluate_time
def gaussianFilter(im, sigma=5):
    # im is PIL image
    return filters.gaussian_filter(im, sigma)

@evaluate_time
def explicitGaussianFilter(im, sigma=5):
    # im is PIL image
    gaus1d = signal.windows.gaussian(15,std = sigma) * sigma
    gauss2d = np.outer(gaus1d,gaus1d.T)
    return filters.convolve(im,gauss2d)

@evaluate_time
def explicitGaussianFilterSep(im, sigma=5):
    # im is PIL image
    gauss1d = signal.windows.gaussian(15,std = sigma).reshape((-1,1)) * sigma
    s = filters.convolve(im,gauss1d)
    return filters.convolve(s,gauss1d.T)


def testGaussianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sigma in params['sd_gauss_noise']:
        im_dirty = addGaussianNoise(im_clean, sigma)
        for filterSize in params['sd_gauss_filter']:
            imgs.append(np.array(im_dirty))
            imgs.append(gaussianFilter(im_dirty, filterSize))
            # imgs.append(np.array(im_dirty))
            imgs.append(explicitGaussianFilter(im_dirty, filterSize))
            # imgs.append(np.array(im_dirty))
            imgs.append(explicitGaussianFilterSep(im_dirty, filterSize))
    return imgs


# -----------------
# Median filter
# -----------------

def medianFilter(im, filterSize):
    return medfilt2d(im, filterSize)

def testMedianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg)
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            imgs.append(medianFilter(im_dirty, filterSize))
    return imgs


# -----------------
# Test image files
# -----------------

path_input = './imgs-P2/'
path_output = './imgs-out-P2/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'lena256.pgm']  # lena256, lena512

# --------------------
# Tests to perform
# --------------------

testsNoises = ['testSandPNoise', 'testGaussianNoise']
testsFilters = ['testAverageFilterSep', 'testGaussianFilter', 'testMedianFilter']
bAllTests = True
if bAllTests:
    tests = testsNoises + testsFilters
else:
    tests = ['testGaussianFilter']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testGaussianNoise': 'Ruido Gaussiano',
             'testSandPNoise': 'Ruido Sal y Pimienta',
             'testAverageFilter': 'Filtro media',
             'testGaussianFilter': 'Filtro Gaussiano',
             'testMedianFilter': 'Filtro mediana',
             'testAverageFilterSep':'Filtro media separado'}

bSaveResultImgs = False

# -----------------------
# Parameters of noises
# -----------------------
percentagesSandP = [0.3]  # ratio (%) of image pixes affected by salt and pepper noise
gauss_sigmas_noise = [3, 5, 10]  # standard deviation (for the [0,255] range) for Gaussian noise

# -----------------------
# Parameters of filters
# -----------------------

gauss_sigmas_filter = [1.2]  # standard deviation for Gaussian filter
avgFilter_sizes = [3, 7, 15]  # sizes of mean (average) filter
medianFilter_sizes = [3, 7, 15]  # sizes of median filter

testsUsingPIL = ['testGaussianFilter']  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)


# -----------------------------------------
# Apply defined tests and display results
# -----------------------------------------

def doTests():
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array

        for test in tests:

            if test == "testGaussianNoise":
                params = gauss_sigmas_noise
                subTitle = ", Sigmas: " + str(params)
            elif test == "testSandPNoise":
                params = percentagesSandP
                subTitle = ", %: " + str(params)
            elif "testAverageFilter" in test :
                params = {}
                params['filterSizes'] = avgFilter_sizes
                params['sp_pctg'] = percentagesSandP
                subTitle = ", filter sizes" + str(params)
            elif test == "testMedianFilter":
                params = {}
                params['filterSizes'] = avgFilter_sizes
                params['sp_pctg'] = percentagesSandP
                subTitle = ", filter sizes" + str(params)
            elif test == "testGaussianFilter":
                params = {}
                params['sd_gauss_noise'] = gauss_sigmas_noise
                params['sd_gauss_filter'] = gauss_sigmas_filter
                subTitle = ", sigmas (noise): " + str(gauss_sigmas_noise) + ", sigmas (filter): " + str(gauss_sigmas_filter)
            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params)
                print("num images", len(outs_np))
            print(len(outs_np))
            # display original image, noisy images and filtered images
            vpu.showInGrid([im] + outs_np, title=nameTests[test] + subTitle)

if __name__ == "__main__":
    doTests()