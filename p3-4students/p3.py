#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from matplotlib import cm
from scipy import signal
from scipy.ndimage import filters
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys
import time

sys.path.append("../p1-4students")
import visualPercepUtils as vpu


# ----------------------
# Fourier Transform
# ----------------------


def FT(im):
    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.fft.fft2.html
    return fft.fftshift(
        fft.fft2(im)
    )  # perform also the shift to have lower frequencies at the center


def IFT(ft):
    return fft.ifft2(
        fft.ifftshift(ft)
    )  # assumes ft is shifted and therefore reverses the shift before IFT


def testFT(im, params=None):
    ft = FT(im)
    print(ft[0].shape)
    phase = np.angle(ft)
    magnitude = np.log(np.absolute(ft))
    im2 = np.absolute(
        IFT(ft)
    )  # IFT is actually a complex number. When applied to real-valued data the imaginary part should be zero, but not exactly for numerical precision issues
    return [magnitude, phase, im2]


# -----------------------
# Convolution theorem
# -----------------------

# the mask corresponding to the average (mean) filter
def avgFilter(filterSize):
    mask = np.ones((filterSize, filterSize))
    return mask / np.sum(mask)


def gaussianFilter(filterSize: int = 15, sigma=1.2):
    gaus1d = signal.windows.gaussian(filterSize, std=sigma)
    gauss2d = np.outer(gaus1d, gaus1d)
    
    # plt.imshow(gauss2d,cmap='gray')
    # plt.show()
    return gauss2d


# apply average filter in the space domain
def averageFilterSpace(im, filterSize):
    return filters.convolve(im, avgFilter(filterSize))


def gaussianFilterSpace(im, filterSize):
    gauss = gaussianFilter()
    res = filters.convolve(im, gauss)
    # plt.imshow(res)
    # plt.show()
    return res

def centerFilter(im, filterMask) -> np.ndarray:

    filterBig = np.zeros_like(
        im, dtype=float
    )  # as large as the image (dtype is important here!)

    # Now, place filter (the "small" filter mask) at the center of the "big" filter

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2) : int(W2 + w2), int(H2 - h2) : int(H2 + h2)] = filterMask

    # FFT of the big filter
    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner
    return filterBig  # both '*' and multiply() perform elementwise product


# apply average filter in the frequency domain
def averageFilterFrequency(im, filterSize):
    filterMask = avgFilter(filterSize)  # the usually small mask
    filterBig = centerFilter(im, filterMask)

    # Finally, IFT of the element-wise product of the FT's
    return np.absolute(IFT(FT(im) * FT(filterBig)))


def gaussianFilterFrequency(im, filterSize):
    filterMask = gaussianFilter(filterSize)  # the usually small mask
    filterBig = centerFilter(im, filterMask)

    # Finally, IFT of the element-wise product of the FT's
    return np.absolute(IFT(FT(im) * FT(filterBig)))


def testConvTheo(im, params=None):
    filterSize = params["filterSize"]

    # image filtered with a convolution in space domain
    imFiltSpace = averageFilterSpace(im, filterSize)

    # image filtered in frequency domain
    imFiltFreq = averageFilterFrequency(im, filterSize)

    # image filtered with a convolution in space domain
    imFiltSpaceGauss = gaussianFilterSpace(im,filterSize)

    # image filtered in frequency domain
    imFiltFreqGauss = gaussianFilterFrequency(im, filterSize)

    # How much do they differ?
    # To quantify the difference, we use the Root Mean Square Measure (https://en.wikipedia.org/wiki/Root_mean_square)
    margin = 5  # exclude some outer pixels to reduce the influence of border effects
    rms = (
        np.linalg.norm(
            imFiltSpace[margin:-margin, margin:-margin]
            - imFiltFreq[margin:-margin, margin:-margin],
            2,
        )
        / np.prod(im.shape)
    )
    print("Images filtered in space and frequency differ in (RMS):", rms)

    return [imFiltSpace, imFiltFreq, imFiltSpaceGauss,imFiltFreqGauss,]


# -----------------------------------
# High-, low- and band-pass filters
# -----------------------------------

# generic band-pass filter (both, R and r, given) which includes the low-pass (r given, R not)
# and the high-pass (R given, r not) as particular cases
def passBandFilter(shape, r=None, R=None):
    m, n = shape
    m2, n2 = np.floor(m / 2.0), np.floor(n / 2.0)
    [vx, vy] = np.meshgrid(np.arange(-m2, m2 + 1), np.arange(-n2, n2 + 1))
    distToCenter = np.sqrt(vx ** 2.0 + vy ** 2.0)
    if R is None:  # low-pass filter assumed
        assert r is not None, "at least one size for filter is expected"
        subr = np.floor(r * 0.1)
        u = r-distToCenter/subr
        filter = distToCenter < (r - subr)  # same as np.less(distToCenter, r)
        filter.astype(float) +

    elif r is None:  # high-pass filter assumed
        filter = distToCenter > R  # same as np.greater(distToCenter, R)
    else:  # both, R and r given, then band-pass filter
        if r > R:
            r, R = (
                R,
                r,
            )  # swap to ensure r < R (alternatively, warn the user, or throw an exception)
        filter = np.logical_and(distToCenter < R, distToCenter > r)
    filter = filter.astype(
        "float"
    )  # convert from boolean to float. Not strictly required

    bDisplay = True
    if bDisplay:
        plt.imshow(filter, cmap="gray")
        plt.show()
        # Image.fromarray((255*filter).astype(np.uint8)).save('filter.png')

    return filter


def testBandPassFilter(im, params=None):
    r, R = params["r"], params["R"]
    filterFreq = passBandFilter(
        im.shape, r, R
    )  # this filter is already in the frequency domain
    filterFreq = fft.ifftshift(
        filterFreq
    )  # shifting to have the origin as the FT(im) will be
    return [np.absolute(fft.ifft2(filterFreq * fft.fft2(im)))]  # the filtered image


# -----------------
# Test image files
# -----------------
path_input = "./imgs-P3/"
path_output = "./imgs-out-P3/"
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + "lena255.pgm"]  # habas, mimbre

# --------------------
# Tests to perform
# --------------------
bAllTests = True
if bAllTests:
    tests = ["testFT", "testConvTheo", "testBandPassFilter"]
else:
    tests = ["testFT"]
    tests = ["testConvTheo"]
    tests = ["testBandPassFilter"]

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {
    "testFT": "Transformada de Fourier 2D",
    "testConvTheo": "Teorema de la convolucion",
    "testBandPassFilter": 'Filtros ("pasa-alta/baja/banda") en la Frecuencia',
}

bSaveResultImgs = False

testsUsingPIL = (
    []
)  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)


# -----------------------------------------
# Apply defined tests and display results
# -----------------------------------------


def doTests():
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile).convert("L")
        im = np.array(im_pil)  # from Image to array

        for test in tests:

            if test == "testFT":
                params = {}
                subTitle = ": I, |F|, ang(F), IFT(F)"
            elif test == "testConvTheo":
                params = {}
                params["filterSize"] = 7
                subTitle = ": I, I*M, IFT(FT(I).FT(M))"
            else:
                params = {}
                r, R = 5, None  # 5,30 #None, 30
                params["r"], params["R"] = r, R
                # let's assume r and R are not both None
                if r is None:
                    filter = "pasa-alta" + " (R=" + str(R) + ")"
                elif R is None:
                    filter = "pasa-baja" + " (r=" + str(r) + ")"
                else:
                    filter = "pasa-banda" + " (r=" + str(r) + ", R=" + str(R) + ")"
                subTitle = ", filtro " + filter

            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params)
            print("num images", len(outs_np))
            print(len(outs_np))

            vpu.showInGrid([im] + outs_np, title=nameTests[test] + subTitle)


def measure_time(fun):
    start = time.time()
    fun()
    end = time.time()
    return end - start

def compare_time():
    im = np.array(Image.open('imgs-P3/lena255.pgm').convert("L"))
    t_frequ = []
    t_space = []
    for i in range(3,20):
        t_frequ.append(measure_time(lambda: averageFilterFrequency(im,i)))
        t_space.append(measure_time(lambda: averageFilterSpace(im,i)))
    vpu.showInGrid([np.array(t_frequ),np.array(t_space)], title='Comparación temporal')



if __name__ == "__main__":
    # compare_time()
    doTests()
