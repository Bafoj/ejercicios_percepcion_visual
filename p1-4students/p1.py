#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
# from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import visualPercepUtils as vpu

def histeq(im, nbins=256):
    imhist, bins = np.histogram(im.flatten(), list(range(nbins+1)), density=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # normalizar (cdf[-1] es el último elemento de la suma acumulada = num. pixels total)
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

def testHistEq(im):
    im2, cdf = histeq(im)
    imgs = [im, im2]
    return [im2, cdf]

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


@applyInEachDim
def checkBoardImg(im:np.ndarray,m:int=5,n:int=3):
  img = im.copy()
  n_filas = math.ceil(img.shape[0] /m)
  n_cols = math.ceil(img.shape[1] /n)
  for v in range(1,m+1):
    indice_fila_ini = n_filas * (v-1)
    indice_fila_fin = n_filas * v
    for h in range(v%2,n+1,2):
        indice_col_ini = n_cols * (h-1)
        indice_col_fin =  n_cols * h
        img[indice_fila_ini:indice_fila_fin,indice_col_ini:indice_col_fin] = 255 - img[indice_fila_ini:indice_fila_fin,indice_col_ini:indice_col_fin]
  return img

def testCheckBoardImg(im:np.ndarray):
    im2 = checkBoardImg(im)
    return [im2]


@applyInEachDim
def darkenImg(im:np.ndarray,p=2)->np.ndarray:
    return (im ** float(p)) / (255 ** (p - 1))
   
@applyInEachDim
def brightenImg(im:np.ndarray,p=2):
    return np.power(255.0 ** (p - 1) * im, 1. / p)  # notice this NumPy function is different to the scalar math.pow(a,b)
  


def testDarkenImg(im):
    im2 = darkenImg(im,p=2) # ¿Es diferente "p=2" aquí que en la definición de la función? ¿Se puede no poner "p="?
    imgs = [im, im2]
    # print im2.shape, hists[0].shape
    return [im2]


def testBrightenImg(im):
    p=2
    im2=brightenImg(im,p)
    imgs = [im, im2]
    # print im2.shape, hists[0].shape
    return [im2]

def saveOutImg(imfile, test, im2):
    _,basename = os.path.dirname(imfile), os.path.basename(imfile)
    fname, fext = os.path.splitext(basename)
                #print(dname,basename)
    pil_im = Image.fromarray(im2.astype(np.uint8))  # from array to Image
    pil_im.save(path_output+'//'+fname + suffixFiles[test] + fext)

path_input = './imgs-P1/'
path_output = './imgs-out-P1/'
bAllFiles = True
formats = ['ppm','pgm']
current = 0
if bAllFiles:
    files = glob.glob(path_input + f"*.{formats[current]}")
else:
    files = [path_input + 'toys.ppm'] # iglesia,huesos

bAllTests = True
if bAllTests:
    tests = ['testHistEq', 'testBrightenImg', 'testDarkenImg','testCheckBoardImg']  # 'testHistEq']:#''testDarkenImg']:
else:
    tests = ['testCheckBoardImg']#['testBrightenImg']
nameTests = {'testHistEq': u"Ecualización de histograma", # Unicode (u) para tildes y otros carácteres
             'testBrightenImg': 'Aclarar imagen',
             'testDarkenImg': 'Oscurecer imagen',
             'testCheckBoardImg':'Cuadricular imagen'}
suffixFiles = {'testHistEq': '_heq',
               'testBrightenImg': '_br',
               'testDarkenImg': '_dk',
                'testCheckBoardImg':'_cuad'}

bSaveResultImgs = True

def doTests():
    print("Testing on", files)
    for imfile in files:
        
        im = np.array(Image.open(imfile)) if current == 0 else np.array(Image.open(imfile).convert('L')) # from Image to array
        for test in tests:
            out = eval(test)(im)
            im2 = out[0]
            vpu.showImgsPlusHists(im, im2, title=nameTests[test])
            if len(out) > 1:
                vpu.showPlusInfo(out[1])
            if bSaveResultImgs:
                saveOutImg(imfile, test, im2)



if __name__== "__main__":
    doTests()

