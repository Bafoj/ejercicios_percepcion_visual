from PIL import Image
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy.signal import windows

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
def filterIm_ej1(im:np.ndarray)->np.ndarray:
    mask = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])
    return filters.convolve(im,mask)

@applyInEachDim
def countIm_ej3(im:np.ndarray,thr:int = 100,sX:int=5,sY:int=5,minCount:int=10):
    levels = (im < thr).astype(int)
    mask = np.ones((sX,sY))
    con = filters.convolve(levels,mask)
    return con >= minCount


if __name__ == '__main__':
    image = np.array(Image.open('ejercicios_examen/lena256.pgm').convert('L'))
    plt.imshow(countIm_ej3(image)*image,cmap='gray')
    plt.show()
