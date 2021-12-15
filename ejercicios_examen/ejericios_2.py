from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters 
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, disk,square
from skimage import measure



def aplf(im:np.ndarray)->np.ndarray:
    mask_x = np.array([[0,0,0],[-1,0,1],[0,0,0]])
    mask_y = mask_x.T
    return np.sqrt(filters.convolve(im,mask_x)**2 +filters.convolve(im,mask_y)**2)
    # return np.abs(filters.convolve(im,mask_x))+np.abs(filters.convolve(im,mask_y))

def binarize(im:np.ndarray,thrs:list[float]):
    return [im > t for t in thrs]

def histReg(blobs:np.ndarray):
    labels = measure.label(blobs,background=0)
    per = ((labels != 0).sum() / labels.size)*100
    print([m.area for m in measure.regionprops(labels)])
    hist = np.histogram([m.area for m in measure.regionprops(labels)],bins=8)
    
    return labels,(per,hist[1],hist[0])

if __name__ == '__main__':
    im = np.array(Image.open('ejercicios_examen/coins.jpg').convert('L'))
    thr = threshold_otsu(im)
    bin_im = im <= thr
    bin_im = closing(bin_im,square(5))
    bin_im,oth =  histReg(bin_im)
    # print(per)
    plt.plot(oth[1])
    plt.show()
    # im = np.array(Image.open('ejercicios_examen/lena256.pgm').convert('L'))
    # plt.imshow(binarize(im,[50,100,150,220])[1].astype(int),cmap='gray')
    # plt.show()
    # im = aplf(im)
    plt.imshow(bin_im,cmap='jet')
    plt.show()