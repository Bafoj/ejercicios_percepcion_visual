from scipy.ndimage import filters, grey_dilation, zoom

import numpy as np

class SIFT:
    # CAVEAT:
    # This is an approximate partial implementation of SIFT.
    # Please, do *not* rely on this beyond its educational use in this course

    '''
    Reference:
    [Lowe04] David G. Lowe.
    Distinctive image features form Scale-Invariant Keypoints
    International Journal of Computer Vision 60(2), 91-110, 2004
    '''

    def __init__(self, scales, octaves):
        self.s = scales + 1
        self.sigma=1.6
        self.k=2**(1 / scales)
        self.octaves = octaves

    def gaussFilterOneOctave(self,im):
        L = []
        sigmas = [(self.k ** i) * self.sigma for i in range(-1,self.s)]
        print("sigmas",sigmas)
        for sigma in sigmas:
            L.append(filters.gaussian_filter(im, sigma))
            # alternatively, Gaussian filter with constant self.sigma the previous Gaussian-filtered image, not the original image
        return L

    def upsample(self,imgs,ratio):
        # upsample each image in list imgs to size (m,n)
        return [zoom(imgs[i],ratio) for i in range(len(imgs))]

    def step1(self,im):
        # compute the scale space (first part of Sect. 3 in [Lowe04])
        L = []
        m0,n0=im.shape # keep size of original image
        ratio=1
        for j in range(self.octaves):
            print("octave",j,"len(L)",len(L))
            img = im.copy() if j==0 else zoom(Loctave[-1],0.5) # in first octave, we process the input image; otherwise, the halved top image from previous octave
            #print("min,max img", img.min(), img.max())
            Loctave = self.gaussFilterOneOctave(img) # 0.5 = next octave
            #print("len(L)",len(L))
            if j==0:
                L += Loctave.copy()
            else:
                print("ratio",ratio)
                L += self.upsample(Loctave[1:-1].copy(),ratio)
            ratio *= 2

        #print("len(L)",len(L))

        D = [L[i]-L[i-1] for i in range(1,len(L))] # remember list "comprehension" (https://www.programiz.com/python-programming/list-comprehension)

        self.D = D
        return L,D

    def detect_extrema(self, A):
        Apyr=np.stack(A) # build 3D array (image pyramid) from a list of 2D arrays

        # define the structuring element
        s1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) # or: np.zeros((3,3))
        s2 = np.array([[0, 0, 0], [0, -np.inf, 0], [0, 0, 0]]) # the central pixel set to -infty so that it will not be the maximum
        se = np.stack([s1, s2, s1])

        maxA = grey_dilation(Apyr,structure=se)
        minA = -grey_dilation(-Apyr,structure=-se) # or: minA = grey_erosion(Apyr,structure=se)

        [ss, ys, xs] = np.where(np.logical_or(Apyr > maxA, Apyr < minA))

        print(ss,ys,xs)
        # extrema points as list of tuples (scale, y, x) with list comprehension again
        self.extrema=[(ss[i], ys[i], xs[i]) for i in range(len(ss))]
        # alternative to list of tuples: build an array from separate lists (with vstack, we'll have one list per column)
        self.extrema=np.vstack((ss,ys,xs))
        print(self.extrema)
        return self.extrema

    def step2(self,im):
        # extrema detection (Sect. 3.1 in [Lowe04])
        self.step1(im) # we don't assume step1() has been called before step2()
        return self.detect_extrema(self.D)


    def remove_low_contrast(self,im,thr):
        # Note: Lowe proposes to remove the low-contrast keypoints *after* their accurate localisation
        # Since we have skipped this step, we consider the original candidate keypoints
        scales = self.extrema[0,:]
        ys = self.extrema[1,:]
        xs = self.extrema[2,:]
        # list comprehension and the usefulf zip (https://realpython.com/python-zip-function/)
        extrema_values=np.array([self.D[scale][y,x] for scale,y,x in zip(scales,ys,xs)])
        idx=np.where(extrema_values>thr)
        self.extrema=self.extrema[:,idx].squeeze()
        return self.extrema

        #print(self.D[scales][len(scales)*[0],len(scales)*[0]])#scales,[self.extrema[1,:],self.extrema[2,:]])

    def step3(self,im,thr):
        # remove low-contrast points (first part of Sect. 2 in [Lowe04])
        self.step2(im) # we don't assume step2() has been called before step3()
        self.remove_low_contrast(im,thr)
        return self.extrema

    def step4(self,im):
        pass
        # missing parts for detector [Lowe04]:
        # - accurate keypoint localization (Sect. 4)
        # - eliminating edge responses (Sect. 4.1)
        # - orientation assignment (Sect. 5)
        # plus descriptor (Sect. 6)