import numpy as np
from skimage.feature import ORB, match_descriptors, plot_matches, CENSURE
from PIL import Image
import matplotlib.pylab as plt
from skimage import transform
import load_cifar10 as cifar
import sift as sift
import sys
sys.path.append("../p1-4students")
import visualPercepUtils as vpu
from skimage import data
from skimage.color import rgb2gray

def draw_keypoints(im,locs,title=None):
    plt.imshow(im,cmap='gray')
    plt.plot(locs[:,1],locs[:,0],'sr')
    plt.axis('off')
    n_points = locs.shape[0]
    print(title, locs, n_points)
    if title is not None:
        str_points = str(n_points) if n_points>=0 else "no"
        print("str_points",str_points)
        plt.title(title+" ("+str_points+" points)")
    plt.show(block=True)

def extractORBkeypoints(im,num_points):
    orb_detector = ORB(n_keypoints=num_points)#,fast_threshold=0.1)
    orb_detector.detect(im)
    return orb_detector.keypoints

def extractORBdescriptors(im,num_points):
    orb_detector = ORB(n_keypoints=num_points)  # fast_threshold=0.4)
    orb_detector.detect_and_extract(im)
    return orb_detector.descriptors

def qualityDetection(dsc1,dsc2,matches):
    return np.sum(dsc1[matches[:,0]]==dsc2[matches[:,1]])

def display_matches(im1,im2,kp1,kp2,kp_matching):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    plt.gray()

    plot_matches(ax, im1, im2, kp1, kp2, matches=kp_matching)
    ax.axis('off')
    ax.set_title("Original Image vs. Transformed Image")
    plt.show(block=True)

def showExtremaSIFT(im,local_extrema,title):
    print(local_extrema.shape[1],"extrema detected")
    scales = np.unique(local_extrema[0,:]) # idx==0 for scale, idx==1 for y's and idx==1 for x's
    bDisplay=True
    if bDisplay:
        for scale in scales:
            points= local_extrema[1:, local_extrema[0, :] == scale] # get x and y rows for the required
            print(points.shape[1],"points in scale",scale)
            draw_keypoints(im,np.transpose(points),title+", scale "+str(scale))

def testSIFT():
    imfile='./imgs-P6/lena256.pgm'
    im = Image.open(imfile).convert('L')
    m,n = im.size
    im.resize((2*m,2*n))
    print("im.shape",im.size)
    im = np.array(im)/255.0

    mySIFT = sift.SIFT(scales=4,octaves=3)

    # Build scale-space and DoG pyramid
    L,D = mySIFT.step1(im)
    print("L.len",len(L))

    bDisplayScaleSpace=True
    if bDisplayScaleSpace:
        vpu.showInGrid( [im]+L,m=None,n=None,title="im and L")
        vpu.showInGrid( [im]+D,m=None,n=None,title="im and D")
        vpu.showInGrid(L+D,m=1,n=len(L),title="L and D")

    # Dectect local extrema
    local_extrema = mySIFT.step2(im)

    bDsiplayExtrema=True
    if bDsiplayExtrema:
        showExtremaSIFT(im,local_extrema,"Before excluding low-contrast candidates")

    # Filter our low-contrast points
    local_extrema=mySIFT.step3(im,thr=0.03)

    bDisplayFilteredExtrema=True
    if bDisplayFilteredExtrema:
        showExtremaSIFT(im,local_extrema,"After excluding low-contrast candidates")

def testORB():
    #Xtr,Ytr=cifar.load_CIFAR_batch()

    imfile='./imgs-P6/uji_biblio1.jpg'
    imfile='./imgs-P6/lena256.pgm'
    img1 = Image.open(imfile).convert('L')
    img1 = rgb2gray(data.camera()) # other available mages: astronaut, camera

    bResize=False
    if bResize:
        half = 0.25
        img1 = img1.resize( [int(half * s) for s in img1.size] )
    #img1 = Image.fromarray(Xtr[0]).convert('L')
    img1 = np.array(img1)
    print(img1.shape)
    #img2 = np.random.random_sample(img1.shape)
    angle=70
    img2 = transform.rotate(img1,angle=angle,clip=True)

    if False:
        plt.imshow(img1,cmap='gray')
        plt.show(block=True)

    num_points=40
    keypoints1 = extractORBkeypoints(img1,num_points)
    keypoints2 = extractORBkeypoints(img2,num_points)
    print(keypoints1,keypoints2)
    draw_keypoints(img1,keypoints1,"ORB keypoints on Image 1")
    draw_keypoints(img2,keypoints2,"ORB keypoints on Image 2")

    descriptors1 = extractORBdescriptors(img1,num_points)
    descriptors2 = extractORBdescriptors(img2,num_points)
    print(descriptors1.shape)

    metric='hamming'
    ratio=0.8

    # matching descriptors1 -> descriptors2
    matches12 = match_descriptors(descriptors1,descriptors2,cross_check=True)#metric=metric,cross_check=True,max_ratio=ratio)
    print("matches12",matches12)
    display_matches(img1,img2,keypoints1,keypoints2,matches12)

    print("Quality of detection:", qualityDetection(descriptors1,descriptors2,matches12))

    # matching descriptors2 -> descriptors1
    '''
    matches21 = match_descriptors(descriptors2,descriptors1,cross_check=True)#metric=metric,cross_check=True,max_ratio=ratio)
    print("matches21",matches21)
    display_matches(img2,img1,keypoints2,keypoints1,matches21)
    '''



if __name__ == "__main__":
    testSIFT()
    testORB()
