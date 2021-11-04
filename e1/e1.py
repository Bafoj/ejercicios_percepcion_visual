import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import glob


def HAN(im, n):
    h, _ = np.histogram(im,bins=n)
    h_normalized = h / np.sum(h)
    res =  np.tri(n).dot(h_normalized)
    # Equivalentes
    # res =  h_normalized.dot(np.tri(n)[::-1,::-1])
    # res =  h_normalized.dot(np.triu(np.ones((n,n))))
    return res

def multiHANhoriz(im, m, n):
    tam_franja = np.ceil(im.shape[0]/m).astype(int)
    return [HAN(im[tam_franja*i:tam_franja*(i+1),:],n) for i in range(m)]


def plotHANs(im, Hs):
    m = len(Hs)
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(m, 2, figure=fig)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(im, cmap='gray')
    for i in range(m):
        ax = fig.add_subplot(gs[i, 1])
        ax.bar(list(range(len(Hs[i]))), Hs[i], alpha=0.3)
        ax.plot(Hs[i], '--')

    fig.suptitle("Testing multiHANhoriz() with m = " + str(m) + " horizontal bands")
    plt.show(block=True)


def doTests(im):
    n_bins = 15
    # test with several horizontal divisions
    for num_franjas_horiz in [1, 2, 3, 5, 6]:
        assert H % num_franjas_horiz == 0, str(num_franjas_horiz) + "no es divisor del número de filas " + str(H)
        Hs = multiHANhoriz(im, num_franjas_horiz, n_bins)
        plotHANs(im, Hs)


if __name__ == "__main__":
    H = 300
    W = 125
    for bInvert in [False, True]:
        for imfile in glob.glob("./imgs/*"):
            im = Image.open(imfile).convert('L')
            im = im.resize((W, H), Image.ANTIALIAS)
            im = np.array(im)
            if bInvert:
                im = 255 - im
            doTests(im)