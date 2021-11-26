import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter

from e2 import *

def load_cube_image():
    basewidth = 200
    img_color = Image.open("./imgs/IMG_20211104_141400.jpg")
    img = img_color.convert('L')
    print(img.size)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img, img_color = [im.resize((basewidth, hsize), Image.ANTIALIAS) for im in [img, img_color]]
    img, img_color = [np.array(im) for im in [img, img_color]]
    return img_color, img


def display_Hough_peaks(h, peaks, angles, dists, theta, rho, title):
    nThetas = len(theta)
    rangeThetas = theta[-1] - theta[0]
    slope = nThetas / rangeThetas
    plt.axis('off')
    plt.imshow(h, cmap='jet')
    for peak, angle, dist in zip(peaks, angles, dists):
        print("peak", peak, "at angle", np.rad2deg(angle), "and distance ", dist)
        # y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        # y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        # ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
        plt.plot(slope * (angle - theta[0]) + 1, dist - rho[0], 'rs',
                 markersize=0.1 * peak)  # size proportional to peak value
    plt.title(title)
    plt.show(block=True)


if __name__ == "__main__":
    # load and display the input image
    im_color, im = load_cube_image()
    plt.imshow(im, cmap='gray')
    plt.title('Original image')
    plt.show(block=True)

    # compute and display the edges
    edges = compute_edges(im)
    plt.imshow(im / 255 + edges, cmap='gray')
    plt.title('Edges and image')
    plt.show(block=True)

    # compute and display the Hough space
    H, thetas, rhos = compute_Hough_space_lines(edges)
    plt.imshow(np.log(H + 0.5), cmap='gray')
    plt.title('Original Hough space')
    plt.show(block=True)

    # filter the Hough space and keep the information of selected peaks
    selected_peaks, selected_rhos, selected_thetas = [], [], []
    for target_theta in DESIRED_ANGLES:
        H_current = filter_angles_hough(H, thetas, target_theta)
        peaks_values, peaks_thetas, peaks_rhos = find_peaks_hough(H_current, thetas, rhos, nPeaksMax=1)
        display_Hough_peaks(H_current, peaks_values, peaks_thetas, peaks_rhos, thetas, rhos,
                            "Filtered around " + str(target_theta) + " degrees")
        selected_peaks.append(peaks_values[0])
        selected_thetas.append(peaks_thetas[0])
        selected_rhos.append(peaks_rhos[0])

    # display the resulting lines
    bSave = True
    rho1, rho2 = selected_rhos
    theta1, theta2 = selected_thetas
    bg_imgs = {'im': 'original image (gray)',
               'im_color': 'original image (color)',
               'edges': 'edges'}
    for key in bg_imgs.keys():
        bg_img = eval(key)
        ax = display_lines(bg_img, selected_thetas, selected_rhos, selected_peaks)  # exercise
        plt.savefig('./imgs/result-'+key+'.png', bbox_inches='tight')
        plt.title('Resulting lines on ' + bg_imgs[key])
        plt.show(block=True)
