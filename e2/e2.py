'''
Tus imports (los necesarios, no más)
'''
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import canny
from skimage.transform import hough_line,hough_line_peaks


def compute_edges(im):
   return canny(im,sigma=4,low_threshold=3,high_threshold=5,use_quantiles=False)


def compute_Hough_space_lines(edges):
    return hough_line(edges, theta=np.linspace(-np.pi / 2, np.pi / 2, 220))


def filter_angles_hough(H, angles, target_angle, margin=5):
    H2 = H.copy()
    inf,sup = np.deg2rad(target_angle-margin),np.deg2rad(target_angle+margin)
    targets = np.logical_or(angles < inf,angles > sup)
    H2[:,targets] = H2[:,targets] * 0.3
    return H2


def find_peaks_hough(H, thetas, rhos, nPeaksMax):
    return hough_line_peaks(H,num_peaks=nPeaksMax,angles=thetas,dists=rhos)


def display_lines(im, thetas, rhos, values):
    if len(im.shape) > 2:
        H, W, C = im.shape
    else:
        H, W = im.shape
    print(W, H)
    plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap='gray')
    scale = 0.05
    for theta, rho, strength in zip(thetas, rhos, values):
        color = 'red' if theta > 0 else 'blue'
        s = np.sin(theta)
        c = np.cos(theta)
        if np.abs(s) < np.abs(c):
            x0 = (rho - 0 * s) / c
            x1 = (rho - H * s) / c
            x0, x1 = [min(W, max(x, 0)) for x in [x0, x1]]
            plt.plot((x0,x1), (0, H), 'y.-', linewidth=scale * strength, color=color)
        else:
            y0 = (rho - W * c) / s
            y1 = (rho - 0 * c) / s
            y0, y1 = [min(W, max(y, 0)) for y in [y0, y1]]
            plt.plot((W,0),(y0,y1), 'y.-', linewidth=scale * strength, color=color)

    return plt.gca()


DESIRED_ANGLES = [-46,46]  # en grados

'''
-------------------------------------------------------------------------------
Tu respuesta a 7a:

Motivo 1.
    Uno de los principales motivos por los que creo que no es del todo preciso 
    se debe a que se ha utilizado canny para detectar los bordes. Esta técnica al
    usar una gausiana para suavizar tiende a alterar ligeramente las posiciones de
    los bordes.

Motivo 2.
    Viendo los bordes de la imagen binaria, se puede observar que las líneas
    negras que conforman los bordes no los ha detectado como tal, sino que
    se han detectado los bordes de las casillas del cubo los cuales están
    ligeramente desplazados.




-------------------------------------------------------------------------------
Tu respuesta a 7b:

Los aspectos del código que hacen que la solución sea tan específica son:
    - El listado de ángulos 'DESIRED_ANGLES' contiene ángulos específicos
      asociados al posicionamiento del cubo en la imagen.
    - En el método compute_edges los valores del detector de Canny pueden
      variar en función de las características de la foto
      (luz, ruido, distancia...)
    - En el método compute_Hough_space_lines se utiliza el método hough_line. Este recibe
      un valor fijo para theta y esto puede variar en función del posicionamiento del objeto
      en la foto.



-------------------------------------------------------------------------------
Tu respuesta a 7c:

Dado el hipotético caso en el que se quiera hacer un programa que a partir de
una secuencia de fotos indique los pasos para resolver un cubo minimizando los
movimientos, la detección de estas aristas podría servir para poder diferenciar
las caras del cubo de Rubik para así facilitar su análisis.



-------------------------------------------------------------------------------
'''
