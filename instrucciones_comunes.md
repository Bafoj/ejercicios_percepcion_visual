**Librerías base**

- numpy: `import numpy as np`
    - Arrays: `np.array(...), np.arange((1,10))`
    - Agregar o quitar axis: `a[:,np.newaxis]`
    - Camniar forma:`a.reshape(-1,3)` _-1 es un comodin_
    - Crear histograma: `np.histogram(array,bins)`
    - Crear transformada furier (fft): `from numpy import fft`

        ```py
        def FFT(im):
            return fft.fftshift(fft.fft2(im)) # se hace la fft visual para hacerlo de facil visualización
        def IFFT(ft):
            return fft.ifft2(fft.ifftshift(ft))
        ``` 
    - Recuerda que cualquier operación matemática se puede hacer mediante broadcasting
    - Los `True` y `False` actuan como **1** y **0**

- PIL: `from PIL import Image` -> `Image.read('path').convert('L')`

- scipy:
    - para hacer convoluciones: 
        ```py
        from scipy.ndimage import filters
        filters.convolve(im,mask)
        ```
    - filtros
        ```py
        from scipy.signal import medfilt2d # filtro media
        ```
    - gausiana
        ```py
        from scipy.signal import windows
        windows.gaussian(size,sigma)
        ```

- skimage:
    - canny:
        ```py
        from skimage import feature
        feature.canny(...)
        ```
    - hough space:
        ```py
        from skimage.transform import hough_line, hough_line_peaks  
        H, thetas, rhos = hough_line(edges, np.linspace(-np.pi/2, np.pi/2, numThetas))

        hough_line_peaks(H,num_peaks=nPeaksMax,angles=thetas,dists=rhos)
        ```
    - threshold otsu:
        ```py
        from skimage.filters import threshold_otsu
        thr = threshold_otsu(im)
        ```
    - morfologias:
        ```py
        from skimage.morphology import opening,closing, square...
        ```
    - Area conexa:
        ```py
        from skimage.measure import label,regionprops
        measure.label(binIm, background=0)
        regions:list[ measure._regionprops.RegionProperties] = measure.regionprops(labelIm)
        ```