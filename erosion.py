from astropy.io import fits
import cv2 as cv
import numpy as np
from photutils.detection import DAOStarFinder
from astropy.stats import mad_std
from scipy.ndimage import gaussian_filter
import os

fits_file = './examples/HorseHead.fits'

hdul = fits.open(fits_file)
data = hdul[0].data
hdul.close()

# Si image couleur (3, H, W), transpose pour (H, W, 3)
if data.ndim == 3 and data.shape[0] == 3:
    data = np.transpose(data, (1, 2, 0))

if data.ndim == 3:
    image = np.stack([((data[:,:,i]-data[:,:,i].min())/np.ptp(data[:,:,i])*255).astype(np.uint8)
                      for i in range(3)], axis=2)
else:
    image = ((data - data.min()) / np.ptp(data) * 255).astype(np.uint8)

cv.imwrite('./results/original.png', cv.cvtColor(image, cv.COLOR_RGB2BGR) if image.ndim==3 else image)

# Calcul étoiles
data_2d = np.mean(data, axis=2) if data.ndim==3 else data
sigma = mad_std(data_2d)

# Seuil qui détecte les étoiles
sources = DAOStarFinder(fwhm=2.5, threshold=1.2*sigma)(data_2d)

# Masque pour étoile
mask = np.zeros(data_2d.shape, dtype=np.uint8)
if sources is not None:
    flux = sources['flux']
    flux_min = flux.min()
    flux_max = flux.max()
    for x, y, f in zip(sources['xcentroid'], sources['ycentroid'], flux):
        # Rayon proportionnel au flux pour couvrir toute l’étoile
        r = int(3 + (f - flux_min) / (flux_max - flux_min) * 7)  # rayon 3 à 10
        rr, cc = np.ogrid[:mask.shape[0], :mask.shape[1]]
        mask[(rr - int(y))**2 + (cc - int(x))**2 <= r**2] = 255

# Lissage du masque pour éviter les bords durs
mask = gaussian_filter(mask.astype(float), sigma=1.5)
mask = (mask > 1).astype(np.uint8) * 255  # masque binaire

if image.ndim == 2:
    final = cv.inpaint(image, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)
else:
    final = np.zeros_like(image)
    for c in range(3):
        final[:,:,c] = cv.inpaint(image[:,:,c], mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)

cv.imwrite('./results/final.png', cv.cvtColor(final, cv.COLOR_RGB2BGR) if final.ndim==3 else final)
