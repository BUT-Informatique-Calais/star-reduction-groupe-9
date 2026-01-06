from astropy.io import fits
import cv2 as cv
import numpy as np
from photutils.detection import DAOStarFinder
from astropy.stats import mad_std
from scipy.ndimage import gaussian_filter
import os

# Open and read the FITS file
fits_file = './examples/HorseHead.fits'
hdul = fits.open(fits_file)
data = hdul[0].data
hdul.close()

# Si image couleur (3, H, W), transpose pour (H, W, 3)
if data.ndim == 3 and data.shape[0] == 3:
    data = np.transpose(data, (1, 2, 0))

if data.ndim == 3:
    # Color image - need to transpose to (height, width, channels)
    if data.shape[0] == 3:  # If channels are first: (3, height, width)
        data = np.transpose(data, (1, 2, 0))
    # If already (height, width, 3), no change needed
    
    # Normalize the entire image to [0, 1] for matplotlib
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Save the data as a png image (no cmap for color images)
    plt.imsave('./results/original.png', data_normalized)
    
    # Normalize each channel separately to [0, 255] for OpenCV
    image = np.zeros_like(data, dtype='uint8')
    for i in range(data.shape[2]):
        channel = data[:, :, i]
        image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
else:
    # Monochrome image
    plt.imsave('./results/original.png', data, cmap='gray')
    
    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')



# Define a kernel for erosion
kernel = np.ones((3,3), np.uint8)
# Perform erosion
eroded_image = cv.erode(image, kernel, iterations=1)

# Save the eroded image 
cv.imwrite('./results/eroded.png', eroded_image)

# Close the file
hdul.close()