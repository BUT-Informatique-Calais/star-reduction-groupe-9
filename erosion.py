from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import cv2

fits_path = './examples/test_M31_linear.fits'
hdulist = fits.open(fits_path)
hdulist.info()

img_data = hdulist[0].data
img_header = hdulist[0].header

# Image couleur
if img_data.ndim == 3:
    if img_data.shape[0] == 3: 
        img_data = np.transpose(img_data, (1, 2, 0))
    img_norm = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    plt.imsave('./results/original.png', img_norm)
    # Conversion pour OpenCV
    img_cv = (img_norm * 255).astype(np.uint8)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
# Image monochrome
else:
    plt.imsave('./results/original.png', img_data, cmap='gray')
    img_cv = ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)

# Créer masque pour étoiles
# Passage en niveaux de gris si nécessaire
img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if img_data.ndim == 3 else img_cv

# Flou pour réduire le bruit au niveau du gris
img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

# Seuil adaptatif pour détecter les étoiles
star_mask = cv2.adaptiveThreshold(
    img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -2
)

cv2.imwrite('./results/star_mask.png', star_mask)

# Définition du noyau
erosion_kernel = np.ones((3,3), np.uint8)

# Réduction des étoiles par érosion
img_eroded = cv2.erode(img_cv, erosion_kernel, iterations=1)
cv2.imwrite('./results/eroded.png', img_eroded)

# Lissage du masque 
mask_smooth = cv2.GaussianBlur(star_mask, (5,5), 0)
mask_float = mask_smooth.astype(np.float32) / 255.0

# Adaptation du masque pour images avec couleur
if img_data.ndim == 3:
    mask_float = mask_float[:, :, np.newaxis]

# Interpolation finale : on mélange l'image érodée et l'originale selon le masque
final_img = (mask_float * img_eroded + (1 - mask_float) * img_cv).astype(np.uint8)
cv2.imwrite('./results/final.png', final_img)

hdulist.close()
print("Traitement terminé (Img dans results)")