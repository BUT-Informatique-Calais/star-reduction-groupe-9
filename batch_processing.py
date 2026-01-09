"""
Lancer le fichier : python3 batch_processing.py --input examples --output results --strength 3
"""

import argparse
import os
import numpy as np
import cv2
from astropy.io import fits
from tqdm import tqdm

def normalize(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min())

def star_reduction(img, strength):
    """
    Réduction d'étoiles
    """

    if strength == 0:
        return img.copy()
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    star_mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2
    )
    kernel = np.ones((3,3), np.uint8)
    img_eroded = cv2.erode(img, kernel, iterations=strength)
    mask_float = cv2.GaussianBlur(star_mask, (5,5),0).astype(np.float32)/255.0

    if img.ndim == 3:
        mask_float = mask_float[:,:,None]
    return (mask_float*img_eroded + (1-mask_float)*img).astype(np.uint8)

def process_fits_file(input_path, output_folder, strength):
    """
    Traitement d'un dossier FITS et sauvegarde 2 image : original + final
    """

    hdul = fits.open(input_path)
    data = hdul[0].data
    hdul.close()

    data = normalize(data)

    # Convertir en uint8 pour OpenCV
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.transpose(data, (1, 2, 0))
        img = cv2.cvtColor((data*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        img = (data*255).astype(np.uint8)

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Sauvegarde de l'image basique
    original_png = os.path.join(output_folder, f"{base_name}_original.png")
    if img.ndim == 3:
        cv2.imwrite(original_png, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(original_png, img)

    # Image avec les étoiles enlevé
    reduced = star_reduction(img, strength)
    reduced_png = os.path.join(output_folder, f"{base_name}_final.png")
    if reduced.ndim == 3:
        cv2.imwrite(reduced_png, cv2.cvtColor(reduced, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(reduced_png, reduced)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Dossier avec tous les FITS")
    parser.add_argument("--output", required=True, help="Dossier de sortie")
    parser.add_argument("--strength", type=int, default=1, help="Force de réduction")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    fits_files = [f for f in os.listdir(args.input) if f.lower().endswith(".fits")]
    if not fits_files:
        print("Pas de FITS trouvé dans :", args.input)
        return

    for fname in tqdm(fits_files, desc="Traitement des images"):
        process_fits_file(os.path.join(args.input, fname), args.output, args.strength)

    print("\nTraitement terminé. images originaux et réduits dans :", args.output)

if __name__ == "__main__":
    main()
