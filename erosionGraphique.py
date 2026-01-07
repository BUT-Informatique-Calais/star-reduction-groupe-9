import tkinter as tk
from tkinter import filedialog, messagebox
from astropy.io import fits
import numpy as np
import cv2
from PIL import Image, ImageTk
import os

class FitsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Réduction d'étoiles - Images FITS")

        self.img_original = None
        self.img_cv = None

        # Bouton ouvrir
        tk.Button(root, text="Ouvrir la photo de l'image astronomique", command=self.load_fits).pack(pady=5)

        # Curseur
        tk.Label(root, text="Supprimer des étoiles").pack()
        self.star_slider = tk.Scale(
            root,
            from_=0,
            to=50,
            orient=tk.HORIZONTAL,
            length=400,
            label="Plus d'étoiles  ←→  Moins d'étoiles",
            command=self.process_image
        )
        self.star_slider.set(0)
        self.star_slider.pack(pady=5)

        # Zone d'affichage
        self.label = tk.Label(root)
        self.label.pack(padx=10, pady=10)

    def load_fits(self):
        path = filedialog.askopenfilename(filetypes=[("FITS files", "*.fits")])
        if not path:
            return

        hdulist = fits.open(path)
        img_data = hdulist[0].data
        hdulist.close()

        # Image couleur
        if img_data.ndim == 3 and img_data.shape[0] == 3:
            img_data = np.transpose(img_data, (1, 2, 0))
            img_norm = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            self.img_cv = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            img_norm = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            self.img_cv = (img_norm * 255).astype(np.uint8)

        # Sauvegarde de l'image originale
        self.img_original = self.img_cv.copy()

        self.star_slider.set(0)
        self.show_image(self.img_original)

    def process_image(self, value=None):
        if self.img_original is None:
            return

        # Intensité du curseur
        strength = int(self.star_slider.get() / 15)

        if strength == 0:
            self.show_image(self.img_original)
            return

        # Passage en niveaux de gris
        img_gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY) if self.img_original.ndim == 3 else self.img_original
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Masque étoiles
        star_mask = cv2.adaptiveThreshold(
            img_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, -2
        )

        # Érosion contrôlée
        kernel = np.ones((3, 3), np.uint8)
        img_eroded = cv2.erode(self.img_original, kernel, iterations=strength)

        # Lissage du masque
        mask_smooth = cv2.GaussianBlur(star_mask, (5, 5), 0)
        mask_float = mask_smooth.astype(np.float32) / 255.0

        if self.img_original.ndim == 3:
            mask_float = mask_float[:, :, np.newaxis]

        # Fusion finale
        final_img = (mask_float * img_eroded + (1 - mask_float) * self.img_original).astype(np.uint8)

        os.makedirs("results", exist_ok=True)
        cv2.imwrite("results/final.png", final_img)

        self.show_image(final_img)

    def show_image(self, img_cv):
        if img_cv.ndim == 2:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((600, 600))
        img_tk = ImageTk.PhotoImage(img_pil)

        self.label.configure(image=img_tk)
        self.label.image = img_tk


if __name__ == '__main__':
    root = tk.Tk()
    app = FitsGUI(root)
    root.mainloop()