"""
Lancer le fichier : python3 erosionGraphique.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
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
        self.current_img = None

        # Bouton ouvrir
        tk.Button(
            root,
            text="Ouvrir la photo de l'image astronomique",
            command=self.load_fits
        ).pack(pady=5)

        # Bouton afficher différence scientifique
        tk.Button(
            root,
            text="Afficher différence scientifique",
            command=self.show_difference
        ).pack(pady=5)

        tk.Button(
            root,
            text="Comparer Avant/Après",
            command=lambda: self.blink_images()
        ).pack(pady=5)


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

        # Bouton sauvegarde
        tk.Button(
            root,
            text="Enregistrer l'image",
            command=self.save_current_image
        ).pack(pady=5)

        # Zone d'affichage
        self.label = tk.Label(root)
        self.label.pack(padx=10, pady=10)

    def load_fits(self):
        # Ouvre un affichage pour sélectionner un fichier FITS
        path = filedialog.askopenfilename(filetypes=[("FITS files", "*.fits")])
        if not path:
            return

        hdulist = fits.open(path)
        img_data = hdulist[0].data
        hdulist.close()

        if img_data.ndim == 3 and img_data.shape[0] == 3:
            img_data = np.transpose(img_data, (1, 2, 0)) # Réorganisation des axes
            img_norm = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            self.img_cv = cv2.cvtColor(
                (img_norm * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR
            )
        else:
            img_norm = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            self.img_cv = (img_norm * 255).astype(np.uint8)

        self.img_original = self.img_cv.copy()
        self.current_img = self.img_original.copy()

        self.star_slider.set(0)
        self.show_image(self.current_img)

    def process_image(self, value=None):
        if self.img_original is None:
            return

        # Calcule la force de réduction des étoiles selon la valeur du slider
        strength = int(self.star_slider.get() / 15)

        # On affiche l'image originale si force est 0
        if strength == 0:
            self.current_img = self.img_original.copy()
            self.show_image(self.current_img)
            return

        img_gray = (
            cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
            if self.img_original.ndim == 3
            else self.img_original
        )

        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        star_mask = cv2.adaptiveThreshold(
            img_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, -2
        )

        # Érosion pour réduire la taille des étoiles
        kernel = np.ones((3, 3), np.uint8)
        img_eroded = cv2.erode(
            self.img_original,
            kernel,
            iterations=strength
        )

        mask_smooth = cv2.GaussianBlur(star_mask, (5, 5), 0)
        mask_float = mask_smooth.astype(np.float32) / 255.0

        if self.img_original.ndim == 3:
            mask_float = mask_float[:, :, np.newaxis]

        final_img = (
            mask_float * img_eroded +
            (1 - mask_float) * self.img_original
        ).astype(np.uint8)

        self.current_img = final_img
        self.show_image(self.current_img)

    def save_current_image(self):
        if self.current_img is None:
            messagebox.showwarning(
                "Aucune image",
                "Aucune image à enregistrer."
            )
            return

        os.makedirs("save", exist_ok=True)

        slider_value = self.star_slider.get()

        filename = simpledialog.askstring(
            "Nom du fichier",
            "Entrez le nom du fichier (laisser vide pour nom automatique) :"
        )

        # Si l'utilisateur annule ou laisse vide
        if not filename:
            filename = f"{slider_value}erosion.png"
        else:
            if not filename.lower().endswith(".png"):
                filename += ".png"

        filepath = os.path.join("save", filename)

        cv2.imwrite(filepath, self.current_img)

        messagebox.showinfo(
            "Image enregistrée",
            f"L'image a été enregistrée avec succès :\n\n{filepath}"
        )

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

    def show_difference(self):
        if self.img_original is None or self.current_img is None:
            return

        # Calcul de la différence pixel par pixel
        diff = cv2.absdiff(self.img_original, self.current_img)

        if diff.ndim == 3:
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            diff_gray = diff.copy()

        diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)

        diff_color = np.zeros(
            (diff_norm.shape[0], diff_norm.shape[1], 3),
            dtype=np.uint8
        )
        diff_color[:, :, 0] = diff_norm // 2
        diff_color[:, :, 1] = diff_norm // 3
        diff_color[:, :, 2] = 0
        # Met en rouge  les pixels où la différence est très grande
        diff_color[diff_norm > 180] = [255, 0, 0]

        os.makedirs("results", exist_ok=True)
        cv2.imwrite("results/etoiles.png", diff_color)

        self.show_image(diff_color)


    def blink_images(self, n=6, interval=500):
        """
        Alterne l'affichage entre l'image originale et l'image traitée.
        n : nombre total d'alternances
        interval : temps entre alternances en ms
        """
        if self.img_original is None or self.current_img is None:
            return

        if n == 0:
            self.show_image(self.current_img)
            return

        # Alterne l'affichage
        img_to_show = self.img_original if n % 2 == 0 else self.current_img
        self.show_image(img_to_show)

        # Appelle récursivement après "interval" ms
        self.root.after(interval, lambda: self.blink_images(n-1, interval))



if __name__ == '__main__':
    root = tk.Tk()
    app = FitsGUI(root)
    root.mainloop()


