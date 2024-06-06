import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np

class ColorBalanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Balance Adjuster")

        self.image = None
        self.original_image = None  # Added initialization for original image
        self.img_label = tk.Label(root)
        self.img_label.pack()

        self.load_btn = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_btn.pack()

        self.show_btn = tk.Button(root, text="Show Image", command=self.show_image, state=tk.DISABLED)
        self.show_btn.pack()

        self.red_scale = tk.Scale(root, from_=0, to=3, resolution=0.01, orient=tk.HORIZONTAL, label="Red Balance")
        self.red_scale.pack()

        self.green_scale = tk.Scale(root, from_=0, to=3, resolution=0.01, orient=tk.HORIZONTAL, label="Green Balance")
        self.green_scale.pack()

        self.blue_scale = tk.Scale(root, from_=0, to=3, resolution=0.01, orient=tk.HORIZONTAL, label="Blue Balance")
        self.blue_scale.pack()

        self.red_scale.set(1.0)
        self.green_scale.set(1.0)
        self.blue_scale.set(1.0)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.original_image = self.image.copy()  # Keep a copy of the original image
            self.show_btn.config(state=tk.NORMAL)  # Enable the "Show Image" button

    def show_image(self):
        if self.image is not None:
            self.image = self.update_image(self)
            self.display_image(self.image)

    def update_image(self, _=None):
        if self.image is not None:
            r_factor = self.red_scale.get()
            g_factor = self.green_scale.get()
            b_factor = self.blue_scale.get()

            # Apply the color balance factors to each channel
            self.image[:, :, 2] = np.clip(self.original_image[:, :, 2] * r_factor, 0, 255)  # Red channel
            self.image[:, :, 1] = np.clip(self.original_image[:, :, 1] * g_factor, 0, 255)  # Green channel
            self.image[:, :, 0] = np.clip(self.original_image[:, :, 0] * b_factor, 0, 255)  # Blue channel

            # Convert the image to RGB format for display
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk
            
            return self.image

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorBalanceApp(root)
    root.mainloop()