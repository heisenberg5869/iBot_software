import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def process_pencil_sketch(image, blur_kernel):
    """
    Convert a BGR image into a B&W pencil sketch.
    
    Args:
        image: Input BGR image.
        blur_kernel: Integer (odd) for Gaussian blur.
    Returns:
        Grayscale sketch image.
    """
    try:
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Invert the grayscale image
        inverted = 255 - gray
        
        # Step 3: Apply Gaussian Blur
        blurred = cv2.GaussianBlur(inverted, (blur_kernel, blur_kernel), 0)
        
        # Step 4: Invert the blurred image
        inverted_blur = 255 - blurred
        
        # Step 5: Divide grayscale by inverted blur
        sketch = cv2.divide(gray, inverted_blur, scale=256.0)
        
        # No additional shading; return the raw sketch
        return np.clip(sketch, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error in pencil sketch processing: {e}")
        return None

def process_color_sketch(image, blur_kernel):
    """
    Convert a BGR image into a color pencil sketch.
    
    Args:
        image: Input BGR image.
        blur_kernel: Integer (odd) for Gaussian blur.
    Returns:
        BGR color sketch image.
    """
    try:
        # Step 1: Convert to HSV and extract channels
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv)
        
        # Step 2: Compute grayscale brightness (perceived luminance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply pencil sketch algorithm (divide + blur) on grayscale
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (blur_kernel, blur_kernel), 0)
        sketch_v = cv2.divide(gray, 255 - blurred, scale=256.0)
        
        # Step 3: Reduce saturation for a more pastel look
        s = cv2.multiply(s, 0.9)
        
        # Step 4: Merge channels (use sketch_v as V)
        final_hsv = cv2.merge([h, s.astype(np.uint8), sketch_v.astype(np.uint8)])
        
        # Step 4: Merge back and convert to BGR
        final_hsv = cv2.merge([h, s.astype(np.uint8), sketch_v.astype(np.uint8)])
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    except Exception as e:
        print(f"Error in color sketch processing: {e}")
        return None

class SketchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Pencil Sketcher")
        
        # Variables
        self.original_img = None
        self.processed_img = None
        self.file_path = ""
        
        # UI Elements
        self.btn_load = tk.Button(root, text="Select Image", command=self.load_image)
        self.btn_load.pack(pady=5)
        
        self.mode_var = tk.StringVar(value="BW")
        tk.Radiobutton(root, text="B&W Sketch", variable=self.mode_var, value="BW", command=self.update_preview).pack()
        tk.Radiobutton(root, text="Color Sketch", variable=self.mode_var, value="COLOR", command=self.update_preview).pack()
        
        tk.Label(root, text="Blur Kernel (Size):").pack()
        self.blur_slider = tk.Scale(root, from_=3, to=99, orient="horizontal", command=self.update_preview)
        self.blur_slider.set(21)
        self.blur_slider.pack()
        
        # Shadow strength slider removed; not needed for new algorithm
        
        self.canvas = tk.Label(root)
        self.canvas.pack(pady=10)
        
        self.btn_save = tk.Button(root, text="Save & Show Comparison", command=self.save_and_compare)
        self.btn_save.pack(pady=5)

    def load_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if self.file_path:
            self.original_img = cv2.imread(self.file_path)
            if self.original_img is None:
                messagebox.showerror("Error", "Could not decode image.")
                return
            self.update_preview()

    def update_preview(self, _=None):
        if self.original_img is None:
            return
        
        # Ensure blur is odd
        blur = self.blur_slider.get()
        if blur % 2 == 0: blur += 1
        
        if self.mode_var.get() == "BW":
            self.processed_img = process_pencil_sketch(self.original_img, blur)
            display_img = self.processed_img
        else:
            self.processed_img = process_color_sketch(self.original_img, blur)
            display_img = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)

        # Resize for preview
        h, w = display_img.shape[:2]
        max_size = 400
        scale = max_size / max(h, w)
        preview = cv2.resize(display_img, (int(w * scale), int(h * scale)))
        
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(preview))
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk

    def save_and_compare(self):
        if self.processed_img is None:
            return
            
        save_path = filedialog.asksaveasfilename(defaultextension=".png")
        if save_path:
            cv2.imwrite(save_path, self.processed_img)
            
            # Show Side-by-Side using Matplotlib
            orig_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            if self.mode_var.get() == "BW":
                proc_disp = self.processed_img
                cmap = 'gray'
            else:
                proc_disp = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
                cmap = None
                
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(orig_rgb)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title("Sketch Result")
            plt.imshow(proc_disp, cmap=cmap)
            plt.axis('off')
            plt.show()
            
            self.root.destroy() # Close GUI after saving one image

def main():
    try:
        user_input = input("How many images would you like to process? ")
        num_images = int(user_input)
    except ValueError:
        print("Invalid number. Exiting.")
        return

    for i in range(num_images):
        print(f"Processing image {i+1} of {num_images}...")
        root = tk.Tk()
        app = SketchGUI(root)
        root.mainloop()

if __name__ == "__main__":
    main()