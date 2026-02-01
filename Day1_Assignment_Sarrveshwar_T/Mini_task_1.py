import cv2
import matplotlib.pyplot as plt

def image_processing_pipeline(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # Convert to RGB for display and Grayscale for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blur, 100, 200)

    # Apply Binary Thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Display images in a 2x2 grid
    titles = ['Original Image', 'Gaussian Blur', 'Canny Edges', 'Binary Threshold']
    images = [img_rgb, blur, edges, thresh]

    plt.figure(figsize=(10, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray' if i > 0 else None)
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# image_processing_pipeline('input_image.jpg')
