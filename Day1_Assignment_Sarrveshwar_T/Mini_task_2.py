import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_image(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return

    # Calculate statistics
    mean_val = np.mean(img)
    std_val = np.std(img)
    min_val, max_val = np.min(img), np.max(img)

    # Calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Grayscale Image")
    axes[0].axis('off')

    # Plot Histogram
    axes[1].plot(hist, color='black')
    axes[1].set_title("Pixel Intensity Histogram")
    axes[1].set_xlabel("Intensity")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim([0, 256])

    # Add statistics text
    stats_text = f"Mean: {mean_val:.2f}\nStd Dev: {std_val:.2f}\nMin: {min_val}\nMax: {max_val}"
    axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

# Example usage:
# analyze_image('path_to_your_image.jpg')
