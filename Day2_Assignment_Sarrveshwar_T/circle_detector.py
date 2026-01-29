import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path):
    """
    Load and preprocess image for circle detection.
    
    Args:
        image_path: Path to input image
        
    Returns:
        tuple: (original_color, preprocessed_gray) or (None, None)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    return img, blurred

def detect_circles(gray_image, dp=1, minDist=50, param1=80,
                   param2=40, minRadius=10, maxRadius=100):
    """
    Detect circles using Hough Circle Transform.
    
    Args:
        gray_image: Preprocessed grayscale image
        dp: Inverse accumulator resolution ratio
        minDist: Minimum distance between circle centers
        param1: Upper Canny threshold
        param2: Accumulator threshold
        minRadius: Minimum circle radius
        maxRadius: Maximum circle radius
        
    Returns:
        numpy array of circles (x, y, radius) or None
    """
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    return circles


def classify_circle_size(radius, min_r, max_r):
    """
    Bonus 4: Size Classification Logic
    Returns color key based on relative size.
    """
    if max_r == min_r:
        return 'medium'
    
    # Define thresholds at 33% and 66% of the range
    range_val = max_r - min_r
    t1 = min_r + range_val * 0.33
    t2 = min_r + range_val * 0.66
    
    if radius < t1:
        return 'small'
    elif radius < t2:
        return 'medium'
    else:
        return 'large'

def visualize_circles(image, circles, save_path=None):
    """
    Draw detected circles on image and display.
    
    Args:
        image: Original color image
        circles: Array of detected circles
        save_path: Optional path to save annotated image
    """
    colors = {
        'small': (0, 255, 0),    # Green for small
        'medium': (255, 255, 0), # Cyan/Yellow-ish for medium
        'large': (0, 0, 255)     # Red for large
    }

    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Calculate radius stats for classification only if multiple circles exist
        radii = circles[0, :, 2]
        if len(radii) > 0:
            min_r, max_r = np.min(radii), np.max(radii)
        
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # Bonus 4: Size Classification
            if len(radii) > 0:
                size_cat = classify_circle_size(radius, min_r, max_r)
                color = colors.get(size_cat, (0, 255, 0))
            else:
                color = (0, 255, 0)
                
            # Draw the outer circle
            cv2.circle(image, center, radius, color, 2)
            # Draw the center of the circle
            cv2.circle(image, center, 2, (0, 0, 255), 3)

            # Optional: Add text label
            # cv2.putText(image, size_cat[0].upper(), (i[0]+5, i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Display the result
    cv2.imshow("Detected Circles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result if save_path is provided
    if save_path is not None:
        cv2.imwrite(save_path, image)

def calculate_statistics(circles):
    """
    Calculate and display statistics about detected circles.
    
    Args:
        circles: Array of detected circles
        
    Returns:
        dict: Statistics dictionary
    """
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Calculate statistics
        num_circles = len(circles[0, :])
        radii = circles[0, :, 2]
        total_radius = np.sum(radii)
        mean_radius = total_radius / num_circles
        
        # Calculate size breakdown
        counts = {'small': 0, 'medium': 0, 'large': 0}
        if len(radii) > 0:
            min_r, max_r = np.min(radii), np.max(radii)
            
            for r in radii:
                size_cat = classify_circle_size(r, min_r, max_r)
                counts[size_cat] += 1
        
        # Create statistics dictionary
        stats = {
            'num_circles': num_circles,
            'total_radius': total_radius,
            'mean_radius': mean_radius,
            'counts': counts
        }
        
        return stats
    else:
        return None

def interactive_gui(original_image, gray_image):
    """
    Bonus 2: Interactive GUI
    """
    window_name = "Circle Detection GUI"
    cv2.namedWindow(window_name)
    
    # Callback for trackbars (does nothing, we just read values in loop)
    def nothing(x):
        pass

    # Create Trackbars
    # Param1: Canny edge threshold
    # Param2: Accumulator threshold
    cv2.createTrackbar("Param1", window_name, 50, 255, nothing)
    cv2.createTrackbar("Param2", window_name, 30, 100, nothing)
    cv2.createTrackbar("MinDist", window_name, 20, 100, nothing)
    
    print("\n--- Interactive Mode ---")
    print("Adjust sliders to tune parameters.")
    print("Press 'q' to exit and finalize.")
    print("Press 's' to save the current result image.")

    final_circles = None

    while True:
        # Get current positions of trackbars
        p1 = cv2.getTrackbarPos("Param1", window_name)
        p2 = cv2.getTrackbarPos("Param2", window_name)
        min_dist = cv2.getTrackbarPos("MinDist", window_name)
        
        # Ensure parameters are valid (avoid crash on 0)
        p1 = max(1, p1)
        p2 = max(1, p2)
        min_dist = max(1, min_dist)

        # Detect circles
        circles = detect_circles(gray_image, param1=p1, param2=p2, minDist=min_dist)
        final_circles = circles # Update final result
        
        # Create a copy to draw on
        display_img = original_image.copy()
        
        if circles is not None:
            c = np.uint16(np.around(circles))
            radii = c[0, :, 2]
            min_r, max_r = (np.min(radii), np.max(radii)) if len(radii) > 0 else (0,0)
            
            colors = {'small': (0, 255, 0), 'medium': (255, 255, 0), 'large': (0, 0, 255)}
            
            for i in c[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                size_cat = classify_circle_size(radius, min_r, max_r)
                color = colors.get(size_cat, (0, 255, 0))
                cv2.circle(display_img, center, radius, color, 2)
                cv2.circle(display_img, center, 2, (0, 0, 255), 3)

            cv2.putText(display_img, f"Circles: {len(c[0])}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else:
            cv2.putText(display_img, "No circles", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the current image (manual save)
            filename = "detected_circles_result.jpg"
            cv2.imwrite(filename, display_img)
            print(f"Image saved as {filename}")
            
    cv2.destroyAllWindows()
    return final_circles, display_img

def main():
    """Main function."""
    print("Circle Detection")
    
    # Path to the input image
    image_path = input(r"Input the path of image : ")
    # Strip quotes if user added them
    image_path = image_path.strip('"').strip("'")
    image_path = os.path.abspath(image_path)
    
    # Load and preprocess image
    original_image, gray_image = preprocess_image(image_path)
    
    if original_image is None:
        return
    
    print("\nStarting Interactive GUI...")
    circles, final_image = interactive_gui(original_image, gray_image)
    
    # Calculate statistics
    stats = calculate_statistics(circles)
    if stats is not None:
        print("\nFinal Statistics:")
        print(f"Number of circles: {stats['num_circles']}")
        print(f"Total radius: {stats['total_radius']}")
        print(f"Mean radius: {stats['mean_radius']:.2f}")
        print(f"Small circles: {stats['counts']['small']}")
        print(f"Medium circles: {stats['counts']['medium']}")
        print(f"Large circles: {stats['counts']['large']}")
        
        # Save the result image
        dir_name = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        save_name = f"result_{base_name}"
        save_path = os.path.join(dir_name, save_name)
        
        if final_image is not None:
            cv2.imwrite(save_path, final_image)
            print(f"Result image saved as: {save_path}")
    else:
        print("No circles detected in final step.")

if __name__ == '__main__':
    main()
