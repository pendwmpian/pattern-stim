import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt

RADIUS = 240

# Load the image in grayscale
image1 = cv2.imread('./data/input1.bmp', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./data/input2.bmp', cv2.IMREAD_GRAYSCALE)

def detectOuterCircle(image, radius=RADIUS):

    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

    edges = cv2.Canny(blurred, threshold1=20, threshold2=10)

    radius = RADIUS
    circle_template = np.zeros((radius * 2, radius * 2), dtype=np.uint8)
    cv2.circle(circle_template, (radius, radius), radius, 255, -1)

    result = cv2.matchTemplate(edges, circle_template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    center_x, center_y = max_loc[0] + radius, max_loc[1] + radius

    return center_x, center_y, radius

def cropCircle(image, x, y, r):
    """Crops a square region around the circle and masks outside the circle."""
    
    x1, y1 = x - r, y - r  
    x2, y2 = x + r, y + r  
    
    cropped_image = image[y1:y2, x1:x2].copy()
    
    mask = np.zeros_like(cropped_image, dtype=np.uint8)
    center = (r, r) 
    cv2.circle(mask, center, r, (255, 255, 255), thickness=-1)
    
    masked_image = cv2.bitwise_and(cropped_image, mask)

    return masked_image

def cropFOV(image, radius = RADIUS):
    """Crops a field of view."""

    center_x, center_y, _ = detectOuterCircle(image, radius)
    out = cropCircle(image, center_x, center_y, radius)
    return out

def background_filter(image, scale_factor=0.1, median_size=5, gaussian_sigma=5):
    """Reduces background reflection using downscaling, median filtering, and low-pass filtering."""
    
    small_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    small_image = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)

    median_filtered = median_filter(small_image, size=median_size)
    low_pass_filtered = gaussian_filter(median_filtered, sigma=gaussian_sigma)

    image_baseline = cv2.resize(low_pass_filtered, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    out = cv2.subtract(image, image_baseline)

    return out

def rotateImage(image, angle):
    """Rotates the image around its center by the specified angle."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return rotated_image

image = background_filter(image1)
cropped_image = cropFOV(image)
output_image = rotateImage(cropped_image, 30)

# Save and display the result
cv2.imwrite('output_image.jpg', output_image)
