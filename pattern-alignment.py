import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt

RADIUS = 240
MINIMUM_ANGLE_UNIT = 1
ANGLE_MAX = 10

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

def fft_cross_correlation(image1, image2):
    """Computes the cross-correlation of two images using FFT to find displacement (x, y)."""

    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # Normalize images
    image1 = (image1 - np.mean(image1)) / np.std(image1)
    image2 = (image2 - np.mean(image2)) / np.std(image2)

    fft1 = np.fft.fft2(image1)
    fft2 = np.fft.fft2(image2)

    cross_power_spectrum = (fft1 * np.conj(fft2)) / np.abs(fft1 * np.conj(fft2) + 1e-10)

    cross_corr = np.fft.ifft2(cross_power_spectrum).real

    corr = np.max(cross_corr)
    y_shift, x_shift = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)

    if y_shift > image1.shape[0] // 2:
        y_shift -= image1.shape[0]
    if x_shift > image1.shape[1] // 2:
        x_shift -= image1.shape[1]

    return corr, x_shift, y_shift

def match_image(im1, im2):
    "Tilt (θ) and displacement (x, y) between two images"

    angles = np.arange(-1 * ANGLE_MAX, ANGLE_MAX + MINIMUM_ANGLE_UNIT, MINIMUM_ANGLE_UNIT)
    corrs = []
    coord = []

    for angle in angles:
        tilted_image = rotateImage(im2, angle)
        corr, x, y = fft_cross_correlation(im1, tilted_image)
        corrs.append(corr)
        coord.append((x, y))

    arg_match = np.argmax(corrs)
    return angles[arg_match], coord[arg_match]

image1_smoothed = background_filter(image1)
image1_cropped = cropFOV(image1_smoothed)
image2_smoothed = background_filter(image2)
image2_cropped = cropFOV(image2_smoothed)

cv2.imwrite('output_image_cropped1.jpg', image1_cropped)
cv2.imwrite('output_image_cropped2.jpg', image2_cropped)

angle, displace = match_image(image1_cropped, image2_cropped)
print(angle, displace)

# need more testcases

# Save and display the result
# cv2.imwrite('output_image.jpg', output_image)
