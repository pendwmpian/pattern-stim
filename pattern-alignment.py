import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt

RADIUS = 240
MINIMUM_ANGLE_UNIT = 1
ANGLE_MAX = 10

# Load the image in grayscale
image1 = cv2.imread('./data/input4.bmp', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./data/input5.bmp', cv2.IMREAD_GRAYSCALE)

def detectOuterCircle(image, radius=RADIUS):

    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

    edges = cv2.Canny(blurred, threshold1=20, threshold2=30)
    cv2.imwrite('output_image_im1_edge.jpg', edges)

    radius = RADIUS
    circle_template = np.zeros((radius * 2, radius * 2), dtype=np.uint8)
    cv2.circle(circle_template, (radius, radius), radius, 255, 2)

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
    return out, (center_x, center_y)

def cropSquare(image, radius = RADIUS):
    r = radius
    wid = int(r / np.sqrt(2))
    x1, y1 = r - wid, r - wid
    x2, y2 = r + wid, r + wid
    
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image

def background_filter(image, scale_factor=0.1, median_size=5, gaussian_sigma=5):
    """Reduces background reflection using downscaling, median filtering, and low-pass filtering."""
    
    small_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    small_image = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)

    median_filtered = median_filter(small_image, size=median_size)
    low_pass_filtered = gaussian_filter(median_filtered, sigma=gaussian_sigma)

    image_baseline = cv2.resize(low_pass_filtered, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    out = cv2.subtract(image, image_baseline)

    return out

def sigmoid_contrast(image, alpha=10, beta=0.5):
    """Applies sigmoid contrast adjustment to an image.
    """
    image_norm = image.astype(np.float32) / np.max(image)

    adjusted = 1 / (1 + np.exp(-alpha * (image_norm - beta)))

    adjusted = (adjusted * 255).astype(np.uint8)

    return adjusted

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

    cv2.imwrite('output_image_im1_std.jpg', image1)

    # Normalize images
    image1 = (image1 - np.mean(image1)) / np.std(image1)
    image2 = (image2 - np.mean(image2)) / np.std(image2)

    fft1 = np.fft.fft2(image1)
    fft2 = np.fft.fft2(image2)

    cross_power_spectrum = (fft1 * np.conj(fft2))

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
    square_image1 = cropSquare(im1)
    im1_smoothed = background_filter(square_image1)
    im1_smoothed = sigmoid_contrast(im1_smoothed)
    hist, bins = np.histogram(im1_smoothed, 256)
    plt.plot(bins[:-1], hist)
    plt.savefig('output_image_hist.jpg')
    cv2.imwrite('output_image_im1.jpg', im1_smoothed)
    im1_smoothed = cv2.convertScaleAbs(im1_smoothed, alpha=3.5, beta=3)

    for angle in angles:
        tilted_image = rotateImage(im2, angle)
        square_tilted_image = cropSquare(tilted_image)
        im2_smoothed = background_filter(square_tilted_image)
        im2_smoothed = sigmoid_contrast(im2_smoothed)
        corr, x, y = fft_cross_correlation(im1_smoothed, im2_smoothed)
        cv2.imwrite('output_image_im2.jpg', im2_smoothed)
        corrs.append(corr)
        coord.append((x, y))

    arg_match = np.argmax(corrs)
    return angles[arg_match], coord[arg_match]

def overlay_images(image1, image2, center1, center2, angle, x_shift, y_shift, alpha=0.5):
    """Overlays two grayscale images after aligning their given rotation centers.
    
        image1 → Blue
        image2 → Green

    """

    h1, w1 = image1.shape
    h2, w2 = image2.shape

    # Compute shift required to align the given rotation centers
    center_dx = center1[0] - center2[0]
    center_dy = center1[1] - center2[1]

    # Create larger canvas to accommodate shifts
    max_h = max(int(h1 * 1.4), int(h2 * 1.4) + abs(y_shift))
    max_w = max(int(w1 * 1.4), int(w2 * 1.4) + abs(x_shift))

    canvas1 = np.zeros((max_h, max_w), dtype=np.uint8)
    canvas2 = np.zeros((max_h, max_w), dtype=np.uint8)

    # Position image1 and image2 on the canvas based on center alignment
    img1_x, img1_y = max(0, -x_shift - center_dx), max(0, -y_shift - center_dy)
    img2_x, img2_y = max(0, x_shift + center_dx), max(0, y_shift + center_dy)

    canvas1[img1_y:img1_y + h1, img1_x:img1_x + w1] = image1
    canvas2[img2_y:img2_y + h2, img2_x:img2_x + w2] = image2

    # Rotate image2 around its given center
    rotation_matrix = cv2.getRotationMatrix2D(center2, angle, 1.0)
    rotated_image2 = cv2.warpAffine(canvas2, rotation_matrix, (max_w, max_h))

    # Convert images to 3-channel for color mapping
    blue_layer = cv2.merge([canvas1, np.zeros_like(canvas1), np.zeros_like(canvas1)])  # Blue
    green_layer = cv2.merge([np.zeros_like(rotated_image2), rotated_image2, np.zeros_like(rotated_image2)])  # Green

    # Blend images with transparency
    overlayed_image = cv2.addWeighted(blue_layer, 1, green_layer, alpha, 0)

    return overlayed_image


image1_cropped, center_im1 = cropFOV(image1)
image2_cropped, center_im2 = cropFOV(image2)

angle, displace = match_image(image1_cropped, image2_cropped)
print(angle, displace)

x_shift, y_shift = displace

output_image = overlay_images(image1, image2, center_im1, center_im2, angle, x_shift, y_shift, alpha=0.7)
cv2.imwrite('output_image.jpg', output_image)
# need more testcases

# Save and display the result
# cv2.imwrite('output_image.jpg', output_image)
