import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt

RADIUS = 240
MINIMUM_ANGLE_UNIT = 1
ANGLE_MAX = 0

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
    return out

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

def overlay_images(image1, image2, angle, x_shift, y_shift, alpha=0.5):
    """Overlays two grayscale images with displacement and rotation.
    
    - image1 → Blue
    - image2 → Green (with transparency)
    - image2 is rotated and shifted by (angle, x_shift, y_shift)
    """

    # Get original image size
    h, w = image1.shape

    # Compute the new bounding box size after rotation
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_val = np.abs(rot_mat[0, 0])
    sin_val = np.abs(rot_mat[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    # Adjust rotation matrix to center the rotated image
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # Rotate image2
    rotated_image2 = cv2.warpAffine(image2, rot_mat, (new_w, new_h))

    # Calculate new canvas size to fit both images with shifts
    max_h = max(h, new_h + abs(y_shift))
    max_w = max(w, new_w + abs(x_shift))

    # Create extended blank canvases
    canvas1 = np.zeros((max_h, max_w), dtype=np.uint8)
    canvas2 = np.zeros((max_h, max_w), dtype=np.uint8)

    # Compute placement positions considering negative shifts
    img1_x, img1_y = max(0, -x_shift), max(0, -y_shift)
    img2_x, img2_y = max(0, x_shift), max(0, y_shift)

    # Place images on canvases
    canvas1[img1_y:img1_y + h, img1_x:img1_x + w] = image1
    canvas2[img2_y:img2_y + new_h, img2_x:img2_x + new_w] = rotated_image2

    # Convert images to 3-channel for color mapping
    blue_layer = cv2.merge([canvas1, np.zeros_like(canvas1), np.zeros_like(canvas1)])  # Blue
    green_layer = cv2.merge([np.zeros_like(canvas2), canvas2, np.zeros_like(canvas2)])  # Green

    # Blend images with transparency
    overlayed_image = cv2.addWeighted(blue_layer, 1, green_layer, alpha, 0)

    return overlayed_image

image1_cropped = cropFOV(image1)
cv2.imwrite('output_image_im1_FOV.jpg', image1_cropped)
image2_cropped = cropFOV(image2)

angle, displace = match_image(image1_cropped, image2_cropped)
print(angle, displace)

x_shift, y_shift = displace

output_image = overlay_images(image1, image2, angle, x_shift, y_shift, alpha=0.7)
cv2.imwrite('output_image.jpg', output_image)
# need more testcases

# Save and display the result
# cv2.imwrite('output_image.jpg', output_image)
