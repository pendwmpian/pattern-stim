import cv2
import numpy as np
import matplotlib.pyplot as plt

RADIUS = 240

# Load the image in grayscale
image = cv2.imread('./data/input.bmp', cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

edges = cv2.Canny(blurred, threshold1=20, threshold2=10)

radius = RADIUS
circle_template = np.zeros((radius * 2, radius * 2), dtype=np.uint8)
cv2.circle(circle_template, (radius, radius), radius, 255, -1)

result = cv2.matchTemplate(edges, circle_template, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
center_x, center_y = max_loc[0] + radius, max_loc[1] + radius

# Draw the detected circle
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.circle(output_image, (center_x, center_y), radius, (0, 255, 0), 2)

# Save and display the result
cv2.imwrite('output_image.jpg', output_image)

