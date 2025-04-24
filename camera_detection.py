import cv2
import numpy as np
import csv

# --- Configuration ---
video_path = './data/video2.mp4'
output_video = './data/tracking_output2.mp4'

# Video cropping bounds
height_min = 520
height_max = 600
x_min = 500
x_max = 1300

# CLAHE parameters
clahe_clip = 2.0
clahe_tiles = (8, 8)

# Adaptive threshold parameters
block_size = 41  # must be odd
c_thresh = 2

# Morphology and contour filtering
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
min_contour_area = 60  # adjust to your animal size

# ---------------------

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

# Get properties
fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare CLAHE
clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tiles)

# For saving output
writer = None

positions = []     # list of (frame_idx, x, y)
prev_pos = None    # previous centroid
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop region of interest
    frame = frame[height_min:height_max, x_min:x_max]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    # Clip bright values; animal is very dark
    _, gray = cv2.threshold(gray, 40, 255, cv2.THRESH_TRUNC)

    # 1) Local contrast enhancement
    equalized = clahe.apply(gray)

    # 2) Adaptive threshold (invert: animal dark -> white)
    th = cv2.adaptiveThreshold(equalized, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               block_size, c_thresh)

    # 3) Morphological cleaning
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4) Contour detection
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        candidates.append(((cx, cy), area))

    # 5) Select the most plausible blob
    chosen_pos = None
    if candidates:
        if prev_pos is None:
            # first frame: largest
            chosen_pos = max(candidates, key=lambda x: x[1])[0]
        else:
            # closest to previous
            chosen_pos = min(candidates,
                             key=lambda x: np.hypot(x[0][0] - prev_pos[0],
                                                    x[0][1] - prev_pos[1]))[0]

    # 6) Record & visualize
    if chosen_pos is not None:
        positions.append((frame_idx, chosen_pos[0], chosen_pos[1]))
        prev_pos = chosen_pos
        cv2.circle(frame, chosen_pos, 6, (0, 0, 255), -1)

    # Prepare side-by-side display
    mask_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    vis = np.hstack((frame, thresh_bgr, mask_bgr))

    # Initialize writer once we know frame size
    if writer is None:
        h, w = vis.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # Write to video
    writer.write(vis)

    # Show on screen
    cv2.imshow('Tracking / Mask', vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# Cleanup
cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

# Save trajectory
with open('trajectory.csv', 'w', newline='') as csvfile:
    writer_csv = csv.writer(csvfile)
    writer_csv.writerow(['frame', 'x', 'y'])
    writer_csv.writerows(positions)

print(f"Done! Saved {len(positions)} positions to trajectory.csv and video to {output_video}")
