import cv2
import numpy as np
import csv
import time

# --- Configuration ---
video_path = './data/video1.mp4'
output_video = './data/tracking_output1.mp4'

# Video cropping bounds
crop_bounds = (520, 600, 500, 1300)

# CLAHE parameters
clahe_clip = 2.0
clahe_tiles = (8, 8)

# Adaptive threshold parameters
block_size = 41  # must be odd
c_thresh = 2

# Morphology and contour filtering
morph_kernel_size = (3, 3)
min_contour_area = 60  # adjust to your animal size

# ---------------------

class positionEstimation():

    def __init__(self, writer=None, crop_bounds=crop_bounds):

        # Video writer 
        self.writer = writer
        # Crop bounds
        self.y_min, self.y_max, self.x_min, self.x_max = crop_bounds
        # CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tiles)
        # Morphology
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)

        self.prev_pos = None
        self.vis = None

    def new_frame(self, frame):

        # Crop region of interest
        roi = frame[self.y_min:self.y_max, self.x_min:self.x_max]

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

        # Clip bright values; animal is very dark
        _, gray = cv2.threshold(gray, 40, 255, cv2.THRESH_TRUNC)

        # 1) Local contrast enhancement
        equalized = self.clahe.apply(gray)

        # 2) Adaptive threshold (invert: animal dark -> white)
        th = cv2.adaptiveThreshold(equalized, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,
                                block_size, c_thresh)

        # 3) Morphological cleaning
        clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, self.kernel, iterations=2)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, self.kernel, iterations=2)

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
            if self.prev_pos is None:
                # first frame: largest
                chosen_pos = max(candidates, key=lambda x: x[1])[0]
            else:
                # closest to previous
                chosen_pos = min(candidates,
                                key=lambda x: np.hypot(x[0][0] - self.prev_pos[0],
                                                        x[0][1] - self.prev_pos[1]))[0]

        # 6) Record & visualize
        if chosen_pos is not None:
            self.prev_pos = chosen_pos

        if self.writer is not None:

            # 6) Record & visualize
            if chosen_pos is not None:
                cv2.circle(roi, chosen_pos, 6, (0, 0, 255), -1)

            # Prepare side-by-side display
            mask_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
            thresh_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            vis = np.vstack((roi, thresh_bgr, mask_bgr))   

            # Write to video
            self.writer.write(vis)

            self.vis = vis

        return chosen_pos


if __name__ == '__main__':

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    # For saving output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (crop_bounds[3] - crop_bounds[2], (crop_bounds[1] - crop_bounds[0]) * 3))

    frame_idx = 0

    posEs = positionEstimation(writer, crop_bounds)
    prev = 0
    while True:
        t = time.time()
        print(t - prev)
        prev = t

        ret, frame = cap.read()
        if not ret:
            break

        pos = posEs.new_frame(frame)

        # print(pos)

        # Show on screen
        cv2.imshow('Tracking / Mask', posEs.vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
