import cv2
import numpy as np
import time
import os # For checking if baseline image exists

# --- Configuration for Background Subtraction Method ---
video_path = './data/video2.mp4'
baseline_image_path = './data/baseline_image.png'
output_video_bg_sub = './data/tracking_output_bg_sub.mp4'

# Video cropping bounds (should be same as used for baseline generation)
crop_bounds = (560, 640, 150, 1700) # y_min, y_max, x_min, x_max

# Background Subtraction Parameters - TUNE THESE
diff_threshold = 30  # Threshold for the absolute difference image
morph_kernel_size_bg_sub = (5, 5) # Kernel for morphological operations
min_contour_area_bg_sub = 100    # Minimum contour area in pixels for the animal

# ---------------------

class PositionEstimationBgSub():

    def __init__(self, writer=None, crop_bounds_param=crop_bounds, baseline_path_param=baseline_image_path):
        self.writer = writer
        self.y_min, self.y_max, self.x_min, self.x_max = crop_bounds_param
        
        if not os.path.exists(baseline_path_param):
            print(f"ERROR: Baseline image not found at {baseline_path_param}")
            print("Please run generate_baseline.py first.")
            self.baseline_image_gray = None # Indicate error
            # Or raise an exception: raise FileNotFoundError(f"Baseline image not found: {baseline_path_param}")
        else:
            self.baseline_image_gray = cv2.imread(baseline_path_param, cv2.IMREAD_GRAYSCALE)
            if self.baseline_image_gray is None:
                print(f"ERROR: Failed to load baseline image from {baseline_path_param}")
                # Or raise an exception
        
        # Morphology kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size_bg_sub)

        self.prev_pos = None
        self.vis = None # For storing the visualization frame

    def new_frame(self, frame):
        if self.baseline_image_gray is None:
            # If baseline failed to load, cannot proceed
            # Optionally return a black screen or skip processing
            # For now, just make vis black if writer exists
            if self.writer:
                roi_h = self.y_max - self.y_min
                roi_w = self.x_max - self.x_min
                self.vis = np.zeros((roi_h * 3, roi_w, 3), dtype=np.uint8) # Placeholder for 3 panels
                # self.writer.write(self.vis) # Avoid writing if error
            return None 
        # Crop region of interest
        roi = frame[self.y_min:self.y_max, self.x_min:self.x_max]

        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
        
        # 1. Calculate absolute difference
        diff_image = cv2.absdiff(gray_roi, self.baseline_image_gray)
        
        # 2. Threshold the difference image
        _, thresh_diff = cv2.threshold(diff_image, diff_threshold, 255, cv2.THRESH_BINARY)
        
        # 3. Morphological cleaning
        # Opening followed by Closing can help remove noise and fill gaps
        opened_mask = cv2.morphologyEx(thresh_diff, cv2.MORPH_OPEN, self.kernel, iterations=1)
        clean_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2) # More closing iterations

        # 4. Contour detection (similar to original camera_detection.py)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area_bg_sub: # Use new min area
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            candidates.append(((cx, cy), area))

        # 5. Select the most plausible blob
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
        
        if chosen_pos:
            self.prev_pos = chosen_pos

        # Visualization
        if self.writer is not None:
            roi_bgr = roi.copy() # Original ROI for drawing
            if chosen_pos:
                cv2.circle(roi_bgr, chosen_pos, 6, (0, 0, 255), -1) # Draw on original ROI

            # Convert intermediate images to BGR for stacking
            diff_image_bgr = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR)
            clean_mask_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
            
            self.vis = np.vstack((roi_bgr, diff_image_bgr, clean_mask_bgr))
            self.writer.write(self.vis)

        return chosen_pos


if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    roi_w = crop_bounds[3] - crop_bounds[2]
    roi_h = crop_bounds[1] - crop_bounds[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Use the new output video filename
    writer = cv2.VideoWriter(output_video_bg_sub, fourcc, fps, (roi_w, roi_h * 3))

    # Instantiate the renamed class
    estimator = PositionEstimationBgSub(writer=writer, crop_bounds_param=crop_bounds)
    
    if estimator.baseline_image_gray is None:
        print("Exiting due to baseline image loading failure.")
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        exit()


    frame_idx = 0
    prev_time = time.time()
    while True:
        # current_time = time.time()
        # print(f"Loop time: {current_time - prev_time}")
        # prev_time = current_time

        ret, frame = cap.read()
        if not ret:
            break

        pos = estimator.new_frame(frame)
        # if pos:
        #     print(f"Frame {frame_idx}: Detected at {pos}")

        if estimator.vis is not None:
            cv2.imshow('StdDev Chunk Tracking', estimator.vis)
        else: # Fallback if vis not ready
            cv2.imshow('StdDev Chunk Tracking - ROI', frame[crop_bounds[0]:crop_bounds[1], crop_bounds[2]:crop_bounds[3]])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
