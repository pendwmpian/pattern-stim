import cv2
import numpy as np
import time
import os

# --- Configuration for Static Background Subtraction Method ---
video_path = './data/video2.mp4'
baseline_image_path = './data/baseline_image.png' # Path to the pre-generated baseline
output_video_static_bg_sub = './data/tracking_output_static_bg_sub.mp4'

# Video cropping bounds
crop_bounds = (560, 640, 150, 1700) # y_min, y_max, x_min, x_max

# Background Subtraction Parameters
diff_threshold = 30
morph_kernel_size_bg_sub = (5, 5)
min_contour_area_bg_sub = 150 # Tune this carefully!

# ---------------------

class PositionEstimationStaticBgSub():
    def __init__(self, writer=None, crop_bounds_param=crop_bounds, baseline_path_param=baseline_image_path):
        self.writer = writer
        self.y_min, self.y_max, self.x_min, self.x_max = crop_bounds_param
        
        if not os.path.exists(baseline_path_param):
            print(f"ERROR: Baseline image not found at {baseline_path_param}")
            print("Please generate it first (e.g., using generate_baseline.py).")
            self.baseline_image_gray = None 
        else:
            self.baseline_image_gray = cv2.imread(baseline_path_param, cv2.IMREAD_GRAYSCALE)
            if self.baseline_image_gray is None:
                print(f"ERROR: Failed to load baseline image from {baseline_path_param}")
            else:
                print(f"Successfully loaded baseline from {baseline_path_param}")
        
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size_bg_sub)
        self.prev_pos = None
        self.vis = None

    # No set_baseline method needed for static version

    def new_frame(self, frame):
        if self.baseline_image_gray is None:
            print("Skipping frame processing: Baseline not available.")
            if self.writer:
                roi_h = self.y_max - self.y_min
                roi_w = self.x_max - self.x_min
                self.vis = np.zeros((roi_h * 3, roi_w, 3), dtype=np.uint8)
            return None 
            
        roi = frame[self.y_min:self.y_max, self.x_min:self.x_max]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
        
        diff_image = cv2.absdiff(gray_roi, self.baseline_image_gray)
        _, thresh_diff = cv2.threshold(diff_image, diff_threshold, 255, cv2.THRESH_BINARY)
        
        opened_mask = cv2.morphologyEx(thresh_diff, cv2.MORPH_OPEN, self.kernel, iterations=1)
        clean_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area_bg_sub:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            candidates.append(((cx, cy), area))

        chosen_pos = None
        if candidates:
            if self.prev_pos is None:
                chosen_pos = max(candidates, key=lambda x: x[1])[0]
            else:
                chosen_pos = min(candidates, key=lambda x: np.hypot(x[0][0] - self.prev_pos[0], x[0][1] - self.prev_pos[1]))[0]
        
        if chosen_pos:
            self.prev_pos = chosen_pos

        # Visualization
        roi_bgr = roi.copy()
        if chosen_pos:
            cv2.circle(roi_bgr, chosen_pos, 6, (0, 0, 255), -1)
        diff_image_bgr = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR)
        clean_mask_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
        self.vis = np.vstack((roi_bgr, diff_image_bgr, clean_mask_bgr))
        
        if self.writer is not None:
            self.writer.write(self.vis)
        return chosen_pos

# No helper function for dynamic baseline generation needed for static version

if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: FPS is 0. Setting to default 30.")
        fps = 30 
        
    roi_w = crop_bounds[3] - crop_bounds[2]
    roi_h = crop_bounds[1] - crop_bounds[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_static_bg_sub, fourcc, fps, (roi_w, roi_h * 3))

    estimator = PositionEstimationStaticBgSub(writer=writer, crop_bounds_param=crop_bounds, baseline_path_param=baseline_image_path)

    if estimator.baseline_image_gray is None:
        print("FATAL: Could not load baseline image. Exiting.")
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        exit()
    
    frames_processed_in_main_loop = 0
    
    prev = 0
    while True:
        t = time.time()
        print(t - prev)
        prev = t
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        pos = estimator.new_frame(frame)
        
        if estimator.vis is not None:
            cv2.imshow('Static Background Subtraction Tracking', estimator.vis)
        elif frame is not None:
            y_min_fb, y_max_fb, x_min_fb, x_max_fb = crop_bounds
            if y_max_fb <= frame.shape[0] and x_max_fb <= frame.shape[1]:
                 cv2.imshow('Static Background Subtraction Tracking - ROI', frame[y_min_fb:y_max_fb, x_min_fb:x_max_fb])
            else:
                 cv2.imshow('Static Background Subtraction Tracking - Raw Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frames_processed_in_main_loop += 1

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
