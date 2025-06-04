import cv2
import numpy as np
import time
import os

# --- Configuration for Adaptive Background Subtraction Method ---
video_path = './data/video2.mp4'
initial_baseline_image_path = './data/baseline_image.png'
output_video_bg_sub = './data/tracking_output_adaptive_bg_sub.mp4'

# Video cropping bounds
crop_bounds = (560, 640, 150, 1700) # y_min, y_max, x_min, x_max

# Adaptive Baseline Parameters
baseline_update_interval_seconds = 60
frames_for_new_baseline = 100
frame_sample_interval_for_baseline = 5

# Background Subtraction Parameters
diff_threshold = 30
morph_kernel_size_bg_sub = (5, 5)
min_contour_area_bg_sub = 150 # Tune this carefully!

# ---------------------

class PositionEstimationBgSub():
    def __init__(self, writer=None, crop_bounds_param=crop_bounds, initial_baseline_path_param=initial_baseline_image_path):
        self.writer = writer
        self.y_min, self.y_max, self.x_min, self.x_max = crop_bounds_param
        
        if not os.path.exists(initial_baseline_path_param):
            print(f"ERROR: Initial baseline image not found at {initial_baseline_path_param}")
            self.baseline_image_gray = None 
        else:
            self.baseline_image_gray = cv2.imread(initial_baseline_path_param, cv2.IMREAD_GRAYSCALE)
            if self.baseline_image_gray is None:
                print(f"ERROR: Failed to load initial baseline image from {initial_baseline_path_param}")
            else:
                print(f"Successfully loaded initial baseline from {initial_baseline_path_param}")
        
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size_bg_sub)
        self.prev_pos = None
        self.vis = None

    def set_baseline(self, new_baseline_image_gray):
        print("Updating baseline image...")
        self.baseline_image_gray = new_baseline_image_gray

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

# --- Helper function to generate/update baseline ---
# Defined at the top level so it's available to the main block.
y_min_crop_g, y_max_crop_g, x_min_crop_g, x_max_crop_g = crop_bounds # Use global crop_bounds

def generate_current_baseline(cap_obj, num_frames_to_sample, sample_interval, current_frame_num_log="N/A"):
    print(f"Generating baseline around frame {current_frame_num_log} using {num_frames_to_sample} samples, interval {sample_interval}...")
    
    frames_for_avg = []
    # Read first frame for dimensions
    ret_init, frame_init = cap_obj.read()
    if not ret_init:
        print("Error: Could not read frame for baseline dimension.")
        return None
    
    cropped_init = frame_init[y_min_crop_g:y_max_crop_g, x_min_crop_g:x_max_crop_g]
    if cropped_init.size == 0:
        print("Error: Crop bounds empty for baseline.")
        return None
    gray_init = cv2.cvtColor(cropped_init, cv2.COLOR_BGR2GRAY)
    accumulator = np.zeros_like(gray_init, dtype=np.float64)
    
    # Add first sampled frame
    accumulator += gray_init.astype(np.float64)
    frames_collected_count = 1
    
    for _ in range(1, num_frames_to_sample):
        for _ in range(sample_interval): # Skip frames
            ret_skip, _ = cap_obj.read()
            if not ret_skip: break 
        if not ret_skip: break # Break outer loop too

        ret_sample, frame_sample = cap_obj.read()
        if not ret_sample:
            print(f"Warning: Video ended during baseline sampling. Using {frames_collected_count} frames.")
            break
        
        cropped_sample = frame_sample[y_min_crop_g:y_max_crop_g, x_min_crop_g:x_max_crop_g]
        gray_sample = cv2.cvtColor(cropped_sample, cv2.COLOR_BGR2GRAY)
        accumulator += gray_sample.astype(np.float64)
        frames_collected_count += 1

    if frames_collected_count == 0:
        return None
    return (accumulator / frames_collected_count).astype(np.uint8)

if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Handle case where FPS might not be read correctly
        print("Warning: FPS is 0. Setting to default 30 for baseline update interval calculation.")
        fps = 30 
        
    roi_w = crop_bounds[3] - crop_bounds[2]
    roi_h = crop_bounds[1] - crop_bounds[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_bg_sub, fourcc, fps, (roi_w, roi_h * 3))

    estimator = PositionEstimationBgSub(writer=writer, crop_bounds_param=crop_bounds, initial_baseline_path_param=initial_baseline_image_path)

    if estimator.baseline_image_gray is None:
        print("FATAL: Could not load initial baseline. Exiting.")
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        exit()
    
    frames_processed_in_main_loop = 0
    baseline_update_trigger_frame_count = int(fps * baseline_update_interval_seconds) if baseline_update_interval_seconds > 0 else 0
    
    print(f"FPS: {fps}, Baseline update interval: {baseline_update_interval_seconds}s, Triggering update every ~{baseline_update_trigger_frame_count} main loop frames.")

    while True:
        if baseline_update_trigger_frame_count > 0 and \
           frames_processed_in_main_loop > 0 and \
           (frames_processed_in_main_loop % baseline_update_trigger_frame_count == 0):
            
            print(f"\n--- Triggering baseline update at main loop frame: {frames_processed_in_main_loop} ---")
            main_loop_current_pos_frames = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            new_baseline = generate_current_baseline(cap, 
                                                     frames_for_new_baseline, 
                                                     frame_sample_interval_for_baseline,
                                                     current_frame_num_log=f"approx {main_loop_current_pos_frames}")
            if new_baseline is not None:
                estimator.set_baseline(new_baseline)
            else:
                print("Warning: Failed to generate new dynamic baseline. Continuing with the old one.")
            
            print(f"Restoring video capture to frame: {main_loop_current_pos_frames} for main loop.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, main_loop_current_pos_frames)
            print("--- Baseline update process complete ---\n")

        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        pos = estimator.new_frame(frame)
        
        if estimator.vis is not None:
            cv2.imshow('Adaptive Background Subtraction Tracking', estimator.vis)
        elif frame is not None:
            y_min_fb, y_max_fb, x_min_fb, x_max_fb = crop_bounds
            if y_max_fb <= frame.shape[0] and x_max_fb <= frame.shape[1]:
                 cv2.imshow('Adaptive Background Subtraction Tracking - ROI', frame[y_min_fb:y_max_fb, x_min_fb:x_max_fb])
            else:
                 cv2.imshow('Adaptive Background Subtraction Tracking - Raw Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frames_processed_in_main_loop += 1

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
