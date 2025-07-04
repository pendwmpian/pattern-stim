import cv2
import numpy as np
import time

# --- Configuration ---
video_path = './data/video0627.mp4'
baseline_output_path = './data/baseline_image_0627.png'
num_frames_to_average = 2000 # Number of frames to average for the baseline
crop_bounds = (560, 680, 150, 1700) # y_min, y_max, x_min, x_max (same as in tracking)
# ---------------------

def create_baseline_image():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    y_min, y_max, x_min, x_max = crop_bounds
    
    # Read the first frame to get dimensions for the accumulator
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    first_frame_cropped = first_frame[y_min:y_max, x_min:x_max]
    if first_frame_cropped.size == 0:
        print(f"Error: Crop bounds resulted in an empty image. Check crop_bounds: {crop_bounds}")
        cap.release()
        return
        
    first_frame_gray = cv2.cvtColor(first_frame_cropped, cv2.COLOR_BGR2GRAY)
    
    # Initialize accumulator with float type for precision during averaging
    accumulator = np.zeros_like(first_frame_gray, dtype=np.float64)
    
    frames_processed = 0
    
    # Add the first frame to the accumulator
    accumulator += first_frame_gray.astype(np.float64)
    frames_processed += 1
    
    print(f"Starting baseline generation using {num_frames_to_average} frames...")
    start_time = time.time()

    for i in range(1, num_frames_to_average): # Already processed one frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Video ended before {num_frames_to_average} frames were processed. Averaging {frames_processed} frames.")
            break
        
        cropped_frame = frame[y_min:y_max, x_min:x_max]
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        accumulator += gray_frame.astype(np.float64)
        frames_processed += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {frames_processed}/{num_frames_to_average} frames...")

    if frames_processed == 0:
        print("Error: No frames were processed.")
        cap.release()
        return

    # Calculate the average
    baseline_image_float = accumulator / frames_processed
    # Convert back to uint8 for saving and typical image operations
    baseline_image_uint8 = baseline_image_float.astype(np.uint8)

    cv2.imwrite(baseline_output_path, baseline_image_uint8)
    end_time = time.time()
    print(f"Baseline image generated from {frames_processed} frames and saved to {baseline_output_path}")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")

    cap.release()
    cv2.destroyAllWindows() # Just in case any imshow was accidentally left

if __name__ == '__main__':
    create_baseline_image()
