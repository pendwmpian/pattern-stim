import cv2
import numpy as np

def crop_image(input_path, output_path, crop_bounds):
    """
    Crops an image based on the provided bounds and saves it in PNG format.

    Args:
        input_path (str): Path to the input image (JPG).
        output_path (str): Path to save the cropped image (PNG).
        crop_bounds (tuple): A tuple (y_min, y_max, x_min, x_max) defining the
                             cropping region.
    """
    try:
        # Read the input image
        img = cv2.imread(input_path)

        if img is None:
            print(f"Error: Could not read the image from {input_path}. Please check the path and file format.")
            print("Make sure 'image.jpg' is in the same directory as your script, or provide the full path.")
            return

        # Unpack crop bounds
        y_min, y_max, x_min, x_max = crop_bounds

        # Perform the cropping
        # OpenCV slicing is [y_start:y_end, x_start:x_end]
        cropped_img = img[y_min:y_max, x_min:x_max]

        # Save the cropped image as PNG
        cv2.imwrite(output_path, cropped_img)
        print(f"Image successfully cropped and saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Define input and output paths
    # The input image path is now specifically './image.jpg'
    input_image_path = "../data/baseline_image_uncropped.jpg"
    output_image_path = "../data/baseline_image.png"

    # Define crop bounds (y_min, y_max, x_min, x_max)
    # This corresponds to:
    # rows from 510 to 589
    # columns from 300 to 1669
    crop_area = (510, 590, 300, 1670)

    # --- For testing purposes: Create a dummy 1080x1920 image if 'image.jpg' doesn't exist ---
    # You can comment out this section if you already have your 'image.jpg'
    try:
        # Check if the input file exists, if not, create a dummy one
        with open(input_image_path, 'rb') as f:
            pass # File exists, do nothing
    except FileNotFoundError:
        print(f"'{input_image_path}' not found. Creating a dummy 1080x1920 black image for testing.")
        dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(input_image_path, dummy_image)
        print(f"Dummy image created at {input_image_path}.")
    # --- End of dummy image creation section ---

    # Call the cropping function
    crop_image(input_image_path, output_image_path, crop_area)