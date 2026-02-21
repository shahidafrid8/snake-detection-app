# scripts/detect.py
# This is the main entry point for the object detection system.
# It orchestrates the detection process by first attempting to use the
# Roboflow API and falling back to the local model if the API call fails.

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from detect_api import detect_with_api
from detect_local import detect_with_local_model

def detect(image_path):
    """
    Performs object detection on an image, first trying the Roboflow API
    and falling back to a local model on any failure.

    Args:
        image_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing the predictions, the method used ('API' or 'LOCAL'), and error message (None if success).
    """
    if not os.path.exists(image_path):
        return None, "FAILED", f"Image not found at {image_path}"

    try:
        print("Attempting detection with Roboflow API...")
        predictions = detect_with_api(image_path)
        print("Inference successful with Roboflow API.")
        return predictions, "API", None
    except Exception as e:
        print(f"Roboflow API failed: {e}")
        print("Falling back to local YOLOv8 model...")
        try:
            predictions = detect_with_local_model(image_path)
            print("Inference successful with local model.")
            return predictions, "LOCAL", None
        except Exception as e:
            print(f"Local model detection also failed: {e}")
            return None, "FAILED", str(e)

if __name__ == '__main__':
    # Example usage:
    # This block will run if the script is executed directly.
    # You can change the image path to test the detection system.
    test_image = "data/test.png"
    
    predictions, method, error_msg = detect(test_image)

    if error_msg:
        print("\nDetection Failed!")
        print(f"Error: {error_msg}")
    else:
        print(f"\nDetection Method Used: {method}")
        print("Predictions:")
        # The structure of 'predictions' will differ between the API and the local model.
        print(predictions)

