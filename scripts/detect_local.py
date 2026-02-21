# scripts/detect_local.py
# This script defines the function for running object detection using a local YOLOv8 model.
# It loads the model from the path specified in the config file and performs inference
# on a given image.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO
from config import LOCAL_MODEL_PATH, CONF_THRESHOLD

def detect_with_local_model(image_path):
    """
    Performs object detection on a local image using a local YOLOv8 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        list: The prediction results from the local model.
    """
    try:
        model = YOLO(LOCAL_MODEL_PATH)
        results = model(image_path, conf=CONF_THRESHOLD)
        return results
    except Exception as e:
        # Raise an exception if model loading or inference fails
        raise Exception(f"An unexpected error occurred during local detection: {e}")
