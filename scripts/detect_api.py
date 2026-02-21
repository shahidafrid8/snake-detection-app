# scripts/detect_api.py
# This script defines the function for running object detection using the Roboflow API.
# It takes an image path and returns the prediction results.
# It raises an exception if the API call fails for any reason.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from roboflow import Roboflow
from config import ROBOFLOW_API_KEY, ROBOFLOW_MODEL_ID, CONF_THRESHOLD

def detect_with_api(image_path):
    """
    Performs object detection on a local image using the Roboflow API.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: The prediction results from the Roboflow API.
    
    Raises:
        Exception: For any errors during API inference.
    """
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        
        # ROBOFLOW_MODEL_ID is expected to be in "project_id/version_number" format
        project_id, version_id = ROBOFLOW_MODEL_ID.split('/')
        
        project = rf.workspace().project(project_id)
        model = project.version(version_id).model

        result = model.predict(image_path, confidence=CONF_THRESHOLD).json()
        return result
    except Exception as e:
        # Catch any exceptions from the roboflow library and raise a generic exception
        raise Exception(f"An unexpected error occurred during API detection: {e}")
