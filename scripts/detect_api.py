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
    if not ROBOFLOW_API_KEY:
        raise Exception("ROBOFLOW_API_KEY is not set. Please configure it in environment variables or Streamlit secrets.")
    
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        
        # ROBOFLOW_MODEL_ID is expected to be in "workspace/project/version" format
        parts = ROBOFLOW_MODEL_ID.split('/')
        if len(parts) == 3:
            workspace, project_id, version = parts
            project = rf.workspace(workspace).project(project_id)
        elif len(parts) == 2:
            # Fallback to format "project/version"
            project_id, version = parts
            project = rf.workspace().project(project_id)
        else:
            raise Exception(f"Invalid ROBOFLOW_MODEL_ID format: {ROBOFLOW_MODEL_ID}. Expected 'workspace/project/version' or 'project/version'")
        
        model = project.version(version).model

        result = model.predict(image_path, confidence=CONF_THRESHOLD).json()
        return result
    except Exception as e:
        # Provide more informative error messages
        error_msg = str(e)
        if "OAuthException" in error_msg or "does not exist" in error_msg:
            raise Exception(f"API authentication failed. Check your API key and model ID: {error_msg}")
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            raise Exception(f"Model not found. Verify ROBOFLOW_MODEL_ID '{ROBOFLOW_MODEL_ID}' is correct: {error_msg}")
        else:
            raise Exception(f"API detection error: {error_msg}")
