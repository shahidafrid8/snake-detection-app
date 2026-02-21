# config.py
# This file contains the configuration settings for the object detection system.
# It includes API keys, model paths, and confidence thresholds.

# Roboflow API Configuration
ROBOFLOW_API_KEY = "69iblLFisbddbkiVO0wy"  # Replace with your actual Roboflow API key
ROBOFLOW_MODEL_ID = "snake-detection-gat5j-nbtyc/1"  # Replace with your Roboflow model ID

# Local Model Configuration
LOCAL_MODEL_PATH = "model/best.pt"

# Detection Confidence Threshold
CONF_THRESHOLD = 0.5
