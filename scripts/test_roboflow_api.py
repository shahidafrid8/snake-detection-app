# scripts/test_roboflow_api.py
# This script tests the Roboflow API connection and model inference
# Use this to verify your API key and model configuration

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from roboflow import Roboflow

def test_api():
    """
    Test Roboflow API connection and run a sample inference.
    """
    if not config.ROBOFLOW_API_KEY:
        print("Error: ROBOFLOW_API_KEY is not set in config.py or environment variables.")
        return
    
    print(f"Testing Roboflow API with key: {config.ROBOFLOW_API_KEY[:10]}...")
    print(f"Model ID: {config.ROBOFLOW_MODEL_ID}")
    
    try:
        rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
        
        # Parse model ID
        parts = config.ROBOFLOW_MODEL_ID.split('/')
        if len(parts) == 3:
            workspace, project_id, version = parts
            project = rf.workspace(workspace).project(project_id)
        elif len(parts) == 2:
            project_id, version = parts
            project = rf.workspace().project(project_id)
        else:
            print(f"Error: Invalid ROBOFLOW_MODEL_ID format: {config.ROBOFLOW_MODEL_ID}")
            return
        
        model = project.version(version).model
        
        # Test with a sample image
        test_image_path = "data/test.png"
        if not os.path.exists(test_image_path):
            print(f"Warning: Test image not found at {test_image_path}")
            print("Please provide a valid image path to test inference.")
            return
        
        print(f"Running inference on {test_image_path}...")
        result = model.predict(test_image_path, confidence=config.CONF_THRESHOLD)
        print("\n✅ API test successful!")
        print("\nPrediction results:")
        print(result.json())
        
    except Exception as e:
        print(f"\n❌ API test failed!")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()