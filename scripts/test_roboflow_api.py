import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from roboflow import Roboflow

rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
project = rf.workspace().project("snake-detection-gat5j-nbtyc")
model = project.version(1).model

# Run inference on test image
result = model.predict("data/test.png")
print(result.json())
project = rf.workspace().project("snake-detection-gat5j-nbtyc")
model = project.version(1).model

# Run inference on test image
result = model.predict("data/test.png")
print(result.json())