import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from scripts.detect import detect

# Set API key from Streamlit secrets
if 'ROBOFLOW_API_KEY' in st.secrets:
    os.environ['ROBOFLOW_API_KEY'] = st.secrets['ROBOFLOW_API_KEY']
elif 'api_key' in st.secrets:  # Alternative key name
    os.environ['ROBOFLOW_API_KEY'] = st.secrets['api_key']
else:
    st.warning("Roboflow API key not set in secrets. Detection may fail. Set ROBOFLOW_API_KEY in app secrets.")

st.title("Snake Detection App")

st.write("Upload an image to detect snakes using Roboflow API or local YOLO model.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image with PIL
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Save temporarily for detection
    temp_path = "temp_image.png"
    img.save(temp_path)
    
    # Run detection
    with st.spinner("Detecting snakes..."):
        predictions, method, error_msg = detect(temp_path)
    
    if error_msg:
        # Provide user-friendly error messages
        if "API key" in error_msg or "OAuthException" in error_msg:
            st.error("Detection failed: Invalid or missing Roboflow API key. Check your API key in Streamlit secrets and ensure it has access to the model.")
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            st.error("Detection failed: Local model file not found. The model may not be available on this platform.")
        elif "cv2" in error_msg or "opencv" in error_msg.lower():
            st.error("Detection failed: OpenCV library issue. Local detection is not supported on this platform.")
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            st.error("Detection failed: Network issue. Check your internet connection for API calls.")
        elif "credits" in error_msg.lower():
            st.error("Detection failed: Roboflow API credits exhausted. Check your Roboflow account.")
        else:
            st.error(f"Detection failed: {error_msg}")
        return
        st.success(f"Detection completed using {method}.")
        
        # Create a copy for drawing
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        
        if method == "API":
            # Draw boxes from API predictions
            if 'predictions' in predictions and predictions['predictions']:
                for pred in predictions['predictions']:
                    x_center = pred['x']
                    y_center = pred['y']
                    width = pred['width']
                    height = pred['height']
                    confidence = pred['confidence']
                    class_name = pred['class']
                    
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                    label = f"{class_name} {confidence:.2f}"
                    draw.text((x1, y1 - 10), label, fill="green")
                st.image(draw_img, caption="Detected Image", use_column_width=True)
            else:
                st.info("No snake detected in the image.")
                st.image(img, caption="Uploaded Image (No Detection)", use_column_width=True)
        elif method == "LOCAL":
            # For local, plot returns numpy array, convert to PIL
            plotted = predictions[0].plot()
            draw_img = Image.fromarray(plotted)
            st.image(draw_img, caption="Detected Image", use_column_width=True)
        
        # Clean up
        os.remove(temp_path)
    else:
        st.error("Detection failed.")