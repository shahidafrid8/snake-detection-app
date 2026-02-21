import streamlit as st
from PIL import Image, ImageDraw
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from scripts.detect import detect

# Set API key from Streamlit secrets
api_key_configured = False
if 'ROBOFLOW_API_KEY' in st.secrets:
    os.environ['ROBOFLOW_API_KEY'] = st.secrets['ROBOFLOW_API_KEY']
    api_key_configured = True
elif 'api_key' in st.secrets:  # Alternative key name
    os.environ['ROBOFLOW_API_KEY'] = st.secrets['api_key']
    api_key_configured = True

if not api_key_configured:
    st.error("⚠️ **Roboflow API Key Not Configured**")
    st.info("""
    To use this app, you need to configure your Roboflow API key:
    
    **For Streamlit Cloud:**
    1. Go to your app settings
    2. Click on "Secrets" in the left sidebar
    3. Add the following:
    ```toml
    ROBOFLOW_API_KEY = "your_api_key_here"
    ```
    4. Get your API key from: https://app.roboflow.com/settings/api
    
    **For Local Development:**
    - Create `.streamlit/secrets.toml` file with the same content
    """)

st.title("🐍 Snake Detection App")

# Sidebar with instructions
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app detects snakes in images using AI.")
    
    st.subheader("How to use:")
    st.write("1. Upload an image")
    st.write("2. Wait for detection")
    st.write("3. View results with bounding boxes")
    
    if api_key_configured:
        st.success("✅ API Key Configured")
    else:
        st.error("❌ API Key Missing")
        st.caption("Add ROBOFLOW_API_KEY in Secrets")
    
    st.divider()
    st.caption("Powered by YOLOv8 & Roboflow")

st.write("Upload an image to detect snakes using Roboflow API or local YOLO model.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image with PIL
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width="stretch")
    
    # Save temporarily for detection using cross-platform temp directory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        temp_path = tmp_file.name
        img.save(temp_path)
    
    # Run detection
    with st.spinner("Detecting snakes..."):
        predictions, method, error_msg = detect(temp_path)
    
    if error_msg:
        # Provide user-friendly error messages
        error_lower = error_msg.lower()
        if "api key" in error_lower or "oauthexception" in error_lower or "does not exist" in error_msg:
            st.error("❌ **API Authentication Failed**")
            st.warning("""
            **Invalid or missing Roboflow API key.** 
            
            Please ensure you have:
            1. Added your API key in Streamlit Cloud Secrets (see instructions above)
            2. Verified the API key is correct from https://app.roboflow.com/settings/api
            3. The key has access to the snake detection model
            """)
        elif "model" in error_lower and ("not found" in error_lower or "not available" in error_lower):
            st.error("❌ **Model Not Found**")
            st.info("""
            The specified Roboflow model is not accessible. Please check:
            - Model ID in config.py is correct
            - Model is deployed on Roboflow
            - Your API key has permissions to access this model
            """)
        elif "credits" in error_lower or "quota" in error_lower:
            st.error("❌ **API Credits Exhausted**")
            st.info("Your Roboflow account has reached its usage limit. Check your account at https://app.roboflow.com")
        elif "network" in error_lower or "connection" in error_lower or "timeout" in error_lower:
            st.error("❌ **Network Error**")
            st.warning("Unable to connect to Roboflow API. Please check your internet connection and try again.")
        elif "local model detection error" in error_lower or "local model not found" in error_lower:
            st.error("❌ **Both API and Local Model Failed**")
            st.warning("""
            **API Error:** Could not connect to Roboflow API
            
            **Local Model:** Model file not available on this deployment
            
            **Solution:** Configure your Roboflow API key in Streamlit Secrets (see instructions at the top)
            """)
        else:
            st.error(f"❌ **Detection Failed**")
            st.code(error_msg, language=None)
            st.info("If you're deploying on Streamlit Cloud, make sure to configure your ROBOFLOW_API_KEY in the app secrets.")
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        st.success(f"✅ Detection completed using {method}.")
        
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
                st.image(draw_img, caption="Detected Image", width="stretch")
            else:
                st.info("No snake detected in the image.")
                st.image(img, caption="Uploaded Image (No Detection)", width="stretch")
        elif method == "LOCAL":
            # For local, plot returns numpy array, convert to PIL
            import cv2
            plotted = predictions[0].plot()
            # Convert BGR to RGB for proper display
            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            draw_img = Image.fromarray(plotted_rgb)
            st.image(draw_img, caption="Detected Image", width="stretch")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)