import streamlit as st
from PIL import Image, ImageDraw
import os
import sys
import tempfile
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from scripts.detect import detect
from scripts.detect_video import detect_video

# Set API key from Streamlit secrets
api_key_configured = False
if 'ROBOFLOW_API_KEY' in st.secrets:
    os.environ['ROBOFLOW_API_KEY'] = st.secrets['ROBOFLOW_API_KEY']
    api_key_configured = True
elif 'api_key' in st.secrets:  # Alternative key name
    os.environ['ROBOFLOW_API_KEY'] = st.secrets['api_key']
    api_key_configured = True

# Set model ID from Streamlit secrets if available
if 'ROBOFLOW_MODEL_ID' in st.secrets:
    os.environ['ROBOFLOW_MODEL_ID'] = st.secrets['ROBOFLOW_MODEL_ID']

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
    st.header("⚙️ Settings")
    
    # Theme toggle
    theme = st.toggle("🌙 Dark Mode", value=True)
    if theme:
        st.markdown("""
        <style>
        :root {
            --background-color: #0e1117;
            --text-color: #fafafa;
            --secondary-bg: #161b22;
            --border-color: #30363d;
        }
        
        html, body, .stApp {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        
        .stApp, [data-testid="stDecoration"] {
            background: #0e1117 !important;
        }
        
        * {
            color: #fafafa !important;
        }
        
        /* Text elements */
        h1, h2, h3, h4, h5, h6, p, span, label, div {
            color: #fafafa !important;
        }
        
        /* Input fields */
        input, textarea, select {
            background-color: #161b22 !important;
            color: #fafafa !important;
            border-color: #30363d !important;
        }
        
        /* Input text color specifically */
        input[type="text"],
        input[type="password"],
        input[type="email"],
        input[type="number"],
        input[type="file"],
        textarea {
            background-color: #161b22 !important;
            color: #fafafa !important;
            border-color: #30363d !important;
        }
        
        /* Input placeholder text */
        input::placeholder,
        textarea::placeholder {
            color: #8b949e !important;
        }
        
        /* Input focus state */
        input:focus,
        textarea:focus,
        select:focus {
            background-color: #0d1117 !important;
            color: #fafafa !important;
            border-color: #58a6ff !important;
            outline: none !important;
        }
        
        /* Autofilled input */
        input:-webkit-autofill,
        input:-webkit-autofill:hover,
        input:-webkit-autofill:focus {
            -webkit-box-shadow: 0 0 0 1000px #161b22 inset !important;
            -webkit-text-fill-color: #fafafa !important;
            border-color: #30363d !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #0d1117 !important;
        }
        
        /* Buttons and interactive elements */
        button {
            color: #fafafa !important;
        }
        
        /* Info boxes */
        .stInfo, .stSuccess, .stWarning, .stError {
            background-color: #161b22 !important;
            color: #fafafa !important;
        }
        
        /* Divider */
        hr {
            border-color: #30363d !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        :root {
            --background-color: #ffffff;
            --text-color: #262730;
            --secondary-bg: #f0f2f6;
            --border-color: #ddd;
        }
        
        html, body, .stApp {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        
        .stApp, [data-testid="stDecoration"] {
            background: #ffffff !important;
        }
        
        * {
            color: #262730 !important;
        }
        
        /* Text elements */
        h1, h2, h3, h4, h5, h6, p, span, label, div {
            color: #262730 !important;
        }
        
        /* Input fields */
        input, textarea, select {
            background-color: #f9f9f9 !important;
            color: #262730 !important;
            border-color: #ddd !important;
        }
        
        /* Input text color specifically */
        input[type="text"],
        input[type="password"],
        input[type="email"],
        input[type="number"],
        input[type="file"],
        textarea {
            background-color: #f9f9f9 !important;
            color: #262730 !important;
            border-color: #ddd !important;
        }
        
        /* Input placeholder text */
        input::placeholder,
        textarea::placeholder {
            color: #999999 !important;
        }
        
        /* Input focus state */
        input:focus,
        textarea:focus,
        select:focus {
            background-color: #ffffff !important;
            color: #262730 !important;
            border-color: #0066cc !important;
            outline: none !important;
        }
        
        /* Autofilled input */
        input:-webkit-autofill,
        input:-webkit-autofill:hover,
        input:-webkit-autofill:focus {
            -webkit-box-shadow: 0 0 0 1000px #f9f9f9 inset !important;
            -webkit-text-fill-color: #262730 !important;
            border-color: #ddd !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
        }
        
        /* Buttons and interactive elements */
        button {
            color: #262730 !important;
        }
        
        /* Info boxes */
        .stInfo, .stSuccess, .stWarning, .stError {
            background-color: #f9f9f9 !important;
            color: #262730 !important;
        }
        
        /* Divider */
        hr {
            border-color: #ddd !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.header("ℹ️ About")
    st.write("This app detects snakes in images and videos using AI.")
    
    st.subheader("How to use:")
    st.write("1. Select file type (Image/Video)")
    st.write("2. Choose detection method")
    st.write("3. Upload file")
    st.write("4. View results & download")
    
    st.divider()
    
    st.subheader("🔑 API Status")
    if api_key_configured:
        st.success("✅ Roboflow API Ready")
    else:
        st.warning("⚠️ API Not Configured")
        st.caption("Will use local model only")
    
    # Check local model
    local_model_path = os.path.join(os.path.dirname(__file__), "model", "best.pt")
    if os.path.exists(local_model_path):
        st.success("✅ Local Model Ready")
    else:
        st.warning("⚠️ Local Model Not Found")
    
    st.divider()
    st.caption("🚀 Powered by YOLOv8 & Roboflow")

# File type selection
col1, col2 = st.columns([1, 1])
with col1:
    file_type = st.selectbox("📁 Select File Type", ["Image", "Video"], index=0)

with col2:
    # Detection method selection
    detection_methods = []
    if api_key_configured:
        detection_methods.append("🌐 Roboflow API")
    if os.path.exists(os.path.join(os.path.dirname(__file__), "model", "best.pt")):
        detection_methods.append("💻 Local Model")
    
    if len(detection_methods) > 0:
        detection_methods.insert(0, "✨ Auto (Best Available)")
        detection_method = st.selectbox("🎯 Detection Method", detection_methods, index=0)
    else:
        st.error("❌ No detection method available!")
        st.stop()

st.divider()

if file_type == "Image":
    st.write("🖼️ Upload an image to detect snakes")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
else:
    st.write("🎥 Upload a video to detect snakes (processed frame by frame)")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Map detection method selection to parameter
    if "Auto" in detection_method:
        method_param = "auto"
    elif "Roboflow" in detection_method:
        method_param = "api"
    elif "Local" in detection_method:
        method_param = "local"
    else:
        method_param = "auto"
    
    if file_type == "Image":
        # ===== IMAGE PROCESSING =====
        # Load image with PIL
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width="stretch")
        
        # Save temporarily for detection using cross-platform temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            temp_path = tmp_file.name
            img.save(temp_path)
        
        # Run detection with selected method
        with st.spinner(f"Detecting snakes using {detection_method}..."):
            predictions, method, error_msg = detect(temp_path, method_param)
        
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
                st.error("❌ **Detection Failed**")
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
                plotted = predictions[0].plot()
                # Convert BGR to RGB for proper display
                plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                draw_img = Image.fromarray(plotted_rgb)
                st.image(draw_img, caption="Detected Image", width="stretch")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    else:  # file_type == "Video"
        # ===== VIDEO PROCESSING =====
        st.video(uploaded_file)
        
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            temp_video_path = tmp_file.name
            tmp_file.write(uploaded_file.read())
        
        # Process video
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress = int((current / total) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {current}/{total} frames ({progress}%)")
        
        with st.spinner(f"Detecting snakes in video using {detection_method}..."):
            output_path, method, error_msg, stats = detect_video(temp_video_path, update_progress, method_param)
        
        if error_msg:
            # Same error handling as images
            error_lower = error_msg.lower()
            if "api key" in error_lower or "oauthexception" in error_lower:
                st.error("❌ **API Authentication Failed**")
                st.warning("Please configure your Roboflow API key in Streamlit Secrets.")
            elif "local model not found" in error_lower:
                st.error("❌ **Both API and Local Model Failed**")
                st.warning("Configure your Roboflow API key for video detection.")
            else:
                st.error("❌ **Video Detection Failed**")
                st.code(error_msg, language=None)
            
            # Clean up
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        else:
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ Video detection completed using {method}!")
            
            # Display stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames", stats['total_frames'])
            with col2:
                st.metric("Detections", stats['detections'])
            with col3:
                st.metric("FPS", stats['fps'])
            
            # Display processed video
            st.subheader("📹 Processed Video")
            if os.path.exists(output_path):
                with open(output_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                
                # Download button
                st.download_button(
                    label="⬇️ Download Detected Video",
                    data=video_bytes,
                    file_name="snake_detected_video.mp4",
                    mime="video/mp4"
                )
                
                # Clean up output video after showing
                os.remove(output_path)
            
            # Clean up temp video
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)