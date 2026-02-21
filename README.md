# 🐍 Snake Detection Project

An intelligent snake detection system using YOLOv8 and Roboflow API with a user-friendly Streamlit web interface.

## 🌟 Features

- **Dual Detection Methods**: 
  - Primary: Roboflow Cloud API (no local GPU required)
  - Fallback: Local YOLOv8 model (requires OpenCV and model file)
- **Web Interface**: Easy-to-use Streamlit app for image upload and detection
- **Command-Line Tools**: Scripts for batch processing and API testing
- **Robust Error Handling**: Informative error messages and automatic fallback

## 📁 Project Structure

```
Snake_Detection_Project/
├── app.py                      # Streamlit web application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── model/
│   └── best.pt                 # Local YOLOv8 model (optional)
├── data/                       # Sample images and datasets
├── scripts/
│   ├── detect.py              # Main detection orchestrator
│   ├── detect_api.py          # Roboflow API detection
│   ├── detect_local.py        # Local YOLOv8 detection
│   ├── detect_image.py        # CLI tool for single image detection
│   └── test_roboflow_api.py   # API connection tester
└── .streamlit/
    └── secrets.toml           # API keys (not in git)
```

## 🚀 Setup

### 1. Create and Activate Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activate on Windows CMD
venv\Scripts\activate.bat
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API Keys

Create `.streamlit/secrets.toml` file:

```toml
ROBOFLOW_API_KEY = "your_roboflow_api_key_here"
# Optional: Override default model
# ROBOFLOW_MODEL_ID = "workspace/project/version"
```

Or set environment variable:

```powershell
$env:ROBOFLOW_API_KEY = "your_api_key_here"
```

## 📖 Usage

### Web Interface (Recommended)

```powershell
streamlit run app.py
```

Then open your browser to the displayed local URL (typically http://localhost:8501)

### Command Line Detection

```powershell
# Navigate to scripts directory
cd scripts

# Run detection on a single image
python detect_image.py
# Enter image path when prompted
```

### Test Roboflow API Connection

```powershell
cd scripts
python test_roboflow_api.py
```

## ⚙️ Configuration

Edit `config.py` to customize:

- `ROBOFLOW_API_KEY`: Your Roboflow API key
- `ROBOFLOW_MODEL_ID`: Model identifier (format: "workspace/project/version")
- `LOCAL_MODEL_PATH`: Path to local YOLOv8 model file
- `CONF_THRESHOLD`: Detection confidence threshold (0.0-1.0, default: 0.5)

## 🔧 Troubleshooting

### Streamlit Not Found
Ensure you activated the virtual environment and installed requirements.

### API Key Errors
- Verify your API key in `.streamlit/secrets.toml`
- Check that your Roboflow model is deployed and accessible
- Ensure you have remaining API credits

### Local Model Fallback
- Download/train a YOLOv8 model and place in `model/best.pt`
- Ensure OpenCV is installed: `pip install opencv-python`

## 📝 Notes

- The system automatically falls back to local model if API fails
- Supported image formats: PNG, JPG, JPEG
- Detection results are displayed with bounding boxes and confidence scores
- Temporary files are automatically cleaned up after processing

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is for educational and research purposes.