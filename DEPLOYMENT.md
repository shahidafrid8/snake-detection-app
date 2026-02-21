# 🚀 Deployment Guide for Streamlit Community Cloud

## Quick Deploy Steps

### 1. Fork or Clone this Repository
Make sure you have this code in your GitHub account.

### 2. Go to Streamlit Community Cloud
Visit: [share.streamlit.io](https://share.streamlit.io)

### 3. Create New App
- Click "New app"
- Select your repository: `shahidafrid8/snake-detection-app`
- Branch: `main`
- Main file: `app.py`

### 4. **IMPORTANT: Configure Secrets**

Before deploying, click **"Advanced settings"** and add your Roboflow API key in the **Secrets** section:

```toml
ROBOFLOW_API_KEY = "your_actual_roboflow_api_key_here"
```

#### How to get your Roboflow API Key:
1. Go to [Roboflow](https://app.roboflow.com)
2. Sign in to your account
3. Navigate to Settings → [API Keys](https://app.roboflow.com/settings/api)
4. Copy your API key
5. Paste it in Streamlit Secrets

### 5. Deploy
Click **"Deploy"** and wait 2-3 minutes for the app to build.

## Troubleshooting

### "Model file not found" Error
- This means the API key is not configured
- Go to your app settings → Secrets
- Add the ROBOFLOW_API_KEY as shown above

### "API Authentication Failed"
- Check that your API key is correct
- Ensure there are no extra spaces in the secrets.toml
- Verify your Roboflow account is active

### App Won't Start
- Check the logs in Streamlit Cloud
- Ensure requirements.txt is present
- Verify all dependencies are compatible

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| ROBOFLOW_API_KEY | Yes | Your Roboflow API key |
| ROBOFLOW_MODEL_ID | No | Override default model (format: workspace/project/version) |

## Local Testing Before Deployment

1. Create `.streamlit/secrets.toml`:
```toml
ROBOFLOW_API_KEY = "your_api_key"
```

2. Run locally:
```bash
streamlit run app.py
```

3. Test with sample images to ensure API works

## Support

- [Streamlit Documentation](https://docs.streamlit.io)
- [Roboflow Documentation](https://docs.roboflow.com)
- [GitHub Issues](https://github.com/shahidafrid8/snake-detection-app/issues)
