# Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub account
- Streamlit Cloud account (free at https://streamlit.io/cloud)
- Your repository pushed to GitHub

## Step-by-Step Deployment

### 1. Prepare Your Repository
Ensure your code is pushed to GitHub:
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository
4. Choose the branch (usually `main`)
5. Set main file path to: `app.py`
6. Click "Deploy"

### 3. Configure Secrets
After deployment:
1. Go to your app's settings (gear icon → Manage app)
2. Click "Secrets"
3. Add the following secrets (copy from your `.env` file):

```toml
GEMINI_API_KEY = "your-actual-api-key"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"
LLM_PROVIDER = "gemini"
```

### 4. Access Your App
Your app will be available at:
`https://[your-username]-[repo-name].streamlit.app`

## Important Notes

### Data & Indexes
- **Local Indexes**: The `indexes/` directory should be committed to Git for deployment
- **Policy PDFs**: Users can upload via the Streamlit interface or place them in `data/policies/`
- **Index Rebuilding**: The "Rebuild index" button will work on the deployed app

### Requirements
- All dependencies in `requirements.txt` are properly specified
- The app uses Streamlit's caching (`@st.cache_resource`) for efficient resource loading

### Troubleshooting
- Check app logs in Streamlit Cloud dashboard if deployment fails
- Ensure all required API keys are set in Secrets
- Verify `requirements.txt` has no conflicting versions
- Test locally with `streamlit run app.py` before pushing

## Local Testing
Before deploying, test locally:
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

## File Structure for Deployment
```
your-repo/
├── .streamlit/
│   ├── config.toml          (Streamlit configuration)
│   └── secrets.toml         (local secrets, in .gitignore)
├── .env                     (environment variables, in .gitignore)
├── .gitignore
├── app.py                   (main entry point)
├── config.py
├── ingest.py
├── requirements.txt
├── README.md
├── src/                     (source code)
├── data/                    (policies directory)
├── indexes/                 (commit this to Git)
└── tests/                   (test suite)
```

## Security Best Practices
✅ DO:
- Store sensitive keys in Streamlit Secrets, never commit `.env`
- Use environment variables via `st.secrets`
- Keep `.streamlit/secrets.toml` in `.gitignore`

❌ DON'T:
- Commit `.env` or API keys to Git
- Hardcode secrets in Python files
- Share API keys in public repositories
