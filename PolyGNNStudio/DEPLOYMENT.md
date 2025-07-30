# PolyGNN Showcase - Deployment Guide

This guide covers various deployment options for the PolyGNN Showcase Streamlit application.

## Prerequisites

- Python 3.9+
- Git repository with the application code
- Access to model files (`results/final_optimized_model.pth`)

## Deployment Options

### 1. Streamlit Cloud (Recommended)

**Pros**: Free, automatic deployments, easy setup
**Cons**: Limited resources, public repositories only

#### Steps:
1. **Push to GitHub**: Ensure your code is in a public GitHub repository
2. **Visit**: [share.streamlit.io](https://share.streamlit.io)
3. **Connect Account**: Sign in with GitHub
4. **Deploy App**:
   - Repository: `your-username/PolyGNN`
   - Branch: `main`
   - Main file path: `PolyGNNStudio/app.py`
5. **Configure**: App will auto-detect `requirements.txt`

#### Important Notes:
- Model files must be included in repository (see `.gitignore` exceptions)
- Memory limit: ~1GB
- CPU-only execution

### 2. Heroku

**Pros**: Free tier available, easy scaling
**Cons**: Ephemeral filesystem, limited free hours

#### Steps:
1. **Install Heroku CLI**
2. **Login**: `heroku login`
3. **Create App**: `heroku create your-polygnn-app`
4. **Deploy**:
   ```bash
   cd PolyGNNStudio
   git init
   git add .
   git commit -m "Initial deployment"
   heroku git:remote -a your-polygnn-app
   git push heroku main
   ```

#### Files Used:
- `Procfile`: Defines how to run the app
- `setup.sh`: Configures Streamlit for Heroku
- `requirements.txt`: Python dependencies

### 3. Railway

**Pros**: Modern platform, automatic deployments, generous free tier
**Cons**: Newer platform

#### Steps:
1. **Visit**: [railway.app](https://railway.app)
2. **Connect GitHub**: Link your repository
3. **Deploy**: Railway auto-detects Streamlit apps
4. **Configure**: Set start command to `streamlit run app.py --server.port=$PORT`

### 4. Docker (Self-hosted)

**Pros**: Complete control, reproducible environments
**Cons**: Requires server management

#### Steps:
1. **Build Image**:
   ```bash
   cd PolyGNNStudio
   docker build -t polygnn-showcase .
   ```

2. **Run Container**:
   ```bash
   docker run -p 8501:8501 polygnn-showcase
   ```

3. **Access**: `http://localhost:8501`

#### Docker Compose (Optional):
```yaml
version: '3.8'
services:
  polygnn-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### 5. Google Cloud Run

**Pros**: Serverless, pay-per-use, auto-scaling
**Cons**: Requires GCP account, cold starts

#### Steps:
1. **Enable Cloud Run API**
2. **Build and Deploy**:
   ```bash
   gcloud run deploy polygnn-showcase \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### 6. AWS App Runner

**Pros**: Fully managed, auto-scaling
**Cons**: Requires AWS account

#### Steps:
1. **Create App Runner Service**
2. **Connect to GitHub repository**
3. **Configure**:
   - Runtime: Python 3.9
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port=8080 --server.address=0.0.0.0`

## Environment Configuration

### Required Files

| File | Purpose | Required For |
|------|---------|--------------|
| `requirements.txt` | Python dependencies | All platforms |
| `.streamlit/config.toml` | Streamlit configuration | All platforms |
| `Dockerfile` | Container definition | Docker, GCP, AWS |
| `Procfile` | Process definition | Heroku |
| `setup.sh` | Heroku setup script | Heroku |
| `package.json` | Metadata | Some platforms |

### Model Files

The application requires trained model files:
- `../results/final_optimized_model.pth` (primary model)
- `../results/hpo/hpo_20250721_074803/best_model.pth` (backup model)

These are included via `.gitignore` exceptions for deployment.

## Performance Optimization

### Memory Usage
- Expected RAM: 1-2GB for model loading
- Streamlit caching reduces repeated loading
- Consider model quantization for memory-constrained environments

### CPU Requirements
- Single-core sufficient for basic usage
- Multi-core recommended for batch processing
- GPU not required (CPU-optimized inference)

### Scaling Considerations
- App is stateless (suitable for horizontal scaling)
- Model loading happens once per container
- Database not required (demo data embedded)

## Troubleshooting

### Common Deployment Issues

1. **Import Errors**:
   - Verify all dependencies in `requirements.txt`
   - Check Python version compatibility

2. **Model Loading Failures**:
   - Ensure model files are included in deployment
   - Check file paths are correct
   - Verify model file sizes (some platforms have limits)

3. **Memory Issues**:
   - Monitor memory usage during model loading
   - Consider using smaller model variants
   - Implement lazy loading if needed

4. **Port Configuration**:
   - Streamlit default: 8501
   - Use `$PORT` environment variable for platform flexibility
   - Configure in `config.toml` or via command line

### Platform-Specific Issues

**Streamlit Cloud**:
- File size limits (model files)
- Memory constraints (1GB)
- Public repository requirement

**Heroku**:
- Ephemeral filesystem (models reload on restart)
- Memory limits on free tier
- Sleep mode after inactivity

**Docker**:
- RDKit installation issues (use conda if needed)
- PyTorch CPU vs GPU variants
- Container resource limits

## Monitoring and Maintenance

### Health Checks
- Built-in Streamlit health endpoint: `/_stcore/health`
- Custom health checks can be added to `app.py`

### Logging
- Streamlit logs to stdout/stderr
- Application logs through Python logging
- Monitor for model loading errors

### Updates
- Code updates via git push (most platforms)
- Model updates may require redeployment
- Dependencies updates via `requirements.txt`

## Security Considerations

- No authentication built-in (add if needed)
- HTTPS recommended for production
- Environment variables for sensitive config
- Rate limiting may be needed for public deployments

## Cost Estimation

| Platform | Free Tier | Paid Pricing |
|----------|-----------|--------------|
| Streamlit Cloud | ✅ Full | N/A |
| Heroku | 550 hours/month | $7+/month |
| Railway | $5 credit | $0.20/GB-hour |
| Google Cloud Run | 2M requests | $0.40/1M requests |
| AWS App Runner | None | $0.064/hour |

## Support

For deployment issues:
1. Check platform-specific documentation
2. Review application logs
3. Test locally first with same configuration
4. Submit issues to the project repository