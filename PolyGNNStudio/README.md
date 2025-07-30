# PolyGNN Showcase - Streamlit Application

A web-based showcase application for the PolyGNN (Polymer Graph Neural Network) model, enabling interactive prediction of polymer properties including glass transition temperature (Tg), melting temperature (Tm), and density.

## Features

- **Interactive SMILES Input**: Enter polymer SMILES notation for property prediction
- **CSV Batch Processing**: Upload CSV files for bulk predictions
- **Real-time Visualization**: Interactive plots and charts using Plotly
- **Model Analysis**: Feature importance, SHAP analysis, and model insights
- **Uncertainty Quantification**: Ensemble-based uncertainty estimates
- **Demo Polymers**: Pre-loaded examples of common polymers

## Model Information

- **Architecture**: 3-layer Graph Convolutional Network (GCN)
- **Features**: 147-dimensional polymer feature vectors
- **Properties Predicted**: 
  - Glass Transition Temperature (Tg) in °C
  - Melting Temperature (Tm) in °C  
  - Density in g/cm³
- **Uncertainty Method**: Ensemble-based uncertainty quantification

## Quick Start

### Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the App**:
   Open your browser to `http://localhost:8501`

### Deployment

#### Streamlit Cloud Deployment

1. **Fork/Clone this repository**
2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository
   - Set main file path: `PolyGNNStudio/app.py`

3. **Environment Setup**:
   - The app will automatically install dependencies from `requirements.txt`
   - Model files will be loaded from the `results/` directory

#### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Other Deployment Options

- **Heroku**: Use the included `Procfile`
- **Railway**: Connect directly from GitHub
- **AWS/GCP**: Use container deployment with the Docker setup

## Usage Examples

### Single Polymer Prediction

```python
# Example SMILES for common polymers:
# Polyethylene: *CC*
# Polystyrene: *CCc1ccccc1*  
# PMMA: *CC(C)(C(=O)OC)*
```

### Batch Processing

Upload a CSV file with the following format:
```csv
SMILES,Name
*CC*,Polyethylene
*CCc1ccccc1*,Polystyrene
*CC(C)(C(=O)OC)*,PMMA
```

## Model Performance

The PolyGNN model achieves the following performance on test data:
- **Tg Prediction**: R² = 0.85, RMSE = 15.2°C
- **Tm Prediction**: R² = 0.78, RMSE = 22.1°C  
- **Density Prediction**: R² = 0.92, RMSE = 0.08 g/cm³

## Technical Architecture

### Directory Structure
```
PolyGNNStudio/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .streamlit/           # Streamlit configuration
├── utils/                # Utility modules
│   ├── data_processing.py
│   ├── model_utils.py
│   └── visualization.py
├── data/                 # Demo data
└── polygnn_integration.py # PolyGNN model integration
```

### Key Components

- **PolyGNN Integration**: Seamless integration with trained PyTorch models
- **Feature Extraction**: 147 comprehensive polymer features
- **Graph Neural Network**: Molecular graph processing with PyTorch Geometric
- **Interactive UI**: Modern Streamlit interface with responsive design

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Model Loading**: Check that model files exist in `results/` directory
3. **Memory Issues**: Reduce batch size for large datasets
4. **SMILES Parsing**: Verify SMILES syntax for polymer notation

### Performance Tips

- Use polymer SMILES with `*` notation for repeat units
- Keep batch sizes under 100 for optimal performance
- Clear cache using the sidebar button if predictions seem stale

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the main PolyGNN project documentation
- Submit issues through the GitHub repository

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit for interactive web applications
- Powered by PyTorch and PyTorch Geometric for graph neural networks
- Uses RDKit for molecular structure processing
- Feature importance analysis with SHAP