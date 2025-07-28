# PolyGNN Showcase Application

## Overview

The PolyGNN Showcase is a Streamlit-based web application designed to demonstrate polymer property predictions using Graph Neural Networks (GNNs). The application allows users to input polymer SMILES notation and receive predictions for key polymer properties including glass transition temperature (Tg), melting temperature (Tm), and density. The application features an interactive web interface with data visualization capabilities and supports both individual SMILES input and batch CSV uploads.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **UI Components**: Interactive widgets for SMILES input, file uploads, and parameter controls
- **Visualization**: Plotly-based interactive charts and graphs
- **State Management**: Streamlit session state for maintaining user data across interactions
- **Responsive Design**: Wide layout configuration for optimal data visualization

### Backend Architecture
- **Core Logic**: Python-based modular architecture with utility modules
- **Model Integration**: Placeholder structure for PolyGNN ensemble model integration
- **Data Processing**: RDKit-based molecular validation and processing pipeline
- **Prediction Engine**: Ensemble prediction system with uncertainty quantification

### Data Processing Pipeline
- **Input Validation**: SMILES notation validation using RDKit molecular parsing
- **Batch Processing**: CSV file upload and processing capabilities
- **Data Transformation**: Molecular descriptor calculation and feature extraction
- **Output Formatting**: Structured prediction results with uncertainty estimates

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Primary Streamlit application entry point
- **Features**: Page configuration, session state management, and main UI orchestration
- **Caching**: Resource caching for model loading optimization

### 2. Data Processing Module (`utils/data_processing.py`)
- **SMILES Validation**: Chemical structure validation using RDKit
- **CSV Processing**: Batch file upload handling and data validation
- **Error Handling**: Comprehensive input validation with user-friendly error messages

### 3. Model Utilities (`utils/model_utils.py`)
- **Model Loading**: Placeholder infrastructure for PolyGNN model integration
- **Prediction Pipeline**: Ensemble prediction system with uncertainty quantification
- **Property Predictions**: Support for Tg, Tm, and density predictions with realistic ranges

### 4. Visualization Module (`utils/visualization.py`)
- **Interactive Plots**: Plotly-based scatter plots, histograms, and SHAP plots
- **Performance Metrics**: R² calculation and prediction accuracy visualization
- **Error Visualization**: Uncertainty quantification with error bars

### 5. Demo Data (`data/demo_polymers.py`)
- **Sample Polymers**: Pre-loaded polymer examples for demonstration
- **Property Data**: Experimental values for common polymers (PE, PS, PVC, PMMA)
- **Application Context**: Real-world polymer applications and descriptions

## Data Flow

1. **Input Stage**: User provides SMILES notation via text input or CSV upload
2. **Validation Stage**: SMILES validation using RDKit molecular parsing
3. **Processing Stage**: Molecular descriptor calculation and feature extraction
4. **Prediction Stage**: Ensemble model prediction with uncertainty quantification
5. **Visualization Stage**: Interactive plot generation and results display
6. **Output Stage**: Downloadable results and performance metrics

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for interactive UI
- **RDKit**: Chemical informatics library for molecular processing
- **Plotly**: Interactive visualization library for charts and graphs
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning utilities for metrics calculation

### Model Dependencies
- **PolyGNN Framework**: Graph neural network implementation (to be integrated)
- **PyTorch/TensorFlow**: Deep learning framework for model execution
- **SHAP**: Model interpretability for feature importance analysis

## Deployment Strategy

### Development Environment
- **Platform**: Replit-based development with Python runtime
- **Dependencies**: Requirements managed through package installation
- **Local Testing**: Streamlit development server for rapid iteration

### Production Considerations
- **Containerization**: Docker containerization for consistent deployment
- **Model Storage**: Separate storage solution for trained model weights
- **Performance**: Model caching and optimization for production workloads
- **Scalability**: Horizontal scaling considerations for multiple users

### Integration Points
- **Model Integration**: Placeholder structure ready for PolyGNN model integration
- **Database Support**: Extensible architecture for future database integration
- **API Extensions**: Modular design allows for REST API development
- **Cloud Deployment**: Architecture supports cloud platform deployment (AWS, GCP, Azure)

## Recent Updates (July 28, 2025)

### Enhanced UI and Functionality
- **Dark Theme**: Implemented dark theme as default with custom color scheme
- **SMILES Visualization**: Added 2D molecular structure rendering using RDKit Draw.MolToImage
- **Interactive Spinners**: Enhanced loading states with progress bars and status updates
- **Enhanced Plotly Charts**: All visualizations now use dark theme with improved interactivity

### New Features Added
- **Feature Sensitivity Analysis**: Interactive sliders to perturb features and analyze prediction changes
- **Extended Demo Library**: Expanded to 10 diverse polymer samples with structure visualization
- **Enhanced Metrics Tab**: Added sub-tabs for Property Analysis, Feature Sensitivity, and Model Insights
- **Interactive Demo Examples**: Two-tab structure with polymer library and interactive predictions
- **Structure Comparison**: Side-by-side visualization of polymer structures and predictions
- **Model Analysis Tab**: Comprehensive performance analysis with 4 sub-sections:
  - Performance Metrics (multi-task R²/RMSE/MAE tables, external test performance)
  - SHAP Insights (beeswarm plots, top-10 feature importance with chain_flexibility at 51%)
  - UQ & Robustness (95% confidence interval coverage, noise sensitivity analysis)
  - Model Overview (3-layer GCN ensemble, 91k parameters, Optuna HPO with 200 trials)

### Technical Improvements
- **Progress Indicators**: Added progress bars and status text for better user feedback
- **Cached Structure Rendering**: Implemented caching for SMILES structure generation
- **Enhanced Error Handling**: Better validation and error messages throughout the application
- **Modular Visualization**: Expanded visualization utilities with sensitivity analysis plots

## Notes for Code Agent

- The application currently uses placeholder implementations for the actual PolyGNN model
- Real model integration will require replacing placeholder functions in `model_utils.py`
- Database integration may be added later using Drizzle ORM with Postgres
- The modular architecture facilitates easy extension and maintenance
- All external dependencies are well-documented and version-controlled
- SMILES structure rendering requires the expat system dependency for RDKit drawing functionality