# PolyGNN Showcase - Deployment Guide

## Streamlit Cloud Deployment

This app is designed to work robustly on Streamlit Cloud with automatic fallbacks for missing dependencies.

### Deployment Files

- **`requirements.txt`**: Main requirements with PyTorch and RDKit
- **`requirements-minimal.txt`**: Fallback requirements without PyTorch (demo mode only)
- **`packages.txt`**: System packages for RDKit X11 support
- **`.streamlit/config.toml`**: Streamlit configuration

### App Modes

1. **Full Mode**: When PyTorch, PyTorch Geometric, and trained model are available
2. **Demo Mode**: When dependencies are missing - generates synthetic predictions

### Automatic Fallbacks

The app gracefully handles missing dependencies:

- **PyTorch missing**: ✅ Switches to demo mode with synthetic predictions
- **RDKit missing**: ✅ Disables structure visualization, keeps core functionality  
- **Model file missing**: ✅ Uses demo mode with realistic synthetic data
- **X11 libraries missing**: ✅ Disables molecular drawing, shows warnings

### Error Handling

All critical imports are wrapped in try-catch blocks:
- `torch` and `torch_geometric` → Demo mode
- `rdkit` → Structure visualization disabled
- Model loading → Synthetic predictions

### Demo Mode Features

When PyTorch is unavailable, demo mode provides:
- ✅ SMILES-based synthetic predictions
- ✅ Realistic property ranges (Tg, Tm, Density)
- ✅ Uncertainty quantification simulation  
- ✅ Full UI functionality
- ✅ Visualization and analysis tools

### Deployment Steps

1. **Push to GitHub** (ensure all files are included)
2. **Connect to Streamlit Cloud**
3. **Set main file**: `PolyGNNStudio/app.py`
4. **App will auto-detect capabilities** and switch to appropriate mode

### Files Not Included in Git

- `final_optimized_model.pth` (model weights)
- Large dependency files

These trigger demo mode automatically - no manual configuration needed.

### Testing Locally

```bash
# Test with full dependencies
streamlit run app.py

# Test demo mode (rename requirements temporarily)
mv requirements.txt requirements-full.txt
cp requirements-minimal.txt requirements.txt
streamlit run app.py
mv requirements-full.txt requirements.txt
```

### Expected Behavior on Streamlit Cloud

- ✅ App loads successfully
- 🎭 Demo mode banner appears  
- ✅ All UI elements functional
- 🔄 Synthetic predictions generated
- ⚠️ Warnings about missing features (structure viz)
- 📊 Full visualization capabilities

The app is designed to provide a complete demonstration experience even without the actual trained model.