# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment
chmod +x setup_conda_env.sh
./setup_conda_env.sh
conda activate polymer-gnn

# Alternative: pip installation
pip install -r requirements.txt
pip install -e .
```

### Data Processing
```bash
# Setup dataset for Tg prediction
python dataset_setup.py --target_property Tg

# Setup dataset for FFV prediction  
python dataset_setup.py --target_property FFV

# Multi-target setup
python dataset_setup.py --multi_target
```

### Model Training
```bash
# Train baseline fingerprint models
python train_polymer_baselines.py --config configs/tg_fingerprint_baseline.yaml
python train_polymer_baselines.py --config configs/tg_fingerprint_baseline.yaml --cv

# Train GCN models
python train_polymer_gcn.py --config configs/tg_gcn_baseline.yaml
python train_polymer_gcn.py --config configs/tg_gcn_enhanced.yaml --cv

# Run hyperparameter optimization
python run_hpo_optimization.py --method grid --max_trials 100
python run_hpo_from_config.py --config configs/hpo_config.yaml
```

### Analysis and Reporting
```bash
# Quick failure analysis
python analysis/quick_failure_analysis.py

# Comprehensive model analysis
python analysis/failure_analysis.py

# Feature importance analysis
python analysis/feature_importance.py

# SHAP analysis
python analysis/shap_failure_analysis.py

# HPO reporting
python analysis/hpo_report.py
```

### Development Tools
```bash
# Run tests
pytest tests/ -v

# Code formatting and quality (available via setup.py dev extras)
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## Architecture Overview

### Core Components

**Data Pipeline (`src/data/`)**
- `molecular_graph.py`: SMILES to graph conversion using RDKit and PyTorch Geometric
- `polymer_dataset.py`: PyTorch dataset classes for polymer data with train/val/test splits
- `bigsmiles_parser.py`: Parser for BigSMILES polymer notation with connection points

**Models (`src/models/`)**
- `polymer_gcn.py`: Graph Convolutional Network for molecular graphs with configurable architecture
- `polymer_baseline.py`: Baseline models using molecular fingerprints and polymer-specific features

**Training (`src/training/`)**
- `gcn_trainer.py`: Training loop for GCN models with cross-validation and evaluation
- `trainer.py`: Training utilities for baseline models

**Features (`src/features/`)**
- `polymer_features.py`: Polymer-specific feature extraction (147 features including MW, DP, structural complexity)

### Data Flow

1. **Raw Data**: Real polymer dataset in `data/raw/train.csv` (7,973 structures)
2. **Processing**: SMILES preprocessing removes connection points (*), validates structures
3. **Graph Generation**: Molecular graphs with 157-dim node features, 14-dim edge features
4. **Feature Engineering**: 147 polymer-specific features + molecular descriptors
5. **Model Training**: Supports baseline (fingerprints) and GCN approaches
6. **Analysis**: Comprehensive failure analysis and feature importance tools

### Key Design Patterns

- **Configuration-driven**: All models use YAML configs in `configs/` directory
- **Modular architecture**: Clear separation between data, models, training, and analysis
- **Reproducible experiments**: Built-in logging, model checkpointing, and result tracking
- **Multi-property support**: Framework handles Tg, FFV, Tc, Density, and Rg prediction

### Target Properties

- **Tg**: Glass transition temperature (°C) - 510 samples, primary focus
- **FFV**: Free volume fraction - 7,029 samples
- **Tc, Density, Rg**: Additional polymer properties for multi-target learning

### Performance Targets

HPO optimization aims for: R² ≥ 0.5, RMSE ≤ 50.0, MAE ≤ 30.0

## Important Notes

- Graph size limited to 300 atoms maximum
- Handles polymer repeat units with connection points (*)
- Uses comprehensive data quality assessment and filtering
- Results stored in `results/` with timestamped analysis reports
- Extensive HPO results in `results/hpo/` with trial-by-trial tracking
- Jupyter notebooks in `notebooks/` for interactive analysis