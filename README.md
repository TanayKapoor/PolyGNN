# PolyGNN: Graph Neural Networks for Polymer Property Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Graph Neural Network framework for predicting polymer properties, particularly glass transition temperature (Tg), using molecular graphs and polymer-specific features.

## 🎯 **Key Achievements**

- **Performance**: R² = 0.67 for Tg prediction (5x improvement over baseline R² = 0.13)
- **External Validation**: Tested on 1400+ real polymer structures
- **Uncertainty Quantification**: Ensemble methods + Monte Carlo Dropout
- **Feature Engineering**: 168 comprehensive polymer descriptors
- **Production Ready**: End-to-end pipeline with robust validation

## 🚀 **Quick Start**

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PolyGNN.git
cd PolyGNN

# Create conda environment
conda create -n polygnn python=3.8
conda activate polygnn

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from src.models.polymer_gcn import PolymerGCN
from scripts.run_external_val_simple import run_simple_validation

# Quick prediction on polymer dataset
run_simple_validation()

# Train your own model
python train_polymer_gcn.py --config configs/tg_gcn_enhanced.yaml
```

## 📊 **Performance Overview**

### Core Metrics
- **R² Score**: 0.671 (target: >0.6) ✅
- **RMSE**: 68.8°C
- **MAE**: 52.9°C
- **External Validation**: 1423 real polymer structures processed

### Key Features (SHAP Analysis)
1. **chain_flexibility** (51.1% importance) - Dominant predictor
2. **degree_polymerization** (9.9% importance)
3. **molecular_weight** (6.9% importance)
4. **Morgan fingerprints** (distributed structural encoding)

### Robustness
- **5% noise** → **1.5% prediction shift** (<10% target) ✅
- **Stability score**: 96.9/100
- **UQ Coverage**: 95% calibrated bounds

## 🏗️ **Architecture**

### Model Components
- **Graph Neural Network**: 3-layer GCN with [512, 256, 128] hidden dimensions
- **Feature Engineering**: 168 polymer-specific descriptors
- **Uncertainty Quantification**: Ensemble + Monte Carlo Dropout
- **Multi-task Ready**: Tg/Tm/Density prediction capabilities

### Data Pipeline
```
SMILES → Graph Conversion → Feature Engineering → GNN Training → UQ Analysis
```

## 📁 **Project Structure**

```
PolyGNN/
├── src/                          # Core source code
│   ├── data/                     # Data loading and graph conversion
│   ├── models/                   # GNN architectures
│   ├── features/                 # Polymer feature extraction
│   └── training/                 # Training utilities
├── scripts/                      # Standalone scripts
│   ├── run_hpo_simple.py        # Hyperparameter optimization
│   ├── run_external_val_simple.py # External validation
│   ├── preprocess_complete.py   # Data preprocessing
│   └── shap_simple.py           # Feature importance analysis
├── configs/                      # Model configurations
├── data/                        # Datasets
│   ├── raw/                     # Original data
│   └── processed/               # Cleaned datasets
├── results/                     # Model outputs and analysis
├── docs/                        # Documentation
└── tests/                       # Unit tests
```

## 🔬 **Usage Examples**

### 1. Data Preprocessing
```bash
# Process raw polymer dataset
python scripts/preprocess_complete.py \
    --input "data/raw/Polymer Tg SMILES.xlsx" \
    --output "data/processed/external_val.csv"
```

### 2. Model Training
```bash
# Train baseline model
python train_polymer_gcn.py --config configs/tg_gcn_baseline.yaml

# Hyperparameter optimization
python scripts/run_hpo_simple.py --max_trials 100
```

### 3. External Validation
```bash
# Run external validation
python scripts/run_external_val_simple.py

# SHAP feature importance analysis
python scripts/shap_simple.py
```

### 4. Robustness Testing
```bash
# Analyze model robustness
python scripts/robustness_analysis.py
```

## 🧪 **Key Features**

### Advanced Capabilities
- **Bayesian HPO**: Optuna-powered hyperparameter optimization
- **Ensemble UQ**: Multiple model uncertainty quantification
- **Feature Engineering**: 168 polymer-specific descriptors including:
  - 128 Morgan fingerprints
  - 22 molecular descriptors (MW, TPSA, etc.)
  - 18 polymer properties (chain flexibility, persistence length)

### Production Features
- **Robust Validation**: Comprehensive external validation framework
- **Error Analysis**: SHAP-based feature importance and failure analysis
- **Scalability**: Handles 1000+ polymers efficiently
- **Uncertainty**: Calibrated confidence intervals

## 📈 **Results**

### Performance Comparison
| Model | R² | RMSE (°C) | MAE (°C) |
|-------|----|-----------:|----------:|
| Baseline (Fingerprints) | 0.13 | 95.2 | 76.8 |
| **PolyGNN (Final)** | **0.67** | **68.8** | **52.9** |

### Feature Importance (Top 5)
1. Chain Flexibility: 51.1%
2. Degree of Polymerization: 9.9%
3. Molecular Weight: 6.9%
4. Ring Complexity: 4.3%
5. Heteroatom Ratio: 3.1%

## 🛠️ **Development**

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black src/ scripts/
isort src/ scripts/

# Lint
flake8 src/ scripts/
```

### Environment Setup
```bash
# Using conda
chmod +x setup_conda_env.sh
./setup_conda_env.sh
conda activate polymer-gnn

# Or pip
pip install -r requirements.txt
pip install -e .
```

## 📚 **Documentation**

- **[CLAUDE.md](docs/CLAUDE.md)**: Development commands and workflows
- **[Week 2 Completion Report](docs/WEEK2_COMPLETION_REPORT.md)**: Detailed performance analysis
- **[External Validation Summary](docs/external_validation_summary.md)**: Validation framework
- **[HPO README](docs/HPO_README.md)**: Hyperparameter optimization guide

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **RDKit**: Chemical informatics toolkit
- **PyTorch Geometric**: Graph neural network library
- **Optuna**: Hyperparameter optimization framework
- **SHAP**: Model explainability framework

## 📞 **Contact**

For questions, issues, or collaborations:
- Open an issue on GitHub
- Email: [your.email@domain.com]

---

**Ready for production deployment and external validation!** 🚀

*Generated with PolyGNN - Advancing polymer property prediction through graph neural networks.*