# Polymer GNN: Graph Neural Networks for Polymer Property Prediction

A comprehensive machine learning framework for predicting polymer properties using Graph Neural Networks (GNNs) with BigSMILES polymer notation.

## Features

- **BigSMILES Parser**: Custom parser for polymer notation with molecular feature extraction
- **Graph Neural Network Models**: PyTorch Geometric-based models for polymer property prediction
- **Multi-Task Learning**: Predict multiple polymer properties simultaneously
- **Comprehensive Environment**: Pre-configured conda environment with all necessary dependencies
- **GPU Support**: CUDA-enabled for accelerated training and inference

## Quick Start

### 1. Environment Setup

Choose one of the following methods:

#### Option A: Using the setup script (Recommended)
```bash
chmod +x setup_conda_env.sh
./setup_conda_env.sh
```

#### Option B: Using environment.yml
```bash
conda env create -f environment.yml
conda activate polymer-gnn
```

#### Option C: Manual setup
```bash
conda create -n polymer-gnn python=3.9 -y
conda activate polymer-gnn
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
conda activate polymer-gnn
python verify_conda_setup.py
```

### 3. Setup Datasets

```bash
python dataset_setup.py
```

### 4. Test BigSMILES Parser

```bash
python src/data/bigsmiles_parser.py
```

## Project Structure

```
polymer-gnn/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   │   ├── bigsmiles_parser.py
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── models/            # Model implementations
│   │   ├── polymer_gnn.py
│   │   └── multi_task_model.py
│   ├── training/          # Training utilities
│   │   ├── trainer.py
│   │   └── utils.py
│   └── evaluation/        # Evaluation metrics
│       └── metrics.py
├── data/                  # Data storage
│   ├── raw/              # Raw datasets
│   ├── processed/        # Processed datasets
│   ├── external/         # External datasets
│   └── interim/          # Intermediate data
├── notebooks/            # Jupyter notebooks
├── tests/               # Unit tests
├── models/              # Trained models
├── results/             # Results and outputs
├── docs/                # Documentation
├── environment.yml      # Conda environment
├── requirements.txt     # Python dependencies
├── setup_conda_env.sh   # Environment setup script
├── verify_conda_setup.py # Verification script
├── dataset_setup.py     # Dataset setup script
└── README.md           # This file
```

## BigSMILES Parser

The BigSMILES parser handles extended SMILES notation for polymers:

```python
from src.data.bigsmiles_parser import BigSMILESParser

parser = BigSMILESParser()

# Parse polyethylene
result = parser.parse_bigsmiles("CC{[CH2][CH2]}CC")
print(f"Repeat units: {result['repeat_units']}")
print(f"Molecular weight: {result['avg_repeat_mw']}")
```

## Environment Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 10GB+ free space

### Software Requirements
- **Python**: 3.9+
- **CUDA**: 11.8+ (for GPU support)
- **Conda**: Latest version

## GPU Configuration

Check your GPU setup:
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
We follow PEP 8 style guidelines. Run linting with:
```bash
flake8 src/
```

### Adding New Models
1. Create model file in `src/models/`
2. Add imports to `src/models/__init__.py`
3. Add tests in `tests/`

## Datasets

### NeurIPS 2025 Polymer Dataset
- **Source**: Competition dataset
- **Format**: CSV with BigSMILES notation
- **Properties**: Glass transition temperature, melting temperature, density, etc.

### External Datasets
- Polymer property databases
- Literature data
- Experimental measurements

## Citation

```bibtex
@software{polymer_gnn,
  title={Polymer GNN: Graph Neural Networks for Polymer Property Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/polymer-gnn}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For questions and support:
- Create an issue on GitHub
- Email: your.email@example.com

## Changelog

### v0.1.0
- Initial release
- BigSMILES parser implementation
- Environment setup scripts
- Basic project structure 