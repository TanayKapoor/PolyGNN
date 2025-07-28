# PolyGNN - Polymer Graph Neural Networks

A comprehensive framework for predicting polymer properties using Graph Neural Networks (GNNs). This project focuses on glass transition temperature (Tg) and free volume fraction (FFV) prediction from polymer molecular structures.

## 🚀 Features

- **Real Polymer Dataset**: 7,973 polymer structures with multiple properties
- **SMILES to Graph Conversion**: Automated molecular graph generation from SMILES strings
- **Multi-property Prediction**: Support for Tg, FFV, and other polymer properties
- **Advanced Polymer Features**: 147 polymer-specific features including molecular weight, degree of polymerization, and structural complexity
- **Data Quality Assessment**: Comprehensive preprocessing and validation
- **PyTorch Geometric Integration**: Built on modern GNN libraries
- **Hyperparameter Optimization**: Built-in HPO system with advanced configurations
- **Modular Architecture**: Clean, extensible codebase

## 📊 Dataset Statistics

| Property | Samples | Range | Quality |
|----------|---------|--------|---------|
| Glass Transition Temperature (Tg) | 510 | -148.03°C to 472.25°C | 93.5% valid |
| Free Volume Fraction (FFV) | 7,029 | 0.23 to 0.78 | 100% valid |

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Conda (recommended)

### Option 1: Conda Environment (Recommended)
```bash
# Clone the repository
git clone https://github.com/user/polygnn.git
cd polygnn

# Create and activate conda environment
chmod +x setup_conda_env.sh
./setup_conda_env.sh
conda activate polymer-gnn

# Install the package
pip install -e .
```

### Option 2: Pip Installation
```bash
# Install requirements
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 🔧 Quick Start

### 1. Dataset Setup
```bash
# Process polymer dataset for Tg prediction
python dataset_setup.py --target_property Tg

# Process dataset for FFV prediction
python dataset_setup.py --target_property FFV

# Multi-target setup
python dataset_setup.py --multi_target
```

### 2. Basic Usage
```python
from src.data import PolymerTgDataset, MolecularGraphConverter

# Load dataset
dataset = PolymerTgDataset(
    root='./data/processed',
    csv_file='data/processed/filtered_tg_dataset.csv',
    smiles_column='processed_smiles',
    target_column='Tg',
    split_type='train'
)

# Convert SMILES to graph
converter = MolecularGraphConverter()
graph = converter.smiles_to_graph('CCO')  # Ethanol

print(f"Nodes: {graph.num_nodes}")
print(f"Edges: {graph.edge_index.shape[1]}")
```

### 3. Data Loading
```python
from torch_geometric.data import DataLoader

# Create data loader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch in train_loader:
    print(f"Batch size: {batch.batch.max().item() + 1}")
    print(f"Total nodes: {batch.x.shape[0]}")
    print(f"Total edges: {batch.edge_index.shape[1]}")
    break
```

## 📁 Project Structure

```
polygnn/
├── src/                           # Core source code
│   ├── data/                      # Data processing modules
│   │   ├── bigsmiles_parser.py    # BigSMILES parsing utilities
│   │   ├── molecular_graph.py     # SMILES to graph conversion
│   │   └── polymer_dataset.py     # Dataset class and utilities
│   ├── models/                    # GNN model implementations
│   ├── training/                  # Training utilities
│   └── evaluation/                # Evaluation metrics and visualization
├── data/                          # Data files
│   ├── raw/                       # Raw dataset files
│   ├── processed/                 # Processed datasets
│   ├── external/                  # External datasets
│   └── interim/                   # Intermediate processing files
├── notebooks/                     # Jupyter notebooks
├── docs/                          # Documentation
├── tests/                         # Unit tests
├── models/                        # Trained model files
├── results/                       # Experiment results
├── dataset_setup.py               # Dataset processing script
├── environment.yml                # Conda environment
├── requirements.txt               # Python dependencies
└── setup.py                       # Package installation
```

## 🔬 Data Processing

The framework handles polymer repeat units with connection points (`*`) and provides:

1. **SMILES Preprocessing**: Removes connection points and validates structures
2. **Graph Conversion**: Creates molecular graphs with node and edge features
3. **Quality Assessment**: Comprehensive data validation and reporting
4. **Train/Val/Test Splits**: Automated 70/15/15 data splitting

## 🧪 Molecular Features

- **Node Features**: 157-dimensional atom descriptors
- **Edge Features**: 14-dimensional bond descriptors
- **Molecular Features**: 13-dimensional molecular descriptors
- **Graph Size**: Up to 300 atoms supported

## 🎯 Supported Properties

- **Tg**: Glass transition temperature (°C)
- **FFV**: Free volume fraction (dimensionless)
- **Tc**: Critical temperature (°C)
- **Density**: Polymer density (g/cm³)
- **Rg**: Radius of gyration (Å)

## 🏗️ Development

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Check imports
isort src/ tests/

# Lint code
flake8 src/ tests/
```

## 📈 Performance

The framework is optimized for:
- **Memory efficiency**: Handles large datasets (7K+ samples)
- **Fast processing**: Batch processing and GPU support
- **Scalability**: Modular design for easy extension

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- RDKit for molecular processing
- PyTorch Geometric for GNN framework
- The polymer science community for datasets and insights

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{polygnn2024,
  title={PolyGNN: Graph Neural Networks for Polymer Property Prediction},
  author={PolyGNN Development Team},
  year={2024},
  url={https://github.com/user/polygnn}
}
```

## 🔗 Links

- [Documentation](./docs/)
- [Dataset Information](./data/README.md)
- [Jupyter Notebooks](./notebooks/)
- [Issue Tracker](https://github.com/user/polygnn/issues) 