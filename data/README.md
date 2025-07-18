# Data Directory Structure

This directory contains all data files for the PolyGNN project.

## Directory Structure

```
data/
├── raw/                    # Raw, unprocessed data files
│   └── train.csv          # Original polymer dataset
├── processed/             # Processed datasets ready for training
│   ├── filtered_tg_dataset.csv      # Tg prediction dataset (510 samples)
│   ├── filtered_ffv_dataset.csv     # FFV prediction dataset (7,029 samples)
│   └── data_quality_report.json     # Data quality assessment report
├── external/              # External datasets and references
├── interim/               # Intermediate data files during processing
└── README.md             # This file
```

## Dataset Description

### Glass Transition Temperature (Tg) Dataset
- **File**: `processed/filtered_tg_dataset.csv`
- **Samples**: 510 polymer structures
- **Target**: Glass transition temperature (°C)
- **Range**: -148.03°C to 472.25°C
- **Quality**: 93.5% valid SMILES after preprocessing

### Free Volume Fraction (FFV) Dataset
- **File**: `processed/filtered_ffv_dataset.csv`
- **Samples**: 7,029 polymer structures
- **Target**: Free volume fraction (dimensionless)
- **Range**: 0.23 to 0.78
- **Quality**: 100% valid SMILES after preprocessing

## Data Processing

The raw polymer dataset contains repeat units with connection points (`*`) that are preprocessed to create molecular graphs suitable for GNN training. The preprocessing includes:

1. **Connection point removal**: `*` symbols are removed from SMILES strings
2. **Validation**: Invalid SMILES are filtered out
3. **Quality assessment**: Molecular properties and outliers are analyzed
4. **Train/validation/test splits**: 70/15/15 split for model training

## Usage

The processed datasets are ready for use with the `PolymerTgDataset` class:

```python
from src.data import PolymerTgDataset

# Load Tg dataset
dataset = PolymerTgDataset(
    root='./data/processed',
    csv_file='data/processed/filtered_tg_dataset.csv',
    smiles_column='processed_smiles',
    target_column='Tg',
    split_type='train'
)

# Load FFV dataset
dataset = PolymerTgDataset(
    root='./data/processed', 
    csv_file='data/processed/filtered_ffv_dataset.csv',
    smiles_column='processed_smiles',
    target_column='FFV',
    split_type='train'
)
```

## Notes

- Raw data files are ignored by git (see `.gitignore`)
- Processed files are available for immediate use
- Data quality reports provide detailed statistics and validation results 