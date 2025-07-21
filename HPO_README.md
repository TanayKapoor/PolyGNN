# Hyperparameter Optimization for Polymer GCN

This document describes the comprehensive hyperparameter optimization (HPO) system implemented for the PolymerGCN model. The system aims to achieve target performance metrics: **R² ≥ 0.5**, **RMSE ≤ 50.0**, **MAE ≤ 30.0**.

## 🎯 Overview

The HPO system provides:
- **Grid Search** and **Random Search** optimization methods
- **Cross-validation** for robust performance estimation  
- **Comprehensive parameter spaces** targeting molecular GNN performance
- **Automatic model retraining** with best parameters
- **Detailed reporting** and visualization
- **Configuration-based** easy experimentation

## 🚀 Quick Start

### 1. Quick Test Run (5-10 minutes)
```bash
python run_hpo_from_config.py --preset quick
```

### 2. Standard HPO Run (2-4 hours)
```bash
python run_hpo_from_config.py --preset standard
```

### 3. Comprehensive Search (8-12 hours)
```bash
python run_hpo_from_config.py --preset comprehensive
```

### 4. Custom Configuration
```bash
python run_hpo_from_config.py --config configs/hpo_config.yaml --search_type aggressive_search
```

## 📁 Files Overview

| File | Description |
|------|-------------|
| `src/training/gcn_trainer.py` | Enhanced trainer with HPO capabilities |
| `run_hpo_optimization.py` | Main HPO script with command-line interface |
| `run_hpo_from_config.py` | Configuration-based HPO runner |
| `configs/hpo_config.yaml` | HPO configuration file with search spaces |
| `analysis/hpo_report.py` | Results analysis and reporting |

## 🔧 Usage Examples

### Basic HPO with Command Line
```bash
# Random search with 50 trials
python run_hpo_optimization.py --method random --max_trials 50

# Grid search (warning: potentially many combinations)
python run_hpo_optimization.py --method grid --cv_folds 5

# Custom settings
python run_hpo_optimization.py \
    --method random \
    --max_trials 100 \
    --cv_folds 5 \
    --primary_metric r2 \
    --max_epochs 75 \
    --device cuda
```

### Configuration-Based HPO
```bash
# Using presets
python run_hpo_from_config.py --preset quick          # 10 trials, 3 CV folds
python run_hpo_from_config.py --preset standard       # 50 trials, 5 CV folds  
python run_hpo_from_config.py --preset comprehensive  # 100 trials, aggressive search

# Custom config file
python run_hpo_from_config.py --config my_hpo_config.yaml

# Override config settings
python run_hpo_from_config.py \
    --config configs/hpo_config.yaml \
    --search_type aggressive_search \
    --max_trials 200 \
    --device cuda
```

### Generate Analysis Reports
```bash
# Analyze specific HPO run
python analysis/hpo_report.py --hpo_dir results/hpo/hpo_20231215_143022

# Analyze all HPO runs
python analysis/hpo_report.py --results_dir results --generate_all
```

## ⚙️ Configuration

### Search Spaces Available

| Search Type | Description | Combinations | Recommended Use |
|-------------|-------------|--------------|-----------------|
| `conservative_search` | Small focused space | ~64 | Quick testing |
| `grid_search` | Standard grid | ~6,400 | Thorough grid search |
| `random_search` | Balanced exploration | User-defined trials | Standard HPO |
| `aggressive_search` | Very comprehensive | Large | Research/production |

### Key Parameters Optimized

**Model Architecture:**
- `hidden_dims`: Network layer sizes (e.g., [128, 64, 32])
- `num_gcn_layers`: Number of GCN layers (2-6)
- `dropout_rate`: Regularization strength (0.0-0.6)
- `pooling_method`: Graph pooling ('mean', 'max', 'sum')
- `activation`: Activation function ('relu', 'gelu', 'tanh')

**Training:**
- `learning_rate`: Optimizer learning rate (1e-6 to 2e-2)
- `weight_decay`: L2 regularization (0 to 1e-2)
- `batch_size`: Training batch size (8-512)

**Features:**
- `use_molecular_features`: Include molecular descriptors
- `use_polymer_features`: Include polymer-specific features

## 📊 Results and Analysis

### Output Structure
```
results/
├── hpo/
│   └── hpo_20231215_143022/          # Timestamped HPO run
│       ├── hpo_results.csv           # Trial-by-trial results
│       ├── hpo_summary.json          # Complete HPO summary
│       ├── best_model.pth            # Best model from HPO
│       ├── hpo_comprehensive_report.pdf  # Detailed analysis
│       └── parameter_summary.csv     # Parameter performance summary
├── final_model/                      # Final retrained model
│   ├── final_optimized_model.pth     # Best model retrained on full data
│   └── final_optimization_results.json
└── final_optimization_results.json   # Complete results with test metrics
```

### Success Criteria Evaluation
The system automatically evaluates against target metrics:
- ✅ **R² ≥ 0.5**: Model explains at least 50% of variance
- ✅ **RMSE ≤ 50.0**: Root mean square error under 50 K
- ✅ **MAE ≤ 30.0**: Mean absolute error under 30 K

### Analysis Reports
The system generates comprehensive PDF reports including:
- Performance distributions and trends
- Parameter importance analysis  
- Best configuration details
- Correlation analysis
- Recommendations for further improvement

## 🔬 Advanced Usage

### Custom Parameter Spaces
Edit `configs/hpo_config.yaml` to define custom search spaces:

```yaml
parameter_grids:
  my_custom_search:
    learning_rate: [1e-4, 5e-4, 1e-3]
    hidden_dims:
      - [256, 128, 64]
      - [512, 256, 128]  
    num_gcn_layers: [3, 4, 5]
    # ... more parameters
```

### Programmatic Usage
```python
from src.training.gcn_trainer import PolymerGCNTrainer
from src.models.polymer_gcn import PolymerGCN

# Create trainer
trainer = PolymerGCNTrainer(model, device='cuda')

# Define search space
param_grid = {
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'hidden_dims': [[128, 64], [256, 128]],
    'num_gcn_layers': [3, 4, 5]
}

# Run HPO
results = trainer.hyperparam_optimize(
    dataset=dataset,
    param_grid=param_grid,
    method='random',
    n_trials=50,
    cv_folds=5
)

# Retrain with best params
final_results = trainer.retrain_with_best_params(
    results, full_dataset, test_dataset
)
```

### Integration with Existing Code
The HPO system seamlessly integrates with existing training scripts:

```python
# In your existing training script
from src.training.gcn_trainer import PolymerGCNTrainer

# Replace your trainer initialization
trainer = PolymerGCNTrainer(model)

# Add HPO before training
if run_hpo:
    hpo_results = trainer.hyperparam_optimize(dataset, param_grid)
    # Model will be updated with best parameters

# Continue with normal training...
```

## 🔍 Monitoring and Debugging

### Real-time Monitoring
- Progress bars show current trial and overall progress
- Live logging displays trial results as they complete
- CSV files update in real-time for external monitoring

### Log Files
```bash
# Main HPO log
tail -f hpo_optimization.log

# Individual trial logs
ls results/hpo/hpo_*/trial_*/
```

### Common Issues and Solutions

**Issue: Out of Memory**
```bash
# Reduce batch size and model size
python run_hpo_optimization.py --search_type conservative_search --device cuda
```

**Issue: Long Training Times**
```bash
# Reduce epochs and patience for HPO trials
python run_hpo_from_config.py --preset quick --max_trials 20
```

**Issue: Poor Performance**
```bash
# Try more aggressive search space
python run_hpo_from_config.py --search_type aggressive_search --max_trials 100
```

## 📈 Expected Results

Based on similar polymer Tg prediction tasks, you can expect:

**Quick Preset (10 trials):**
- Time: 5-15 minutes
- Expected R²: 0.25-0.45
- Purpose: Rapid prototyping

**Standard Preset (50 trials):**
- Time: 2-4 hours  
- Expected R²: 0.35-0.55
- Purpose: Production HPO

**Comprehensive Preset (100+ trials):**
- Time: 8-12 hours
- Expected R²: 0.45-0.65
- Purpose: Research-grade optimization

## 🛠️ Customization

### Adding New Metrics
Modify `src/training/gcn_trainer.py`:
```python
def _calculate_metrics(self, predictions, targets, loss):
    # Add your custom metrics here
    custom_metric = your_metric_function(targets, predictions)
    return {
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'custom_metric': custom_metric
    }
```

### New Search Methods
Extend the HPO system with Bayesian optimization, evolutionary algorithms, etc.:
```python
# In hyperparam_optimize method
elif method == 'bayesian':
    param_combinations = self._bayesian_optimization(param_grid, n_trials)
```

### Custom Success Criteria
Edit success criteria in configs or code:
```yaml
success_criteria:
  r2_target: 0.6      # Stricter R² requirement
  rmse_target: 40.0   # Stricter RMSE requirement  
  mae_target: 25.0    # Stricter MAE requirement
```

## 🤝 Contributing

To extend the HPO system:
1. Add new parameter spaces in config files
2. Implement new search algorithms in the trainer
3. Enhance reporting with additional analysis
4. Add new metrics and success criteria

## 📚 References

- PyTorch Geometric documentation for GNN implementation details
- Polymer informatics literature for Tg prediction benchmarks  
- Hyperparameter optimization best practices for molecular ML

---

**Need help?** Check the log files, review the generated analysis reports, or modify the search space based on preliminary results. 