#!/usr/bin/env python3
"""
Train Polymer GCN Models

Training script for Graph Convolutional Network models for polymer property prediction.
Uses PyTorch Geometric for graph neural network training.

Usage:
    python train_polymer_gcn.py --config configs/tg_gcn_baseline.yaml
    python train_polymer_gcn.py --config configs/tg_gcn_baseline.yaml --quick
    python train_polymer_gcn.py --config configs/tg_gcn_baseline.yaml --cv
"""

import argparse
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.polymer_dataset import PolymerTgDataset
from src.data.molecular_graph import MolecularGraphConverter
from src.models.polymer_gcn import PolymerGCN, create_gcn_model_from_config
from src.training.gcn_trainer import PolymerGCNTrainer


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_graph_datasets(config: Dict[str, Any], quick_mode: bool = False):
    """Create graph datasets for training."""
    logger = logging.getLogger(__name__)
    
    # Graph converter parameters
    graph_config = config['graph']
    converter_kwargs = {
        'max_atoms': graph_config['max_atoms'],
        'include_hydrogens': graph_config['include_hydrogens'],
        'use_chirality': graph_config['use_chirality'],
        'use_bond_types': graph_config['use_bond_types']
    }
    
    # Create datasets
    data_config = config['data']
    
    # Get polymer feature configuration only if enabled in model config
    model_config = config.get('model', {})
    use_polymer_features = model_config.get('use_polymer_features', False)
    polymer_feature_kwargs = config.get('polymer_features', {}) if use_polymer_features else None
    
    train_dataset = PolymerTgDataset(
        root='./data/processed',
        csv_file=data_config['dataset_path'],
        smiles_column=data_config['smiles_column'],
        target_column=data_config['target_column'],
        dp_column=data_config.get('dp_column'),
        mw_column=data_config.get('mw_column'),
        split_type='train',
        split_ratios=(
            data_config['splits']['train_ratio'],
            data_config['splits']['val_ratio'],
            data_config['splits']['test_ratio']
        ),
        graph_converter_kwargs=converter_kwargs,
        polymer_feature_kwargs=polymer_feature_kwargs,
        random_state=config['experiment']['random_seed']
    )
    
    val_dataset = PolymerTgDataset(
        root='./data/processed',
        csv_file=data_config['dataset_path'],
        smiles_column=data_config['smiles_column'],
        target_column=data_config['target_column'],
        dp_column=data_config.get('dp_column'),
        mw_column=data_config.get('mw_column'),
        split_type='val',
        split_ratios=(
            data_config['splits']['train_ratio'],
            data_config['splits']['val_ratio'],
            data_config['splits']['test_ratio']
        ),
        graph_converter_kwargs=converter_kwargs,
        polymer_feature_kwargs=polymer_feature_kwargs,
        random_state=config['experiment']['random_seed']
    )
    
    test_dataset = PolymerTgDataset(
        root='./data/processed',
        csv_file=data_config['dataset_path'],
        smiles_column=data_config['smiles_column'],
        target_column=data_config['target_column'],
        dp_column=data_config.get('dp_column'),
        mw_column=data_config.get('mw_column'),
        split_type='test',
        split_ratios=(
            data_config['splits']['train_ratio'],
            data_config['splits']['val_ratio'],
            data_config['splits']['test_ratio']
        ),
        graph_converter_kwargs=converter_kwargs,
        polymer_feature_kwargs=polymer_feature_kwargs,
        random_state=config['experiment']['random_seed']
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def apply_feature_scaling(train_dataset, val_dataset, test_dataset, config):
    """Apply feature scaling to molecular features if enabled."""
    logger = logging.getLogger(__name__)
    
    scaling_config = config['advanced']['feature_scaling']
    if not scaling_config.get('molecular_features', False):
        return train_dataset, val_dataset, test_dataset
    
    # Extract molecular features from training data
    train_mol_features = []
    for i in range(len(train_dataset)):
        data = train_dataset[i]
        if hasattr(data, 'mol_features'):
            train_mol_features.append(data.mol_features.numpy())
    
    if train_mol_features:
        train_mol_features = np.array(train_mol_features)
        
        # Fit scaler on training data
        scaler = StandardScaler()
        scaler.fit(train_mol_features)
        
        # Transform all datasets
        for dataset in [train_dataset, val_dataset, test_dataset]:
            for i in range(len(dataset)):
                data = dataset[i]
                if hasattr(data, 'mol_features'):
                    scaled_features = scaler.transform(data.mol_features.numpy().reshape(1, -1))
                    data.mol_features = torch.tensor(scaled_features.squeeze(), dtype=torch.float)
        
        logger.info("Applied StandardScaler to molecular features")
    
    return train_dataset, val_dataset, test_dataset


def run_single_training(config: Dict[str, Any],
                       train_dataset,
                       val_dataset, 
                       test_dataset,
                       quick_mode: bool = False):
    """Run a single training experiment."""
    logger = logging.getLogger(__name__)
    
    # Create model
    model_config = config['model']
    model = create_gcn_model_from_config(model_config, model_config['node_feature_dim'])
    
    # Create trainer
    trainer = PolymerGCNTrainer(
        model=model,
        device=config['training'].get('device', 'auto'),
        results_dir=config['results']['save_dir'],
        model_name=config['results']['model_name']
    )
    
    # Training parameters
    training_config = config['training']
    if quick_mode:
        quick_config = config.get('quick_mode', {})
        training_config = {**training_config}  # copy
        training_config['epochs'] = quick_config.get('epochs', 10)
        training_config['early_stopping']['patience'] = min(5, training_config['epochs'] // 2)
    
    # Debug: Check parameter types
    logger.info(f"DEBUG: weight_decay = {training_config['weight_decay']} (type: {type(training_config['weight_decay'])})")
    logger.info(f"DEBUG: learning_rate = {training_config['learning_rate']} (type: {type(training_config['learning_rate'])})")
    
    # Train model
    logger.info("Starting GCN training...")
    start_time = time.time()
    
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=int(training_config['batch_size']),
        max_epochs=int(training_config['epochs']),
        learning_rate=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay']),
        patience=int(training_config['early_stopping']['patience'])
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Test evaluation
    if test_dataset is not None:
        predictions, targets = trainer.predict(test_dataset, batch_size=32)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        test_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(targets, predictions))),
            'mae': float(mean_absolute_error(targets, predictions)),
            'r2': float(r2_score(targets, predictions))
        }
        results['test_metrics'] = test_metrics
        
        logger.info("Test Results:")
        logger.info(f"  R²: {test_metrics['r2']:.4f}")
        logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  MAE: {test_metrics['mae']:.4f}")
    
    # Check success criteria
    check_success_criteria(results, config)
    
    return results


def run_cross_validation(config: Dict[str, Any], quick_mode: bool = False):
    """Run cross-validation experiment."""
    logger = logging.getLogger(__name__)
    
    # Create full dataset
    data_config = config['data']
    graph_config = config['graph']
    converter_kwargs = {
        'max_atoms': graph_config['max_atoms'],
        'include_hydrogens': graph_config['include_hydrogens'],
        'use_chirality': graph_config['use_chirality'],
        'use_bond_types': graph_config['use_bond_types']
    }
    
    full_dataset = PolymerTgDataset(
        root='./data/processed',
        csv_file=data_config['dataset_path'],
        smiles_column=data_config['smiles_column'],
        target_column=data_config['target_column'],
        split_type='all',
        graph_converter_kwargs=converter_kwargs,
        random_state=config['experiment']['random_seed']
    )
    
    # Quick mode sampling
    if quick_mode:
        quick_config = config.get('quick_mode', {})
        max_samples = quick_config.get('max_samples', 100)
        if len(full_dataset) > max_samples:
            # Create smaller dataset
            indices = np.random.choice(len(full_dataset), max_samples, replace=False)
            subset_data = [full_dataset[i] for i in indices]
            full_dataset = subset_data
            logger.info(f"Quick mode: using {len(full_dataset)} samples")
    
    # Create model and trainer
    model_config = config['model']
    model = create_gcn_model_from_config(model_config, model_config['node_feature_dim'])
    trainer = PolymerGCNTrainer(
        model=model,
        device=config['training'].get('device', 'auto'),
        results_dir=config['results']['save_dir'],
        model_name=config['results']['model_name']
    )
    
    # Cross-validation parameters
    cv_config = config['cross_validation']
    training_config = config['training']
    
    if quick_mode:
        quick_config = config.get('quick_mode', {})
        cv_config = {**cv_config}  # copy
        cv_config['n_folds'] = quick_config.get('cv_folds', 3)
        cv_config['epochs'] = quick_config.get('epochs', 10)
    
    # Run cross-validation
    logger.info(f"Starting {cv_config['n_folds']}-fold cross validation...")
    start_time = time.time()
    
    cv_results = trainer.cross_validate(
        dataset=full_dataset,
        n_folds=cv_config['n_folds'],
        batch_size=training_config['batch_size'],
        max_epochs=cv_config.get('epochs', 50),
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        patience=cv_config.get('patience', 8)
    )
    
    cv_time = time.time() - start_time
    logger.info(f"Cross-validation completed in {cv_time:.2f} seconds")
    
    # Check success criteria
    check_success_criteria(cv_results, config, is_cv=True)
    
    return cv_results


def check_success_criteria(results: Dict[str, Any], config: Dict[str, Any], is_cv: bool = False):
    """Check if results meet success criteria."""
    logger = logging.getLogger(__name__)
    
    criteria = config['monitoring']['success_criteria']
    
    if is_cv:
        r2 = results['mean_r2']
        rmse = results['mean_rmse']
        mae = results['mean_mae']
        logger.info(f"\nSuccess Criteria Check (Cross-Validation):")
    else:
        metrics = results.get('test_metrics', results.get('final_metrics', {}))
        r2 = metrics.get('r2', 0)
        rmse = metrics.get('rmse', float('inf'))
        mae = metrics.get('mae', float('inf'))
        logger.info(f"\nSuccess Criteria Check:")
    
    # Check criteria
    meets_r2 = r2 >= criteria.get('min_r2', 0.5)
    meets_rmse = rmse <= criteria.get('max_rmse', 50.0)
    meets_mae = mae <= criteria.get('max_mae', 30.0)
    
    logger.info(f"  R² ≥ {criteria.get('min_r2', 0.5)}: {r2:.4f} {'✅' if meets_r2 else '❌'}")
    logger.info(f"  RMSE ≤ {criteria.get('max_rmse', 50.0)}: {rmse:.4f} {'✅' if meets_rmse else '❌'}")
    logger.info(f"  MAE ≤ {criteria.get('max_mae', 30.0)}: {mae:.4f} {'✅' if meets_mae else '❌'}")
    
    if meets_r2 and meets_rmse and meets_mae:
        logger.info("🎉 All success criteria met!")
    else:
        logger.info("⚠️  Some criteria not met - consider hyperparameter tuning")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train polymer GCN models")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--quick', action='store_true',
                       help='Run in quick mode for testing')
    parser.add_argument('--cv', action='store_true',
                       help='Run cross-validation instead of single training')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'],
                       help='Override device setting')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override quick mode
    if args.quick:
        config['quick_mode']['enabled'] = True
    
    # Override device
    if args.device:
        config['training']['device'] = args.device
    
    # Setup logging
    setup_logging(
        log_level=config['results']['log_level'],
        log_file=config['results'].get('log_file')
    )
    logger = logging.getLogger(__name__)
    
    # Set random seeds
    set_random_seeds(config['experiment']['random_seed'])
    
    logger.info(f"Starting GCN experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")
    if args.quick:
        logger.info("🚀 Running in QUICK MODE for testing")
    
    try:
        # Determine execution mode
        if args.cv or config['cross_validation']['enabled']:
            # Cross-validation mode
            logger.info("Running GCN cross-validation experiment")
            results = run_cross_validation(config, quick_mode=args.quick)
            
        else:
            # Single training mode
            logger.info("Running single GCN training experiment")
            
            # Create datasets
            train_dataset, val_dataset, test_dataset = create_graph_datasets(config, quick_mode=args.quick)
            
            # Apply feature scaling
            train_dataset, val_dataset, test_dataset = apply_feature_scaling(
                train_dataset, val_dataset, test_dataset, config
            )
            
            # Run training
            results = run_single_training(
                config, train_dataset, val_dataset, test_dataset, quick_mode=args.quick
            )
        
        logger.info("🎉 GCN experiment completed successfully!")
        logger.info(f"Results saved to: {config['results']['save_dir']}")
        
    except Exception as e:
        logger.error(f"GCN experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 