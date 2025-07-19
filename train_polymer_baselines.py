#!/usr/bin/env python3
"""
Train Polymer Fingerprint Baseline Models

Main training script for polymer property prediction using molecular fingerprints
and polymer-specific chain features. Supports both single training runs and 
cross-validation experiments.

Usage:
    python train_polymer_baselines.py --config configs/tg_fingerprint_baseline.yaml
    python train_polymer_baselines.py --config configs/tg_fingerprint_baseline.yaml --quick
    python train_polymer_baselines.py --config configs/tg_fingerprint_baseline.yaml --cv
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

from src.models.polymer_baseline import (
    PolymerFingerprintBaseline, 
    PolymerFeatureExtractor, 
    PolymerFingerprintDataset,
    create_baseline_model_and_extractor
)
from src.training.trainer import PolymerBaselineTrainer


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


def load_and_preprocess_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    logger = logging.getLogger(__name__)
    
    # Load dataset
    data_config = config['data']
    df = pd.read_csv(data_config['dataset_path'])
    logger.info(f"Loaded {len(df)} samples from {data_config['dataset_path']}")
    
    # Basic validation
    smiles_col = data_config['smiles_column']
    target_col = data_config['target_column']
    
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in dataset")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Remove missing values
    initial_len = len(df)
    df = df.dropna(subset=[smiles_col, target_col])
    logger.info(f"Removed {initial_len - len(df)} samples with missing values")
    
    # Target transformation
    target_transform = data_config.get('target_transform', {})
    if target_transform.get('enabled', False):
        if target_transform['method'] == 'clip':
            clip_min = target_transform.get('clip_min', -np.inf)
            clip_max = target_transform.get('clip_max', np.inf)
            original_range = (df[target_col].min(), df[target_col].max())
            df[target_col] = df[target_col].clip(lower=clip_min, upper=clip_max)
            new_range = (df[target_col].min(), df[target_col].max())
            logger.info(f"Clipped target values: {original_range} -> {new_range}")
    
    # Data quality summary
    logger.info(f"Final dataset: {len(df)} samples")
    logger.info(f"Target range: {df[target_col].min():.2f} to {df[target_col].max():.2f}")
    logger.info(f"SMILES length range: {df[smiles_col].str.len().min()} to {df[smiles_col].str.len().max()}")
    
    return df


def create_datasets(df: pd.DataFrame, 
                   config: Dict[str, Any], 
                   feature_extractor: PolymerFeatureExtractor,
                   quick_mode: bool = False):
    """Create train/val/test datasets."""
    logger = logging.getLogger(__name__)
    
    # Quick mode sampling
    if quick_mode:
        quick_config = config.get('quick_mode', {})
        max_samples = quick_config.get('max_samples', 100)
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            logger.info(f"Quick mode: using {len(df)} samples")
    
    # Extract data
    data_config = config['data']
    smiles_list = df[data_config['smiles_column']].tolist()
    targets = df[data_config['target_column']].tolist()
    
    # Create splits
    splits_config = data_config['splits']
    train_ratio = splits_config['train_ratio']
    val_ratio = splits_config['val_ratio']
    test_ratio = splits_config['test_ratio']
    
    # First split: train+val vs test
    train_val_smiles, test_smiles, train_val_targets, test_targets = train_test_split(
        smiles_list, targets,
        test_size=test_ratio,
        shuffle=splits_config['shuffle'],
        random_state=42
    )
    
    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_smiles, val_smiles, train_targets, val_targets = train_test_split(
        train_val_smiles, train_val_targets,
        test_size=val_size,
        shuffle=splits_config['shuffle'],
        random_state=42
    )
    
    logger.info(f"Data splits: {len(train_smiles)} train, {len(val_smiles)} val, {len(test_smiles)} test")
    
    # Create datasets
    train_dataset = PolymerFingerprintDataset(train_smiles, train_targets, feature_extractor)
    val_dataset = PolymerFingerprintDataset(val_smiles, val_targets, feature_extractor)
    test_dataset = PolymerFingerprintDataset(test_smiles, test_targets, feature_extractor)
    
    return train_dataset, val_dataset, test_dataset


def apply_feature_scaling(datasets, config):
    """Apply feature scaling to chain features if enabled."""
    logger = logging.getLogger(__name__)
    
    scaling_config = config['advanced']['feature_scaling']
    if not scaling_config.get('chain_features', False):
        return datasets
    
    train_dataset, val_dataset, test_dataset = datasets
    
    # Fit scaler on training data
    scaler = StandardScaler()
    train_chain_features = train_dataset.chain_features
    scaler.fit(train_chain_features)
    
    # Transform all datasets
    train_dataset.chain_features = scaler.transform(train_chain_features)
    val_dataset.chain_features = scaler.transform(val_dataset.chain_features)
    test_dataset.chain_features = scaler.transform(test_dataset.chain_features)
    
    logger.info("Applied StandardScaler to chain features")
    return datasets


def run_single_training(config: Dict[str, Any], 
                       train_dataset, 
                       val_dataset, 
                       test_dataset,
                       quick_mode: bool = False):
    """Run a single training experiment."""
    logger = logging.getLogger(__name__)
    
    # Create model and trainer
    model, _ = create_baseline_model_and_extractor(config['model'])
    trainer = PolymerBaselineTrainer(
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
        training_config['epochs'] = quick_config.get('epochs', 5)
        training_config['early_stopping']['patience'] = min(3, training_config['epochs'] // 2)
    
    # Train model
    logger.info("Starting single training run...")
    start_time = time.time()
    
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=training_config['batch_size'],
        epochs=training_config['epochs'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        patience=training_config['early_stopping']['patience']
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Test evaluation
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_metrics = trainer.validate_epoch(test_loader, torch.nn.MSELoss())
        results['test_metrics'] = test_metrics
        
        logger.info("Test Results:")
        logger.info(f"  R²: {test_metrics['r2']:.4f}")
        logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  MAE: {test_metrics['mae']:.4f}")
    
    # Check success criteria
    check_success_criteria(results, config)
    
    return results


def run_cross_validation(config: Dict[str, Any], 
                        full_dataset,
                        quick_mode: bool = False):
    """Run cross-validation experiment."""
    logger = logging.getLogger(__name__)
    
    # Create model and trainer
    model, _ = create_baseline_model_and_extractor(config['model'])
    trainer = PolymerBaselineTrainer(
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
        cv_config['n_folds'] = quick_config.get('cv_folds', 2)
        training_config = {**training_config}  # copy  
        training_config['epochs'] = cv_config.get('epochs', 20)
    
    # Run cross-validation
    logger.info(f"Starting {cv_config['n_folds']}-fold cross validation...")
    start_time = time.time()
    
    cv_results = trainer.cross_validate(
        dataset=full_dataset,
        n_folds=cv_config['n_folds'],
        batch_size=training_config['batch_size'],
        epochs=training_config['epochs'],
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
    parser = argparse.ArgumentParser(description="Train polymer fingerprint baseline models")
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
    
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")
    if args.quick:
        logger.info("🚀 Running in QUICK MODE for testing")
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(config)
        
        # Create feature extractor
        _, feature_extractor = create_baseline_model_and_extractor(config['model'])
        
        # Determine execution mode
        if args.cv or config['cross_validation']['enabled']:
            # Cross-validation mode
            logger.info("Running cross-validation experiment")
            
            # Create full dataset for CV
            smiles_list = df[config['data']['smiles_column']].tolist()
            targets = df[config['data']['target_column']].tolist()
            
            if args.quick:
                quick_config = config.get('quick_mode', {})
                max_samples = quick_config.get('max_samples', 100)
                if len(smiles_list) > max_samples:
                    indices = np.random.choice(len(smiles_list), max_samples, replace=False)
                    smiles_list = [smiles_list[i] for i in indices]
                    targets = [targets[i] for i in indices]
            
            full_dataset = PolymerFingerprintDataset(smiles_list, targets, feature_extractor)
            
            # Apply scaling if needed
            if config['advanced']['feature_scaling'].get('chain_features', False):
                scaler = StandardScaler()
                full_dataset.chain_features = scaler.fit_transform(full_dataset.chain_features)
                logger.info("Applied StandardScaler to chain features")
            
            # Run cross-validation
            results = run_cross_validation(config, full_dataset, quick_mode=args.quick)
            
        else:
            # Single training mode
            logger.info("Running single training experiment")
            
            # Create datasets
            train_dataset, val_dataset, test_dataset = create_datasets(
                df, config, feature_extractor, quick_mode=args.quick
            )
            
            # Apply feature scaling
            datasets = apply_feature_scaling((train_dataset, val_dataset, test_dataset), config)
            train_dataset, val_dataset, test_dataset = datasets
            
            # Run training
            results = run_single_training(
                config, train_dataset, val_dataset, test_dataset, quick_mode=args.quick
            )
        
        logger.info("🎉 Experiment completed successfully!")
        logger.info(f"Results saved to: {config['results']['save_dir']}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 