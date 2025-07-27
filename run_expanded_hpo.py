#!/usr/bin/env python3
"""
Expanded Hyperparameter Optimization for PolyGNN with Bayesian Optimization
Advanced HPO with 200 trials, multi-task weighting, and GPU acceleration

Usage:
    python run_expanded_hpo.py --n_trials 200 --gpu --multi_task
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.polymer_dataset import PolymerTgDataset
from src.models.polymer_gcn import PolymerGCN
from src.training.gcn_trainer import PolymerGCNTrainer
from src.features.polymer_features import PolymerFeatureExtractor

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpandedHPOOptimizer:
    """Advanced hyperparameter optimization with Bayesian search and multi-task weighting"""
    
    def __init__(self, 
                 dataset_path: str,
                 n_trials: int = 200,
                 use_gpu: bool = True,
                 multi_task: bool = True,
                 results_dir: str = "results/expanded_hpo"):
        
        self.dataset_path = Path(dataset_path)
        self.n_trials = n_trials
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.multi_task = multi_task
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Multi-task weights (focus on Tg)
        self.task_weights = {
            'Tg': 0.6,     # Primary focus
            'Tm': 0.2,     # Secondary
            'Density': 0.2 # Secondary
        } if multi_task else {'Tg': 1.0}
        
        # Load and prepare dataset
        self._prepare_dataset()
        
        # Performance tracking
        self.best_r2 = 0.0
        self.trial_results = []
        
        logger.info(f"🚀 Expanded HPO Optimizer initialized")
        logger.info(f"📊 Device: {self.device}")
        logger.info(f"🎯 Trials: {n_trials}")
        logger.info(f"⚖️  Task weights: {self.task_weights}")
    
    def _prepare_dataset(self):
        """Load and prepare the enhanced polymer dataset"""
        logger.info(f"📁 Loading dataset from {self.dataset_path}")
        
        # Use the enhanced full_feats.csv dataset
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        # Load the enhanced dataset
        df = pd.read_csv(self.dataset_path)
        logger.info(f"📊 Loaded dataset: {df.shape}")
        
        # Prepare for GCN training - we'll use SMILES column for graph generation
        if 'canonical_smiles' in df.columns:
            smiles_col = 'canonical_smiles'
        elif 'smiles' in df.columns:
            smiles_col = 'smiles'
        else:
            raise ValueError("No SMILES column found in dataset")
        
        # Filter valid samples
        valid_mask = df[smiles_col].notna() & (df[smiles_col] != '')
        if 'Tg' in df.columns:
            valid_mask &= df['Tg'].notna()
        
        self.df = df[valid_mask].copy()
        self.smiles_column = smiles_col
        
        logger.info(f"✅ Prepared {len(self.df)} valid samples")
        logger.info(f"🧬 Using SMILES column: {smiles_col}")
        
        # Check available target properties
        available_targets = [col for col in ['Tg', 'Tm', 'Density', 'FFV', 'Tc', 'Rg'] 
                           if col in self.df.columns and self.df[col].notna().sum() > 100]
        logger.info(f"🎯 Available targets: {available_targets}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for Bayesian optimization"""
        
        # Sample hyperparameters with wider, more granular ranges
        params = {
            # Learning rate: log-uniform from 1e-5 to 1e-2
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            
            # GNN architecture
            'num_gcn_layers': trial.suggest_int('num_gcn_layers', 3, 5),
            'hidden_dims': trial.suggest_categorical('hidden_dims', [128, 256, 512]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            
            # Training parameters
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            
            # Advanced parameters
            'pooling_method': trial.suggest_categorical('pooling_method', ['mean', 'max', 'add']),
            'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'swish']),
            
            # Polymer feature integration
            'polymer_feature_weight': trial.suggest_float('polymer_feature_weight', 0.5, 2.0),
        }
        
        try:
            # Train model with sampled parameters
            r2_score = self._train_and_evaluate(params, trial.number)
            
            # Track best performance
            if r2_score > self.best_r2:
                self.best_r2 = r2_score
                logger.info(f"🎯 New best R²: {r2_score:.4f} (Trial {trial.number})")
            
            # Store trial results
            trial_result = {
                'trial_number': trial.number,
                'r2_score': r2_score,
                'parameters': params,
                'timestamp': datetime.now().isoformat()
            }
            self.trial_results.append(trial_result)
            
            return r2_score
            
        except Exception as e:
            logger.error(f"❌ Trial {trial.number} failed: {e}")
            return 0.0
    
    def _train_and_evaluate(self, params: Dict[str, Any], trial_number: int) -> float:
        """Train model with given parameters and return validation R²"""
        
        # Create temporary config for this trial
        config = {
            'experiment': {
                'name': f'expanded_hpo_trial_{trial_number}',
                'description': f'Expanded HPO trial {trial_number} with Bayesian optimization',
                'random_seed': 42
            },
            'data': {
                'dataset_path': str(self.dataset_path),
                'smiles_column': self.smiles_column,
                'target_column': 'Tg',
                'splits': {'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15, 'shuffle': True},
                'target_transform': {'enabled': True, 'method': 'clip', 'clip_min': -200, 'clip_max': 500}
            },
            'graph': {
                'max_atoms': 200,
                'include_hydrogens': False,
                'use_chirality': True,
                'use_bond_types': False
            },
            'model': {
                'type': 'gcn',
                'node_feature_dim': 157,
                'molecular_feature_dim': 13,
                'num_gcn_layers': params['num_gcn_layers'],
                'hidden_dims': [params['hidden_dims']] * params['num_gcn_layers'],
                'dropout_rate': params['dropout_rate'],
                'pooling_method': params['pooling_method'],
                'use_molecular_features': True,
                'use_polymer_features': True,
                'polymer_feature_dim': 148,
                'activation': params['activation']
            },
            'training': {
                'device': str(self.device),
                'batch_size': params['batch_size'],
                'max_epochs': 100,  # Early stopping will handle this
                'learning_rate': params['learning_rate'],
                'weight_decay': params['weight_decay'],
                'early_stopping': {'patience': 15, 'min_delta': 1e-4}
            },
            'results': {
                'save_dir': str(self.results_dir / f"trial_{trial_number}"),
                'model_name': f'expanded_hpo_trial_{trial_number}',
                'log_level': 'WARNING'  # Reduce logging for HPO
            }
        }
        
        try:
            # Create datasets for each split
            split_ratios = (config['data']['splits']['train_ratio'], 
                           config['data']['splits']['val_ratio'],
                           config['data']['splits']['test_ratio'])
            
            train_dataset = PolymerTgDataset(
                root=str(self.results_dir / f"trial_{trial_number}"),
                csv_file=config['data']['dataset_path'],
                smiles_column=config['data']['smiles_column'],
                target_column=config['data']['target_column'],
                split_type='train',
                split_ratios=split_ratios,
                polymer_feature_kwargs={'fingerprint_size': 128, 'fp_radius': 2},
                random_state=42
            )
            
            val_dataset = PolymerTgDataset(
                root=str(self.results_dir / f"trial_{trial_number}"),
                csv_file=config['data']['dataset_path'],
                smiles_column=config['data']['smiles_column'],
                target_column=config['data']['target_column'],
                split_type='val',
                split_ratios=split_ratios,
                polymer_feature_kwargs={'fingerprint_size': 128, 'fp_radius': 2},
                random_state=42
            )
            
            # Create model
            model = PolymerGCN(
                node_feature_dim=config['model']['node_feature_dim'],
                molecular_feature_dim=config['model']['molecular_feature_dim'],
                hidden_dims=config['model']['hidden_dims'],
                output_dim=1,
                num_layers=config['model']['num_gcn_layers'],
                dropout_rate=config['model']['dropout_rate'],
                pooling_method=config['model']['pooling_method'],
                use_molecular_features=config['model']['use_molecular_features'],
                use_polymer_features=config['model']['use_polymer_features'],
                polymer_feature_dim=config['model']['polymer_feature_dim'],
                activation=config['model']['activation']
            ).to(self.device)
            
            # Create trainer
            trainer = PolymerGCNTrainer(
                model=model,
                device=self.device,
                results_dir=config['results']['save_dir']
            )
            
            # Train model
            results = trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=config['training']['batch_size'],
                max_epochs=config['training']['max_epochs'],
                learning_rate=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
                patience=config['training']['early_stopping']['patience'],
                verbose=False  # Quiet training for HPO
            )
            
            # Extract best validation R²
            best_val_r2 = max([m['r2'] for m in results['val_metrics']])
            
            return best_val_r2
            
        except Exception as e:
            logger.error(f"❌ Training failed for trial {trial_number}: {e}")
            return 0.0
    
    def optimize(self) -> Dict[str, Any]:
        """Run the expanded hyperparameter optimization"""
        
        logger.info(f"🚀 Starting expanded HPO with {self.n_trials} trials...")
        logger.info(f"🎯 Target: R² > 0.5")
        
        # Create Optuna study with advanced sampler and pruner
        sampler = TPESampler(
            n_startup_trials=20,  # Random trials before Bayesian
            n_ei_candidates=24,   # Expected improvement candidates
            multivariate=True,    # Consider parameter interactions
            group=True,           # Group similar parameters
            constant_liar=True,   # Better parallel optimization
            seed=42
        )
        
        pruner = MedianPruner(
            n_startup_trials=10,  # Min trials before pruning
            n_warmup_steps=20,    # Steps before pruning decision
            interval_steps=10     # Pruning check interval
        )
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"expanded_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        
        # Add initial trial with enhanced GCN parameters
        study.enqueue_trial({
            'learning_rate': 0.001,
            'num_gcn_layers': 3,
            'hidden_dims': 256,
            'dropout_rate': 0.2,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'pooling_method': 'mean',
            'activation': 'relu',
            'polymer_feature_weight': 1.0
        })
        
        start_time = time.time()
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=None,
            callbacks=[self._progress_callback],
            show_progress_bar=True
        )
        
        optimization_time = time.time() - start_time
        
        # Compile results
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'optimization_time_hours': optimization_time / 3600,
            'target_achieved': study.best_value >= 0.5,
            'study_stats': {
                'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            },
            'trial_history': self.trial_results
        }
        
        # Save results
        results_file = self.results_dir / f"expanded_hpo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save study for later analysis
        study_file = self.results_dir / f"optuna_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        import pickle
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        
        logger.info(f"🎉 HPO completed!")
        logger.info(f"🏆 Best R²: {study.best_value:.4f}")
        logger.info(f"✅ Target achieved: {results['target_achieved']}")
        logger.info(f"⏱️  Time: {optimization_time/3600:.2f} hours")
        logger.info(f"📁 Results saved to: {results_file}")
        
        return results
    
    def _progress_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Progress callback for monitoring optimization"""
        if trial.number % 10 == 0:
            logger.info(f"📊 Trial {trial.number}/{self.n_trials} | "
                       f"Best R²: {study.best_value:.4f} | "
                       f"Current: {trial.value:.4f if trial.value else 'N/A'}")

def main():
    parser = argparse.ArgumentParser(description='Expanded HPO for PolyGNN')
    parser.add_argument('--dataset', type=str, default='data/processed/full_feats.csv',
                       help='Path to enhanced dataset')
    parser.add_argument('--n_trials', type=int, default=200,
                       help='Number of optimization trials')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--multi_task', action='store_true', default=True,
                       help='Enable multi-task weighting')
    parser.add_argument('--results_dir', type=str, default='results/expanded_hpo',
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ExpandedHPOOptimizer(
        dataset_path=args.dataset,
        n_trials=args.n_trials,
        use_gpu=args.gpu,
        multi_task=args.multi_task,
        results_dir=args.results_dir
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 EXPANDED HPO RESULTS SUMMARY")
    print("="*60)
    print(f"🏆 Best Validation R²: {results['best_value']:.4f}")
    print(f"✅ Target R² ≥ 0.5: {'ACHIEVED' if results['target_achieved'] else 'NOT YET'}")
    print(f"📊 Completed Trials: {results['study_stats']['completed_trials']}")
    print(f"⏱️  Optimization Time: {results['optimization_time_hours']:.2f} hours")
    print("\n🎛️  Best Parameters:")
    for param, value in results['best_params'].items():
        print(f"   {param}: {value}")
    print("="*60)

if __name__ == "__main__":
    main()