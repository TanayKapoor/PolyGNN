#!/usr/bin/env python3
"""
Uncertainty Quantification Analysis for PolyGNN
Practical implementation using existing trained models and datasets

Usage:
    python run_uq_analysis.py --model_dir results/simple_hpo --ensemble_size 5
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.polymer_dataset import PolymerTgDataset
from src.models.polymer_gcn import PolymerGCN
from src.training.gcn_trainer import PolymerGCNTrainer

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PracticalUQ:
    """Practical uncertainty quantification using multiple model runs"""
    
    def __init__(self, 
                 dataset_path: str = "data/processed/full_feats.csv",
                 results_dir: str = "results/uq_analysis",
                 device: str = "auto"):
        
        self.dataset_path = Path(dataset_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"🎯 UQ Analysis initialized")
        logger.info(f"📊 Device: {self.device}")
        logger.info(f"📁 Results dir: {self.results_dir}")
    
    def create_ensemble_from_different_seeds(self, 
                                           ensemble_size: int = 5,
                                           config: Optional[Dict] = None) -> List[nn.Module]:
        """Create ensemble by training models with different random seeds"""
        
        if config is None:
            # Use best HPO parameters as default
            config = {
                'learning_rate': 0.001,
                'num_gcn_layers': 3,
                'hidden_dims': 256,
                'dropout_rate': 0.2,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'pooling_method': 'mean',
                'activation': 'relu'
            }
        
        logger.info(f"🏗️  Training ensemble of {ensemble_size} models with different seeds...")
        
        models = []
        
        for i in range(ensemble_size):
            seed = 42 + i * 17  # Different seeds for diversity
            logger.info(f"🎭 Training model {i+1}/{ensemble_size} with seed {seed}")
            
            # Set seed for reproducibility within each model
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create dataset with this seed
            train_dataset = PolymerTgDataset(
                root=str(self.results_dir / f"ensemble_model_{i}"),
                csv_file=str(self.dataset_path),
                smiles_column='canonical_smiles',
                target_column='Tg',
                split_type='train',
                split_ratios=(0.7, 0.15, 0.15),
                polymer_feature_kwargs={'fingerprint_size': 128, 'fp_radius': 2},
                random_state=seed
            )
            
            val_dataset = PolymerTgDataset(
                root=str(self.results_dir / f"ensemble_model_{i}"),
                csv_file=str(self.dataset_path),
                smiles_column='canonical_smiles',
                target_column='Tg',
                split_type='val',
                split_ratios=(0.7, 0.15, 0.15),
                polymer_feature_kwargs={'fingerprint_size': 128, 'fp_radius': 2},
                random_state=seed
            )
            
            # Create model
            model = PolymerGCN(
                node_feature_dim=157,
                molecular_feature_dim=13,
                hidden_dims=[config['hidden_dims']] * config['num_gcn_layers'],
                output_dim=1,
                num_layers=config['num_gcn_layers'],
                dropout_rate=config['dropout_rate'],
                pooling_method=config['pooling_method'],
                use_molecular_features=True,
                use_polymer_features=True,
                polymer_feature_dim=148,
                activation=config['activation']
            ).to(self.device)
            
            # Create trainer
            trainer = PolymerGCNTrainer(
                model=model,
                device=self.device,
                results_dir=str(self.results_dir / f"ensemble_model_{i}")
            )
            
            # Train model
            try:
                results = trainer.train(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    batch_size=config['batch_size'],
                    max_epochs=50,  # Reduced for speed
                    learning_rate=config['learning_rate'],
                    weight_decay=config['weight_decay'],
                    patience=10,
                    verbose=False
                )
                
                models.append(model)
                logger.info(f"✅ Model {i+1} trained successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to train model {i+1}: {e}")
                continue
        
        logger.info(f"🎭 Successfully trained {len(models)} ensemble models")
        return models
    
    def ensemble_predict(self, 
                        models: List[nn.Module], 
                        dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ensemble predictions with uncertainty estimates"""
        
        all_predictions = []
        targets = None
        
        # Collect predictions from all models
        for i, model in enumerate(models):
            model.eval()
            model_preds = []
            model_targets = []
            
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device)
                    pred = model(batch)
                    model_preds.append(pred.cpu().numpy())
                    
                    # Only collect targets once
                    if i == 0:
                        model_targets.append(batch.y.cpu().numpy())
            
            predictions = np.concatenate(model_preds, axis=0)
            all_predictions.append(predictions)
            
            if i == 0:
                targets = np.concatenate(model_targets, axis=0)
        
        # Convert to numpy array: (n_models, n_samples)
        predictions = np.array(all_predictions)
        
        # Calculate ensemble statistics
        mean_pred = np.mean(predictions, axis=0).flatten()
        std_pred = np.std(predictions, axis=0).flatten()
        targets = targets.flatten()
        
        return mean_pred, std_pred, targets
    
    def monte_carlo_dropout_predict(self, 
                                  model: nn.Module, 
                                  dataloader: DataLoader, 
                                  n_samples: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Monte Carlo Dropout prediction"""
        
        logger.info(f"🎲 Running MC Dropout with {n_samples} samples...")
        
        # Enable dropout during inference
        model.train()
        
        all_predictions = []
        targets = None
        
        with torch.no_grad():
            for _ in range(n_samples):
                mc_preds = []
                mc_targets = []
                
                for batch in dataloader:
                    batch = batch.to(self.device)
                    pred = model(batch)
                    mc_preds.append(pred.cpu().numpy())
                    
                    # Only collect targets once
                    if targets is None:
                        mc_targets.append(batch.y.cpu().numpy())
                
                predictions = np.concatenate(mc_preds, axis=0)
                all_predictions.append(predictions)
                
                if targets is None:
                    targets = np.concatenate(mc_targets, axis=0)
        
        # Calculate MC statistics
        predictions = np.array(all_predictions)
        mean_pred = np.mean(predictions, axis=0).flatten()
        std_pred = np.std(predictions, axis=0).flatten()
        targets = targets.flatten()
        
        return mean_pred, std_pred, targets
    
    def analyze_calibration(self, 
                           predictions: np.ndarray,
                           uncertainties: np.ndarray,
                           targets: np.ndarray) -> Dict:
        """Analyze prediction calibration"""
        
        logger.info("🎯 Analyzing uncertainty calibration...")
        
        # Calculate absolute errors
        errors = np.abs(predictions - targets)
        
        # Error-uncertainty correlation
        pearson_corr = stats.pearsonr(errors, uncertainties)[0]
        spearman_corr = stats.spearmanr(errors, uncertainties)[0]
        
        # Coverage analysis for different confidence levels
        confidence_levels = [0.5, 0.68, 0.95, 0.99]
        coverage_results = []
        
        for conf_level in confidence_levels:
            # Calculate z-score for confidence level
            z_score = stats.norm.ppf(0.5 + conf_level/2)
            
            # Check if targets fall within confidence interval
            lower_bound = predictions - z_score * uncertainties
            upper_bound = predictions + z_score * uncertainties
            
            coverage = np.mean((targets >= lower_bound) & (targets <= upper_bound))
            coverage_results.append(coverage)
        
        # Outlier detection based on high uncertainty
        uncertainty_threshold = np.percentile(uncertainties, 95)  # Top 5%
        high_uncertainty_mask = uncertainties > uncertainty_threshold
        
        results = {
            'error_correlation': {
                'pearson': pearson_corr,
                'spearman': spearman_corr
            },
            'coverage': {
                'confidence_levels': confidence_levels,
                'actual_coverage': coverage_results,
                'expected_coverage': confidence_levels
            },
            'outlier_analysis': {
                'threshold': uncertainty_threshold,
                'n_outliers': np.sum(high_uncertainty_mask),
                'outlier_fraction': np.mean(high_uncertainty_mask),
                'outlier_mae': np.mean(errors[high_uncertainty_mask]) if np.any(high_uncertainty_mask) else 0,
                'normal_mae': np.mean(errors[~high_uncertainty_mask]) if np.any(~high_uncertainty_mask) else 0
            }
        }\n        \n        return results\n    \n    def plot_uq_analysis(self, \n                        predictions: np.ndarray,\n                        uncertainties: np.ndarray,\n                        targets: np.ndarray,\n                        calibration_results: Dict,\n                        method_name: str = \"Ensemble\"):\n        \"\"\"Create comprehensive UQ visualization\"\"\"\n        \n        fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n        \n        # 1. Predictions vs Targets with uncertainty\n        ax1 = axes[0, 0]\n        scatter = ax1.scatter(targets, predictions, c=uncertainties, \n                            cmap='viridis', alpha=0.6, s=30)\n        ax1.plot([targets.min(), targets.max()], [targets.min(), targets.max()], \n                'r--', alpha=0.8, label='Perfect Prediction')\n        ax1.set_xlabel('True Tg (°C)')\n        ax1.set_ylabel('Predicted Tg (°C)')\n        ax1.set_title(f'{method_name}: Predictions vs Targets\\n(Color = Uncertainty)')\n        ax1.legend()\n        plt.colorbar(scatter, ax=ax1, label='Uncertainty')\n        \n        # Add R² to plot\n        r2 = r2_score(targets, predictions)\n        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, \n                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n        \n        # 2. Error vs Uncertainty correlation\n        ax2 = axes[0, 1]\n        errors = np.abs(predictions - targets)\n        ax2.scatter(uncertainties, errors, alpha=0.6, s=30)\n        \n        # Add correlation line\n        z = np.polyfit(uncertainties, errors, 1)\n        p = np.poly1d(z)\n        ax2.plot(uncertainties, p(uncertainties), \"r--\", alpha=0.8)\n        \n        ax2.set_xlabel('Predicted Uncertainty')\n        ax2.set_ylabel('Absolute Error (°C)')\n        ax2.set_title(f'Error-Uncertainty Correlation\\nPearson: {calibration_results[\"error_correlation\"][\"pearson\"]:.3f}')\n        \n        # 3. Coverage plot\n        ax3 = axes[1, 0]\n        expected = calibration_results['coverage']['confidence_levels']\n        actual = calibration_results['coverage']['actual_coverage']\n        \n        ax3.plot(expected, actual, 'bo-', markersize=8, label='Actual Coverage')\n        ax3.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Calibration')\n        ax3.set_xlabel('Expected Coverage')\n        ax3.set_ylabel('Actual Coverage')\n        ax3.set_title('Calibration Plot')\n        ax3.legend()\n        ax3.grid(True, alpha=0.3)\n        \n        # 4. Uncertainty distribution\n        ax4 = axes[1, 1]\n        \n        # Histogram of uncertainties\n        ax4.hist(uncertainties, bins=30, alpha=0.7, color='skyblue', edgecolor='black')\n        ax4.axvline(np.mean(uncertainties), color='red', linestyle='--', \n                   label=f'Mean: {np.mean(uncertainties):.2f}')\n        ax4.axvline(np.percentile(uncertainties, 95), color='orange', linestyle='--',\n                   label=f'95th percentile: {np.percentile(uncertainties, 95):.2f}')\n        ax4.set_xlabel('Uncertainty')\n        ax4.set_ylabel('Frequency')\n        ax4.set_title('Uncertainty Distribution')\n        ax4.legend()\n        \n        plt.tight_layout()\n        \n        # Save plot\n        plot_path = self.results_dir / f\"uq_analysis_{method_name.lower()}.png\"\n        plt.savefig(plot_path, dpi=300, bbox_inches='tight')\n        logger.info(f\"📊 UQ analysis plot saved to {plot_path}\")\n        plt.show()\n    \n    def run_complete_uq_analysis(self, ensemble_size: int = 5) -> Dict:\n        \"\"\"Run complete UQ analysis pipeline\"\"\"\n        \n        logger.info(\"🚀 Starting Complete UQ Analysis Pipeline\")\n        logger.info(\"=\"*60)\n        \n        # Step 1: Create test dataset\n        logger.info(\"📊 Creating test dataset...\")\n        test_dataset = PolymerTgDataset(\n            root=str(self.results_dir / \"test_dataset\"),\n            csv_file=str(self.dataset_path),\n            smiles_column='canonical_smiles',\n            target_column='Tg',\n            split_type='test',\n            split_ratios=(0.7, 0.15, 0.15),\n            polymer_feature_kwargs={'fingerprint_size': 128, 'fp_radius': 2},\n            random_state=42\n        )\n        \n        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)\n        logger.info(f\"✅ Test dataset created with {len(test_dataset)} samples\")\n        \n        # Step 2: Create ensemble models\n        logger.info(\"🎭 Creating ensemble models...\")\n        ensemble_models = self.create_ensemble_from_different_seeds(ensemble_size)\n        \n        if len(ensemble_models) == 0:\n            raise ValueError(\"Failed to train any ensemble models\")\n        \n        # Step 3: Ensemble UQ Analysis\n        logger.info(\"🎯 Running Ensemble UQ Analysis...\")\n        ens_preds, ens_uncertainty, targets = self.ensemble_predict(ensemble_models, test_loader)\n        ens_calibration = self.analyze_calibration(ens_preds, ens_uncertainty, targets)\n        \n        # Step 4: Monte Carlo Dropout Analysis (using first model)\n        logger.info(\"🎲 Running Monte Carlo Dropout Analysis...\")\n        mc_preds, mc_uncertainty, _ = self.monte_carlo_dropout_predict(\n            ensemble_models[0], test_loader, n_samples=20\n        )\n        mc_calibration = self.analyze_calibration(mc_preds, mc_uncertainty, targets)\n        \n        # Step 5: Create visualizations\n        logger.info(\"📊 Creating UQ visualizations...\")\n        self.plot_uq_analysis(ens_preds, ens_uncertainty, targets, ens_calibration, \"Ensemble\")\n        self.plot_uq_analysis(mc_preds, mc_uncertainty, targets, mc_calibration, \"MC_Dropout\")\n        \n        # Step 6: Compile results\n        results = {\n            'ensemble_analysis': {\n                'n_models': len(ensemble_models),\n                'performance': {\n                    'r2': r2_score(targets, ens_preds),\n                    'rmse': np.sqrt(mean_squared_error(targets, ens_preds)),\n                    'mae': mean_absolute_error(targets, ens_preds)\n                },\n                'calibration': ens_calibration\n            },\n            'mc_dropout_analysis': {\n                'n_samples': 20,\n                'performance': {\n                    'r2': r2_score(targets, mc_preds),\n                    'rmse': np.sqrt(mean_squared_error(targets, mc_preds)),\n                    'mae': mean_absolute_error(targets, mc_preds)\n                },\n                'calibration': mc_calibration\n            },\n            'comparison': {\n                'ensemble_pearson_corr': ens_calibration['error_correlation']['pearson'],\n                'mc_dropout_pearson_corr': mc_calibration['error_correlation']['pearson'],\n                'ensemble_95_coverage': ens_calibration['coverage']['actual_coverage'][-2],\n                'mc_dropout_95_coverage': mc_calibration['coverage']['actual_coverage'][-2]\n            }\n        }\n        \n        # Save results\n        results_file = self.results_dir / \"uq_analysis_results.json\"\n        with open(results_file, 'w') as f:\n            json.dump(results, f, indent=2, default=str)\n        \n        logger.info(\"🎉 UQ Analysis Complete!\")\n        logger.info(f\"📁 Results saved to {results_file}\")\n        \n        # Print summary\n        self._print_uq_summary(results)\n        \n        return results\n    \n    def _print_uq_summary(self, results: Dict):\n        \"\"\"Print UQ analysis summary\"\"\"\n        \n        print(\"\\n\" + \"=\"*70)\n        print(\"🎯 UNCERTAINTY QUANTIFICATION ANALYSIS SUMMARY\")\n        print(\"=\"*70)\n        \n        ens = results['ensemble_analysis']\n        mc = results['mc_dropout_analysis']\n        comp = results['comparison']\n        \n        print(f\"\\n🎭 ENSEMBLE ANALYSIS ({ens['n_models']} models):\")\n        print(f\"   R²: {ens['performance']['r2']:.4f}\")\n        print(f\"   RMSE: {ens['performance']['rmse']:.2f}°C\")\n        print(f\"   Error-Uncertainty Correlation: {comp['ensemble_pearson_corr']:.3f}\")\n        print(f\"   95% Coverage: {comp['ensemble_95_coverage']:.3f} (Target: 0.95)\")\n        \n        print(f\"\\n🎲 MONTE CARLO DROPOUT ANALYSIS ({mc['n_samples']} samples):\")\n        print(f\"   R²: {mc['performance']['r2']:.4f}\")\n        print(f\"   RMSE: {mc['performance']['rmse']:.2f}°C\")\n        print(f\"   Error-Uncertainty Correlation: {comp['mc_dropout_pearson_corr']:.3f}\")\n        print(f\"   95% Coverage: {comp['mc_dropout_95_coverage']:.3f} (Target: 0.95)\")\n        \n        print(f\"\\n📊 UQ QUALITY ASSESSMENT:\")\n        ens_corr = comp['ensemble_pearson_corr']\n        mc_corr = comp['mc_dropout_pearson_corr']\n        \n        print(f\"   Error-Uncertainty Correlation Target: > 0.7\")\n        print(f\"   Ensemble: {'✅ GOOD' if ens_corr > 0.7 else '⚠️  MODERATE' if ens_corr > 0.5 else '❌ POOR'}\")\n        print(f\"   MC Dropout: {'✅ GOOD' if mc_corr > 0.7 else '⚠️  MODERATE' if mc_corr > 0.5 else '❌ POOR'}\")\n        \n        ens_cov = comp['ensemble_95_coverage']\n        mc_cov = comp['mc_dropout_95_coverage']\n        \n        print(f\"\\n   95% Coverage Target: ~0.95\")\n        print(f\"   Ensemble: {'✅ GOOD' if 0.90 <= ens_cov <= 0.98 else '⚠️  MODERATE'}\")\n        print(f\"   MC Dropout: {'✅ GOOD' if 0.90 <= mc_cov <= 0.98 else '⚠️  MODERATE'}\")\n        \n        print(\"\\n🚀 Ready for Week 3: External validation with robust UQ!\")\n        print(\"=\"*70 + \"\\n\")\n\ndef main():\n    parser = argparse.ArgumentParser(description='UQ Analysis for PolyGNN')\n    parser.add_argument('--dataset', type=str, default='data/processed/full_feats.csv',\n                       help='Path to dataset')\n    parser.add_argument('--ensemble_size', type=int, default=5,\n                       help='Number of models in ensemble')\n    parser.add_argument('--results_dir', type=str, default='results/uq_analysis',\n                       help='Results directory')\n    parser.add_argument('--device', type=str, default='auto',\n                       help='Device to use (auto, cpu, cuda)')\n    \n    args = parser.parse_args()\n    \n    # Validate dataset\n    if not Path(args.dataset).exists():\n        print(f\"❌ Dataset not found: {args.dataset}\")\n        sys.exit(1)\n    \n    # Create UQ analyzer\n    uq_analyzer = PracticalUQ(\n        dataset_path=args.dataset,\n        results_dir=args.results_dir,\n        device=args.device\n    )\n    \n    # Run complete analysis\n    results = uq_analyzer.run_complete_uq_analysis(args.ensemble_size)\n    \n    return results\n\nif __name__ == \"__main__\":\n    main()