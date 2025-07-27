"""
Uncertainty Quantification for Polymer GNN Models
Implements ensemble methods and Monte Carlo Dropout for robust predictions

Features:
- Ensemble averaging with variance estimation
- Monte Carlo Dropout for inference-time uncertainty
- Calibration metrics and error correlation analysis
- High-variance outlier detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class MCDropoutGCN(nn.Module):
    """GCN with Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, base_model: nn.Module, dropout_rate: float = 0.2):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.mc_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, batch, mc_samples: int = 1):
        """Forward pass with optional MC sampling"""
        if mc_samples == 1:
            return self.base_model(batch)
        
        # Monte Carlo sampling
        predictions = []
        self.train()  # Enable dropout
        
        with torch.no_grad():
            for _ in range(mc_samples):
                pred = self.base_model(batch)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        var_pred = np.var(predictions, axis=0)
        
        return torch.tensor(mean_pred), torch.tensor(var_pred)

class EnsembleGCN:
    """Ensemble of GCN models for uncertainty quantification"""
    
    def __init__(self, models: List[nn.Module], device: torch.device):
        self.models = models
        self.device = device
        self.n_models = len(models)
        
        # Set all models to eval mode
        for model in self.models:
            model.eval()
    
    def predict(self, dataloader, return_variance: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Ensemble prediction with uncertainty estimation"""
        all_predictions = []
        all_targets = []
        
        # Collect predictions from all models
        for i, model in enumerate(self.models):
            model_preds = []
            targets = []
            
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device)
                    pred = model(batch)
                    model_preds.append(pred.cpu().numpy())
                    
                    # Only collect targets once
                    if i == 0:
                        if hasattr(batch, 'y'):
                            targets.append(batch.y.cpu().numpy())
                        elif hasattr(batch, 'target'):
                            targets.append(batch.target.cpu().numpy())
            
            model_preds = np.concatenate(model_preds, axis=0)
            all_predictions.append(model_preds)
            
            if i == 0:
                all_targets = np.concatenate(targets, axis=0)
        
        # Convert to numpy array: (n_models, n_samples, n_outputs)
        predictions = np.array(all_predictions)
        
        # Calculate ensemble statistics
        mean_pred = np.mean(predictions, axis=0)
        
        if return_variance:
            var_pred = np.var(predictions, axis=0)
            return mean_pred, var_pred, all_targets
        
        return mean_pred, None, all_targets
    
    def predict_single(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single batch prediction with uncertainty"""
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(batch)
                predictions.append(pred.cpu())
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        
        return mean_pred, var_pred

class UncertaintyCalibrator:
    """Calibration analysis for uncertainty estimates"""
    
    def __init__(self):
        self.calibration_results = {}
    
    def calibrate(self, 
                  predictions: np.ndarray,
                  uncertainties: np.ndarray,
                  targets: np.ndarray,
                  confidence_levels: List[float] = [0.5, 0.68, 0.95, 0.99]) -> Dict:
        """Analyze prediction calibration"""
        
        results = {
            'confidence_levels': confidence_levels,
            'coverage': [],
            'expected_coverage': [],
            'miscalibration': [],
            'error_correlation': {},
            'outlier_detection': {}
        }
        
        # Calculate absolute errors
        errors = np.abs(predictions - targets)
        
        # Error-uncertainty correlation
        if len(uncertainties.shape) > 1:
            uncertainties = np.sqrt(np.sum(uncertainties, axis=1))  # Total uncertainty
        
        correlation = stats.pearsonr(errors.flatten(), uncertainties.flatten())[0]
        results['error_correlation']['pearson'] = correlation
        results['error_correlation']['spearman'] = stats.spearmanr(errors.flatten(), uncertainties.flatten())[0]
        
        # Coverage analysis
        std_uncertainties = np.sqrt(uncertainties)
        
        for conf_level in confidence_levels:
            # Calculate z-score for confidence level
            z_score = stats.norm.ppf(0.5 + conf_level/2)
            
            # Check if targets fall within confidence interval
            lower_bound = predictions - z_score * std_uncertainties
            upper_bound = predictions + z_score * std_uncertainties
            
            if len(targets.shape) > 1:
                # Multi-output case
                coverage = np.mean(
                    (targets >= lower_bound) & (targets <= upper_bound)
                )
            else:
                coverage = np.mean(
                    (targets >= lower_bound.flatten()) & (targets <= upper_bound.flatten())
                )
            
            results['coverage'].append(coverage)
            results['expected_coverage'].append(conf_level)
            results['miscalibration'].append(abs(coverage - conf_level))
        
        # High uncertainty outlier detection
        uncertainty_threshold = np.percentile(uncertainties, 95)  # Top 5% uncertain
        high_uncertainty_mask = uncertainties > uncertainty_threshold
        
        results['outlier_detection'] = {
            'threshold': uncertainty_threshold,
            'n_outliers': np.sum(high_uncertainty_mask),
            'outlier_fraction': np.mean(high_uncertainty_mask),
            'outlier_mae': np.mean(errors[high_uncertainty_mask]) if np.any(high_uncertainty_mask) else 0,
            'normal_mae': np.mean(errors[~high_uncertainty_mask]) if np.any(~high_uncertainty_mask) else 0
        }
        
        self.calibration_results = results
        return results
    
    def plot_calibration(self, save_path: Optional[Path] = None):
        """Plot calibration analysis"""
        if not self.calibration_results:
            logger.warning("No calibration results to plot. Run calibrate() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Coverage vs Expected Coverage
        ax1 = axes[0, 0]
        expected = self.calibration_results['expected_coverage']
        actual = self.calibration_results['coverage']
        
        ax1.plot(expected, actual, 'bo-', label='Actual Coverage')
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax1.set_xlabel('Expected Coverage')
        ax1.set_ylabel('Actual Coverage')
        ax1.set_title('Calibration Plot')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Miscalibration
        ax2 = axes[0, 1]
        miscal = self.calibration_results['miscalibration']
        ax2.bar(range(len(miscal)), miscal, color='red', alpha=0.7)
        ax2.set_xlabel('Confidence Level')
        ax2.set_ylabel('Miscalibration')
        ax2.set_title('Miscalibration by Confidence Level')
        ax2.set_xticks(range(len(expected)))
        ax2.set_xticklabels([f'{x:.2f}' for x in expected])
        
        # 3. Error-Uncertainty Correlation
        ax3 = axes[1, 0]
        corr_pearson = self.calibration_results['error_correlation']['pearson']
        corr_spearman = self.calibration_results['error_correlation']['spearman']
        
        correlations = [corr_pearson, corr_spearman]
        labels = ['Pearson', 'Spearman']
        colors = ['blue', 'green']
        
        bars = ax3.bar(labels, correlations, color=colors, alpha=0.7)
        ax3.set_ylabel('Correlation Coefficient')
        ax3.set_title('Error-Uncertainty Correlation')
        ax3.axhline(y=0.7, color='red', linestyle='--', label='Target (0.7)')
        ax3.legend()
        
        # Add correlation values on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom')
        
        # 4. Outlier Analysis
        ax4 = axes[1, 1]
        outlier_data = self.calibration_results['outlier_detection']
        
        mae_values = [outlier_data['outlier_mae'], outlier_data['normal_mae']]
        labels = ['High Uncertainty\n(Outliers)', 'Normal Uncertainty']
        colors = ['red', 'blue']
        
        bars = ax4.bar(labels, mae_values, color=colors, alpha=0.7)
        ax4.set_ylabel('Mean Absolute Error')
        ax4.set_title(f"Outlier Detection\n({outlier_data['n_outliers']} outliers, "
                     f"{outlier_data['outlier_fraction']:.1%} of data)")
        
        # Add MAE values on bars
        for bar, mae in zip(bars, mae_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{mae:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Calibration plot saved to {save_path}")
        
        plt.show()

class UncertaintyQuantifier:
    """Main UQ class combining ensemble and MC dropout methods"""
    
    def __init__(self, 
                 models: List[nn.Module],
                 device: torch.device,
                 mc_samples: int = 20):
        
        self.ensemble = EnsembleGCN(models, device)
        self.device = device
        self.mc_samples = mc_samples
        self.calibrator = UncertaintyCalibrator()
        
        logger.info(f"🎯 UQ initialized with {len(models)} ensemble models")
        logger.info(f"🎲 MC samples: {mc_samples}")
    
    def predict_with_uncertainty(self, 
                                dataloader,
                                method: str = 'ensemble') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Main prediction method with uncertainty quantification"""
        
        if method == 'ensemble':
            return self.ensemble.predict(dataloader, return_variance=True)
        
        elif method == 'mc_dropout':
            # Implement MC dropout if needed
            logger.warning("MC Dropout not yet implemented for dataloader. Using ensemble.")
            return self.ensemble.predict(dataloader, return_variance=True)
        
        else:
            raise ValueError(f"Unknown UQ method: {method}")
    
    def evaluate_uncertainty(self, 
                           dataloader,
                           method: str = 'ensemble',
                           save_plots: bool = True,
                           results_dir: Path = Path("results/uq")) -> Dict:
        """Complete uncertainty evaluation pipeline"""
        
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get predictions with uncertainty
        predictions, uncertainties, targets = self.predict_with_uncertainty(
            dataloader, method=method
        )
        
        # Calculate performance metrics
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        if len(targets.shape) > 1:
            targets = targets.flatten()
        if len(uncertainties.shape) > 1:
            uncertainties = uncertainties.flatten()
        
        # Basic metrics
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        
        # Calibration analysis
        calibration_results = self.calibrator.calibrate(predictions, uncertainties, targets)
        
        # Create comprehensive results
        results = {
            'method': method,
            'performance': {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'n_samples': len(predictions)
            },
            'uncertainty_stats': {
                'mean_uncertainty': np.mean(uncertainties),
                'std_uncertainty': np.std(uncertainties),
                'min_uncertainty': np.min(uncertainties),
                'max_uncertainty': np.max(uncertainties)
            },
            'calibration': calibration_results
        }
        
        # Save plots
        if save_plots:
            plot_path = results_dir / f"calibration_analysis_{method}.png"
            self.calibrator.plot_calibration(save_path=plot_path)
        
        logger.info(f"🎯 UQ Evaluation Results:")
        logger.info(f"   R²: {r2:.4f}")
        logger.info(f"   RMSE: {rmse:.2f}")
        logger.info(f"   Error-Uncertainty Correlation: {calibration_results['error_correlation']['pearson']:.3f}")
        logger.info(f"   95% Coverage: {calibration_results['coverage'][-2]:.3f} (Expected: 0.95)")
        
        return results

def create_ensemble_from_hpo_results(hpo_results: Dict,
                                   dataset,
                                   device: torch.device,
                                   top_k: int = 5) -> List[nn.Module]:
    """Create ensemble from top HPO results"""
    
    # This would be implemented to load top-k models from HPO results
    # For now, return placeholder
    logger.info(f"🏗️  Creating ensemble from top-{top_k} HPO results")
    
    models = []
    # Implementation would load actual trained models here
    # models = load_top_k_models(hpo_results, dataset, device, top_k)
    
    return models

# Example usage functions
def run_uncertainty_analysis(model_paths: List[str],
                           dataset_path: str,
                           device: torch.device,
                           results_dir: str = "results/uq_analysis"):
    """Complete uncertainty analysis pipeline"""
    
    # Load ensemble models
    models = []
    for path in model_paths:
        # model = load_model(path, device)
        # models.append(model)
        pass
    
    # Create UQ system
    uq_system = UncertaintyQuantifier(models, device)
    
    # Load validation data
    # dataloader = create_dataloader(dataset_path, split='validation')
    
    # Run evaluation
    # results = uq_system.evaluate_uncertainty(dataloader, save_plots=True, 
    #                                         results_dir=Path(results_dir))
    
    return None  # Placeholder