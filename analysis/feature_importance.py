"""
Feature Importance Analysis for Polymer GNN Models

This module provides comprehensive analysis of feature importance in polymer GNN models,
including polymer-specific features (molecular weight, degree of polymerization, Morgan fingerprints).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from torch_geometric.loader import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis results."""
    feature_names: List[str]
    importance_scores: np.ndarray
    importance_std: np.ndarray
    baseline_performance: Dict[str, float]
    feature_group_importance: Dict[str, float]
    analysis_metadata: Dict[str, Any]


class PolymerFeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analyzer for polymer GNN models.
    Evaluates the contribution of different feature groups and individual features.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the feature importance analyzer.
        
        Args:
            model: Trained polymer GNN model
            device: Device to run analysis on
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()
        
        # Feature group definitions
        self.feature_groups = {
            'graph_features': 'Graph structural features from GCN layers',
            'molecular_features': 'RDKit molecular descriptors',
            'polymer_molecular_weight': 'Repeating unit molecular weight',
            'polymer_degree_polymerization': 'Degree of polymerization encoding',
            'polymer_morgan_fingerprint': 'Morgan fingerprint of repeating unit'
        }
        
        logger.info("Initialized PolymerFeatureImportanceAnalyzer")
    
    def analyze_feature_importance(self,
                                 test_dataset,
                                 batch_size: int = 32,
                                 n_repeats: int = 10,
                                 random_state: int = 42) -> FeatureImportanceResult:
        """
        Perform comprehensive feature importance analysis.
        
        Args:
            test_dataset: Test dataset for analysis
            batch_size: Batch size for evaluation
            n_repeats: Number of permutation repeats
            random_state: Random seed for reproducibility
            
        Returns:
            FeatureImportanceResult containing analysis results
        """
        logger.info("Starting feature importance analysis...")
        
        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Get baseline performance
        logger.info("Computing baseline performance...")
        baseline_metrics = self._compute_baseline_performance(test_loader)
        
        # Analyze feature group importance
        logger.info("Analyzing feature group importance...")
        group_importance = self._analyze_feature_groups(test_loader, baseline_metrics)
        
        # Analyze individual polymer feature importance
        logger.info("Analyzing individual polymer features...")
        polymer_feature_importance = self._analyze_polymer_features(
            test_loader, n_repeats=n_repeats, random_state=random_state
        )
        
        # Combine results
        feature_names = polymer_feature_importance['feature_names']
        importance_scores = polymer_feature_importance['importance_scores']
        importance_std = polymer_feature_importance['importance_std']
        
        # Create result object
        result = FeatureImportanceResult(
            feature_names=feature_names,
            importance_scores=importance_scores,
            importance_std=importance_std,
            baseline_performance=baseline_metrics,
            feature_group_importance=group_importance,
            analysis_metadata={
                'n_samples': len(test_dataset),
                'n_repeats': n_repeats,
                'random_state': random_state,
                'analysis_date': datetime.now().isoformat(),
                'model_parameters': sum(p.numel() for p in self.model.parameters())
            }
        )
        
        logger.info("Feature importance analysis completed")
        return result
    
    def _compute_baseline_performance(self, data_loader) -> Dict[str, float]:
        """Compute baseline model performance metrics."""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                predictions = self.model(batch)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch.y.cpu().numpy().flatten())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        return {
            'mse': float(mean_squared_error(targets, predictions)),
            'mae': float(mean_absolute_error(targets, predictions)), 
            'r2': float(r2_score(targets, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(targets, predictions)))
        }
    
    def _analyze_feature_groups(self, data_loader, baseline_metrics: Dict) -> Dict[str, float]:
        """Analyze importance of different feature groups."""
        group_importance = {}
        
        # Test without molecular features
        if self.model.use_molecular_features:
            perf_no_mol = self._evaluate_without_features(data_loader, exclude='molecular')
            group_importance['molecular_features'] = baseline_metrics['r2'] - perf_no_mol['r2']
        
        # Test without polymer features
        if self.model.use_polymer_features:
            perf_no_polymer = self._evaluate_without_features(data_loader, exclude='polymer')
            group_importance['polymer_features'] = baseline_metrics['r2'] - perf_no_polymer['r2']
            
            # Analyze individual polymer feature components
            group_importance.update(self._analyze_polymer_feature_components(data_loader, baseline_metrics))
        
        return group_importance
    
    def _analyze_polymer_feature_components(self, data_loader, baseline_metrics: Dict) -> Dict[str, float]:
        """Analyze individual components of polymer features."""
        component_importance = {}
        
        # Create modified datasets for ablation study
        for component in ['mw', 'dp', 'fingerprint']:
            try:
                perf = self._evaluate_without_polymer_component(data_loader, exclude_component=component)
                component_importance[f'polymer_{component}'] = baseline_metrics['r2'] - perf['r2']
            except Exception as e:
                logger.warning(f"Could not evaluate without {component}: {e}")
                component_importance[f'polymer_{component}'] = 0.0
        
        return component_importance
    
    def _evaluate_without_features(self, data_loader, exclude: str) -> Dict[str, float]:
        """Evaluate model performance without specific feature groups."""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                # Modify batch to exclude features
                if exclude == 'molecular' and hasattr(batch, 'mol_features'):
                    # Temporarily remove molecular features
                    original_mol_features = batch.mol_features
                    delattr(batch, 'mol_features')
                    predictions = self.model(batch)
                    batch.mol_features = original_mol_features
                elif exclude == 'polymer' and hasattr(batch, 'polymer_features'):
                    # Temporarily remove polymer features
                    original_polymer_features = batch.polymer_features
                    delattr(batch, 'polymer_features')
                    predictions = self.model(batch)
                    batch.polymer_features = original_polymer_features
                else:
                    predictions = self.model(batch)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch.y.cpu().numpy().flatten())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        return {
            'mse': float(mean_squared_error(targets, predictions)),
            'mae': float(mean_absolute_error(targets, predictions)),
            'r2': float(r2_score(targets, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(targets, predictions)))
        }
    
    def _evaluate_without_polymer_component(self, data_loader, exclude_component: str) -> Dict[str, float]:
        """Evaluate model performance without specific polymer feature components."""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                if hasattr(batch, 'polymer_features'):
                    # Create modified polymer features
                    original_features = batch.polymer_features.clone()
                    modified_features = self._mask_polymer_component(original_features, exclude_component)
                    batch.polymer_features = modified_features
                    
                    predictions = self.model(batch)
                    
                    # Restore original features
                    batch.polymer_features = original_features
                else:
                    predictions = self.model(batch)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch.y.cpu().numpy().flatten())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        return {
            'mse': float(mean_squared_error(targets, predictions)),
            'mae': float(mean_absolute_error(targets, predictions)),
            'r2': float(r2_score(targets, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(targets, predictions)))
        }
    
    def _mask_polymer_component(self, features: torch.Tensor, component: str) -> torch.Tensor:
        """Mask specific polymer feature components."""
        masked_features = features.clone()
        
        if component == 'mw':
            # Mask molecular weight (first feature)
            masked_features[..., 0] = 0
        elif component == 'dp':
            # Mask degree of polymerization (second feature)
            masked_features[..., 1] = 0
        elif component == 'fingerprint':
            # Mask Morgan fingerprint (features 2 onward)
            masked_features[..., 2:] = 0
        
        return masked_features
    
    def _analyze_polymer_features(self, data_loader, n_repeats: int, random_state: int) -> Dict:
        """Analyze importance of individual polymer features using permutation."""
        # Get feature names from polymer feature extractor
        sample_batch = next(iter(data_loader))
        if not hasattr(sample_batch, 'polymer_features'):
            logger.warning("No polymer features found in data")
            return {
                'feature_names': [],
                'importance_scores': np.array([]),
                'importance_std': np.array([])
            }
        
        polymer_dim = sample_batch.polymer_features.shape[-1]
        feature_names = ['unit_molecular_weight', 'degree_polymerization']
        feature_names += [f'morgan_fp_{i}' for i in range(polymer_dim - 2)]
        
        # Perform permutation importance analysis
        importance_scores = []
        importance_stds = []
        
        baseline_score = self._compute_baseline_performance(data_loader)['r2']
        
        for feature_idx in range(polymer_dim):
            feature_scores = []
            
            for repeat in range(n_repeats):
                # Permute specific feature and evaluate
                score = self._evaluate_with_permuted_feature(data_loader, feature_idx, random_state + repeat)
                importance = baseline_score - score['r2']
                feature_scores.append(importance)
            
            importance_scores.append(np.mean(feature_scores))
            importance_stds.append(np.std(feature_scores))
        
        return {
            'feature_names': feature_names,
            'importance_scores': np.array(importance_scores),
            'importance_std': np.array(importance_stds)
        }
    
    def _evaluate_with_permuted_feature(self, data_loader, feature_idx: int, random_state: int) -> Dict[str, float]:
        """Evaluate model with a specific polymer feature permuted."""
        np.random.seed(random_state)
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                if hasattr(batch, 'polymer_features'):
                    # Permute specific feature
                    modified_features = batch.polymer_features.clone()
                    feature_values = modified_features[..., feature_idx].cpu().numpy()
                    permuted_values = np.random.permutation(feature_values)
                    modified_features[..., feature_idx] = torch.tensor(permuted_values).to(self.device)
                    
                    # Temporarily replace features
                    original_features = batch.polymer_features
                    batch.polymer_features = modified_features
                    predictions = self.model(batch)
                    batch.polymer_features = original_features
                else:
                    predictions = self.model(batch)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch.y.cpu().numpy().flatten())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        return {
            'r2': float(r2_score(targets, predictions))
        }


def generate_feature_importance_report(result: FeatureImportanceResult,
                                     output_dir: str = 'results/feature_importance',
                                     save_plots: bool = True) -> str:
    """
    Generate comprehensive feature importance report.
    
    Args:
        result: Feature importance analysis results
        output_dir: Directory to save report and plots
        save_plots: Whether to save visualization plots
        
    Returns:
        Path to generated report file
    """
    logger.info("Generating feature importance report...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate report content
    report_data = _create_report_content(result)
    
    # Save JSON report
    json_path = output_path / 'feature_importance_report.json'
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Generate HTML report
    html_path = output_path / 'feature_importance_report.html'
    _generate_html_report(report_data, html_path)
    
    # Generate plots
    if save_plots:
        _create_feature_importance_plots(result, output_path)
    
    # Generate summary table
    summary_path = output_path / 'feature_importance_summary.csv'
    _create_summary_table(result, summary_path)
    
    logger.info(f"Feature importance report generated: {html_path}")
    return str(html_path)


def _create_report_content(result: FeatureImportanceResult) -> Dict:
    """Create structured report content."""
    return {
        'analysis_summary': {
            'analysis_date': result.analysis_metadata['analysis_date'],
            'n_samples': result.analysis_metadata['n_samples'],
            'n_features_analyzed': len(result.feature_names),
            'model_parameters': result.analysis_metadata['model_parameters']
        },
        'baseline_performance': result.baseline_performance,
        'feature_group_importance': result.feature_group_importance,
        'individual_features': {
            'names': result.feature_names,
            'importance_scores': result.importance_scores.tolist(),
            'importance_std': result.importance_std.tolist()
        },
        'top_features': _get_top_features(result, n_top=10),
        'recommendations': _generate_recommendations(result)
    }


def _get_top_features(result: FeatureImportanceResult, n_top: int = 10) -> List[Dict]:
    """Get top N most important features."""
    indices = np.argsort(result.importance_scores)[::-1][:n_top]
    
    top_features = []
    for i, idx in enumerate(indices):
        top_features.append({
            'rank': i + 1,
            'feature_name': result.feature_names[idx],
            'importance_score': float(result.importance_scores[idx]),
            'importance_std': float(result.importance_std[idx])
        })
    
    return top_features


def _generate_recommendations(result: FeatureImportanceResult) -> List[str]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []
    
    # Check polymer feature importance
    polymer_importance = result.feature_group_importance.get('polymer_features', 0)
    if polymer_importance > 0.1:
        recommendations.append(
            f"Polymer features contribute significantly to model performance (ΔR² = {polymer_importance:.3f}). "
            "Continue using polymer-specific features."
        )
    elif polymer_importance < 0.05:
        recommendations.append(
            f"Polymer features show limited impact (ΔR² = {polymer_importance:.3f}). "
            "Consider feature engineering or alternative polymer representations."
        )
    
    # Check individual feature components
    mw_importance = result.feature_group_importance.get('polymer_mw', 0)
    dp_importance = result.feature_group_importance.get('polymer_dp', 0)
    
    if mw_importance > dp_importance:
        recommendations.append(
            "Molecular weight is more important than degree of polymerization. "
            "Focus on accurate molecular weight estimation."
        )
    
    # Check Morgan fingerprint importance
    fp_features = [i for i, name in enumerate(result.feature_names) if name.startswith('morgan_fp_')]
    if fp_features:
        fp_importance = np.mean([result.importance_scores[i] for i in fp_features])
        if fp_importance < 0.01:
            recommendations.append(
                "Morgan fingerprint features show low importance. "
                "Consider alternative structural representations or different fingerprint parameters."
            )
    
    return recommendations


def _generate_html_report(report_data: Dict, output_path: Path):
    """Generate HTML report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Polymer GNN Feature Importance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Polymer GNN Feature Importance Analysis Report</h1>
        
        <div class="metric">
            <h2>Analysis Summary</h2>
            <p><strong>Analysis Date:</strong> {report_data['analysis_summary']['analysis_date']}</p>
            <p><strong>Samples Analyzed:</strong> {report_data['analysis_summary']['n_samples']}</p>
            <p><strong>Features Analyzed:</strong> {report_data['analysis_summary']['n_features_analyzed']}</p>
            <p><strong>Model Parameters:</strong> {report_data['analysis_summary']['model_parameters']:,}</p>
        </div>
        
        <h2>Baseline Performance</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>R²</td><td>{report_data['baseline_performance']['r2']:.4f}</td></tr>
            <tr><td>RMSE</td><td>{report_data['baseline_performance']['rmse']:.4f}</td></tr>
            <tr><td>MAE</td><td>{report_data['baseline_performance']['mae']:.4f}</td></tr>
        </table>
        
        <h2>Feature Group Importance</h2>
        <table>
            <tr><th>Feature Group</th><th>Importance (ΔR²)</th></tr>
            {_create_group_importance_table(report_data['feature_group_importance'])}
        </table>
        
        <h2>Top 10 Most Important Features</h2>
        <table>
            <tr><th>Rank</th><th>Feature Name</th><th>Importance Score</th><th>Std Dev</th></tr>
            {_create_top_features_table(report_data['top_features'])}
        </table>
        
        <h2>Recommendations</h2>
        <ul>
            {_create_recommendations_list(report_data['recommendations'])}
        </ul>
        
        <p><em>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def _create_group_importance_table(group_importance: Dict) -> str:
    """Create HTML table rows for feature group importance."""
    rows = []
    for group, importance in group_importance.items():
        rows.append(f"<tr><td>{group.replace('_', ' ').title()}</td><td>{importance:.4f}</td></tr>")
    return '\n'.join(rows)


def _create_top_features_table(top_features: List[Dict]) -> str:
    """Create HTML table rows for top features."""
    rows = []
    for feature in top_features:
        rows.append(
            f"<tr><td>{feature['rank']}</td><td>{feature['feature_name']}</td>"
            f"<td>{feature['importance_score']:.4f}</td><td>{feature['importance_std']:.4f}</td></tr>"
        )
    return '\n'.join(rows)


def _create_recommendations_list(recommendations: List[str]) -> str:
    """Create HTML list items for recommendations."""
    items = [f"<li>{rec}</li>" for rec in recommendations]
    return '\n'.join(items)


def _create_feature_importance_plots(result: FeatureImportanceResult, output_dir: Path):
    """Create visualization plots for feature importance."""
    plt.style.use('default')
    
    # Plot 1: Top features bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top 15 features
    top_indices = np.argsort(result.importance_scores)[::-1][:15]
    top_names = [result.feature_names[i] for i in top_indices]
    top_scores = result.importance_scores[top_indices]
    top_stds = result.importance_std[top_indices]
    
    bars = ax1.barh(range(len(top_names)), top_scores, xerr=top_stds, capsize=3)
    ax1.set_yticks(range(len(top_names)))
    ax1.set_yticklabels(top_names)
    ax1.set_xlabel('Importance Score')
    ax1.set_title('Top 15 Most Important Features')
    ax1.grid(axis='x', alpha=0.3)
    
    # Color bars by feature type
    colors = []
    for name in top_names:
        if name == 'unit_molecular_weight':
            colors.append('#1f77b4')  # blue
        elif name == 'degree_polymerization':
            colors.append('#ff7f0e')  # orange
        elif name.startswith('morgan_fp_'):
            colors.append('#2ca02c')  # green
        else:
            colors.append('#d62728')  # red
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Feature group importance
    if result.feature_group_importance:
        groups = list(result.feature_group_importance.keys())
        importances = list(result.feature_group_importance.values())
        
        bars2 = ax2.bar(groups, importances)
        ax2.set_ylabel('Importance (ΔR²)')
        ax2.set_title('Feature Group Importance')
        ax2.tick_params(axis='x', rotation=45)
        
        # Color bars
        colors2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(groups)]
        for bar, color in zip(bars2, colors2):
            bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Feature importance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(result.importance_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Importance Score')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Importance Scores')
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'importance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_summary_table(result: FeatureImportanceResult, output_path: Path):
    """Create CSV summary table."""
    df = pd.DataFrame({
        'feature_name': result.feature_names,
        'importance_score': result.importance_scores,
        'importance_std': result.importance_std,
        'rank': np.argsort(result.importance_scores)[::-1] + 1
    })
    
    # Add feature type
    df['feature_type'] = df['feature_name'].apply(lambda x: 
        'molecular_weight' if x == 'unit_molecular_weight' else
        'degree_polymerization' if x == 'degree_polymerization' else
        'morgan_fingerprint' if x.startswith('morgan_fp_') else
        'other'
    )
    
    df = df.sort_values('importance_score', ascending=False)
    df.to_csv(output_path, index=False)


def compare_model_performance(baseline_results: Dict,
                            enhanced_results: Dict,
                            output_path: str = 'results/model_comparison.json') -> Dict:
    """
    Compare performance between baseline and polymer-enhanced models.
    
    Args:
        baseline_results: Results from baseline model
        enhanced_results: Results from polymer-enhanced model  
        output_path: Path to save comparison results
        
    Returns:
        Comparison analysis dictionary
    """
    comparison = {
        'baseline_model': baseline_results,
        'enhanced_model': enhanced_results,
        'improvements': {},
        'analysis_date': datetime.now().isoformat()
    }
    
    # Calculate improvements
    for metric in ['r2', 'mae', 'rmse']:
        if metric in baseline_results and metric in enhanced_results:
            if metric == 'r2':
                # Higher is better for R²
                improvement = enhanced_results[metric] - baseline_results[metric]
            else:
                # Lower is better for MAE, RMSE
                improvement = baseline_results[metric] - enhanced_results[metric]
            
            comparison['improvements'][metric] = {
                'absolute': improvement,
                'relative_percent': (improvement / abs(baseline_results[metric])) * 100
            }
    
    # Save comparison
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison


# Example usage and testing
if __name__ == "__main__":
    print("Feature importance analysis module loaded successfully!")
    print("Use PolymerFeatureImportanceAnalyzer for comprehensive analysis.") 