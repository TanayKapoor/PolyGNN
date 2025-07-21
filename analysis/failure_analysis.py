"""
Comprehensive Failure Analysis for Polymer GNN Models

Deep dive into prediction failures to understand model weaknesses and identify
patterns in failed predictions. Includes error distribution analysis, worst-case
identification, and feature importance for failure cases.

Features:
- Load trained models and generate test predictions
- Error distribution analysis with histograms and statistics
- Identification of worst prediction failures
- Analysis of failure patterns by polymer structure
- SHAP analysis for feature importance on failures
- Visualization of error distributions and failure cases
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

# ML and analysis tools
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Import project modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.polymer_gcn import PolymerGCN, create_gcn_model_from_config
from src.training.gcn_trainer import PolymerGCNTrainer
from src.data.polymer_dataset import PolymerTgDataset
from src.features.polymer_features import PolymerFeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PolymerFailureAnalyzer:
    """Comprehensive failure analysis for polymer prediction models."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.models = {}
        self.predictions = {}
        self.error_analysis = {}
        
        # Set up feature extractor for structural analysis
        self.feature_extractor = PolymerFeatureExtractor(
            fingerprint_size=128,
            include_chain_descriptors=True,
            include_complexity=True,
            include_molecular_descriptors=True
        )
    
    def load_model_and_predictions(self, model_name: str, config_path: Optional[str] = None) -> Dict:
        """
        Load a trained model and generate predictions on test data.
        
        Args:
            model_name: Name of the model (e.g., 'tg_gcn_enhanced')
            config_path: Optional path to model config file
            
        Returns:
            Dictionary with predictions, targets, and metadata
        """
        logger.info(f"Loading model: {model_name}")
        
        # Load model checkpoint
        model_path = self.results_dir / f"{model_name}_best.pth"
        if not model_path.exists():
            model_path = self.results_dir / f"{model_name}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Load model results for configuration
        results_path = self.results_dir / f"{model_name}_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                model_results = json.load(f)
        else:
            model_results = {}
        
        # Load test dataset (you may need to adjust path)
        dataset_path = "data/processed/filtered_tg_dataset.csv"
        
        # Create test dataset
        test_dataset = PolymerTgDataset(
            root='./data/test',
            csv_file=dataset_path,
            smiles_col='processed_smiles',
            target_col='Tg',
            split_type='test',  # Only test data
            use_polymer_features=True
        )
        
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        # Load model architecture (you may need to adjust based on your config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model with appropriate configuration
        model = PolymerGCN(
            node_feature_dim=157,
            hidden_dims=[256, 128, 64],  # Default, may need to adjust
            num_gcn_layers=3,
            molecular_feature_dim=13,
            use_molecular_features=True,
            use_polymer_features=True,
            polymer_feature_dim=147
        ).to(device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Create trainer for predictions
        trainer = PolymerGCNTrainer(model, results_dir=str(self.results_dir))
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions, targets = trainer.predict(test_dataset, batch_size=32)
        
        # Calculate errors
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        # Store results
        result_data = {
            'model_name': model_name,
            'predictions': predictions,
            'targets': targets,
            'errors': errors,
            'abs_errors': abs_errors,
            'test_dataset': test_dataset,
            'model_results': model_results,
            'metrics': {
                'r2': r2_score(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions)),
                'mae': mean_absolute_error(targets, predictions),
                'max_abs_error': np.max(abs_errors),
                'std_error': np.std(errors)
            }
        }
        
        self.predictions[model_name] = result_data
        logger.info(f"Model loaded successfully. R²: {result_data['metrics']['r2']:.4f}")
        
        return result_data
    
    def analyze_error_distribution(self, model_name: str) -> Dict:
        """
        Analyze the distribution of prediction errors.
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Dictionary with error distribution statistics
        """
        if model_name not in self.predictions:
            raise ValueError(f"Model {model_name} not loaded. Run load_model_and_predictions first.")
        
        data = self.predictions[model_name]
        errors = data['errors']
        abs_errors = data['abs_errors']
        
        logger.info(f"Analyzing error distribution for {model_name}")
        
        # Calculate distribution statistics
        error_stats = {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'skewness': stats.skew(errors),
            'kurtosis': stats.kurtosis(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'q25_abs_error': np.percentile(abs_errors, 25),
            'q50_abs_error': np.percentile(abs_errors, 50),
            'q75_abs_error': np.percentile(abs_errors, 75),
            'q90_abs_error': np.percentile(abs_errors, 90),
            'q95_abs_error': np.percentile(abs_errors, 95),
            'q99_abs_error': np.percentile(abs_errors, 99),
        }
        
        # Identify error categories
        error_categories = {
            'excellent': np.sum(abs_errors < 10),  # < 10°C error
            'good': np.sum((abs_errors >= 10) & (abs_errors < 25)),  # 10-25°C
            'moderate': np.sum((abs_errors >= 25) & (abs_errors < 50)),  # 25-50°C
            'poor': np.sum((abs_errors >= 50) & (abs_errors < 100)),  # 50-100°C
            'terrible': np.sum(abs_errors >= 100),  # > 100°C error
        }
        
        total_samples = len(abs_errors)
        error_percentages = {k: (v / total_samples * 100) for k, v in error_categories.items()}
        
        analysis_result = {
            'model_name': model_name,
            'total_samples': total_samples,
            'error_stats': error_stats,
            'error_categories': error_categories,
            'error_percentages': error_percentages
        }
        
        self.error_analysis[model_name] = analysis_result
        return analysis_result
    
    def identify_failure_cases(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """
        Identify the worst prediction failures.
        
        Args:
            model_name: Name of the model to analyze
            top_n: Number of worst cases to return
            
        Returns:
            DataFrame with worst prediction cases and their details
        """
        if model_name not in self.predictions:
            raise ValueError(f"Model {model_name} not loaded. Run load_model_and_predictions first.")
        
        data = self.predictions[model_name]
        dataset = data['test_dataset']
        
        # Create failure analysis dataframe
        failure_df = pd.DataFrame({
            'sample_idx': range(len(data['predictions'])),
            'prediction': data['predictions'],
            'target': data['targets'],
            'error': data['errors'],
            'abs_error': data['abs_errors']
        })
        
        # Get SMILES strings from dataset
        smiles_list = []
        dp_list = []
        mw_list = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            # Extract SMILES from sample if available
            if hasattr(sample, 'smiles'):
                smiles_list.append(sample.smiles)
            else:
                smiles_list.append(f"sample_{i}")  # Fallback
            
            # Try to get DP and MW if available
            dp_list.append(getattr(sample, 'dp', None))
            mw_list.append(getattr(sample, 'mw', None))
        
        failure_df['smiles'] = smiles_list
        failure_df['dp'] = dp_list
        failure_df['mw'] = mw_list
        
        # Sort by absolute error (worst first)
        worst_cases = failure_df.nlargest(top_n, 'abs_error')
        
        # Add failure categories
        def categorize_failure(abs_error):
            if abs_error >= 100:
                return "Catastrophic (>100°C)"
            elif abs_error >= 50:
                return "Severe (50-100°C)"  
            elif abs_error >= 25:
                return "Moderate (25-50°C)"
            elif abs_error >= 10:
                return "Minor (10-25°C)"
            else:
                return "Good (<10°C)"
        
        worst_cases['failure_category'] = worst_cases['abs_error'].apply(categorize_failure)
        
        # Add relative error percentage
        worst_cases['relative_error_pct'] = (worst_cases['abs_error'] / np.abs(worst_cases['target']) * 100)
        
        logger.info(f"Identified {len(worst_cases)} worst failure cases for {model_name}")
        
        return worst_cases
    
    def analyze_structural_patterns(self, model_name: str, failure_df: pd.DataFrame) -> Dict:
        """
        Analyze structural patterns in failed predictions.
        
        Args:
            model_name: Name of the model
            failure_df: DataFrame with failure cases
            
        Returns:
            Dictionary with structural failure patterns
        """
        logger.info(f"Analyzing structural patterns in failures for {model_name}")
        
        structural_analysis = {
            'aromatic_failures': 0,
            'branched_failures': 0,
            'long_chain_failures': 0,
            'heteroatom_failures': 0,
            'complex_failures': 0
        }
        
        feature_analysis = []
        
        for idx, row in failure_df.iterrows():
            smiles = row['smiles']
            abs_error = row['abs_error']
            
            # Extract structural features for failed cases
            try:
                # Get polymer features
                dp = row['dp'] if pd.notna(row['dp']) else 1000  # Default DP
                features = self.feature_extractor.extract_features(smiles, dp)
                
                # Analyze structural characteristics
                if features is not None:
                    feature_breakdown = self.feature_extractor.get_feature_breakdown(features)
                    
                    # Analyze patterns
                    fingerprint_features = features[2:130]  # Fingerprint portion
                    complexity_features = features[135:141]  # Complexity features
                    
                    # Check for aromatic content (high fingerprint density)
                    aromatic_score = np.sum(fingerprint_features > 0) / len(fingerprint_features)
                    if aromatic_score > 0.1:
                        structural_analysis['aromatic_failures'] += 1
                    
                    # Check branching (complexity features)
                    if len(complexity_features) > 0 and complexity_features[3] > 0.5:  # Branching factor
                        structural_analysis['branched_failures'] += 1
                    
                    # Check for heteroatoms
                    if len(complexity_features) > 0 and complexity_features[1] > 0.2:  # Heteroatom ratio
                        structural_analysis['heteroatom_failures'] += 1
                    
                    # Overall complexity
                    complexity_score = np.mean(complexity_features) if len(complexity_features) > 0 else 0
                    if complexity_score > 0.3:
                        structural_analysis['complex_failures'] += 1
                    
                    feature_analysis.append({
                        'sample_idx': row['sample_idx'],
                        'smiles': smiles,
                        'abs_error': abs_error,
                        'aromatic_score': aromatic_score,
                        'complexity_score': complexity_score,
                        'n_nonzero_features': torch.sum(features != 0).item(),
                        'total_features': len(features)
                    })
                    
            except Exception as e:
                logger.warning(f"Could not analyze structure for {smiles}: {e}")
        
        return {
            'structural_patterns': structural_analysis,
            'feature_analysis': feature_analysis,
            'total_analyzed': len(feature_analysis)
        }
    
    def create_error_visualizations(self, model_name: str, save_plots: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive error distribution visualizations.
        
        Args:
            model_name: Name of the model to visualize
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary with plot objects and statistics
        """
        if model_name not in self.predictions:
            raise ValueError(f"Model {model_name} not loaded.")
        
        data = self.predictions[model_name]
        errors = data['errors']
        abs_errors = data['abs_errors']
        predictions = data['predictions']
        targets = data['targets']
        
        logger.info(f"Creating error visualizations for {model_name}")
        
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Error Distribution (Histogram)',
                'Absolute Error Distribution',
                'Predictions vs Targets',
                'Error vs Target Value',
                'Residuals vs Predicted',
                'Absolute Error Percentiles'
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Error distribution histogram
        fig.add_trace(
            go.Histogram(x=errors, nbinsx=50, name='Error Distribution',
                        opacity=0.7, marker_color='blue'),
            row=1, col=1
        )
        
        # Add normal distribution overlay
        x_norm = np.linspace(np.min(errors), np.max(errors), 100)
        y_norm = stats.norm.pdf(x_norm, np.mean(errors), np.std(errors))
        y_norm = y_norm * len(errors) * (np.max(errors) - np.min(errors)) / 50  # Scale to histogram
        
        fig.add_trace(
            go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Fit',
                      line=dict(color='red', dash='dash')),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Absolute error distribution
        fig.add_trace(
            go.Histogram(x=abs_errors, nbinsx=50, name='Absolute Error Distribution',
                        opacity=0.7, marker_color='orange'),
            row=1, col=2
        )
        
        # 3. Predictions vs Targets (parity plot)
        fig.add_trace(
            go.Scatter(x=targets, y=predictions, mode='markers',
                      name='Predictions', marker=dict(size=6, opacity=0.6)),
            row=2, col=1
        )
        
        # Add ideal line
        min_val, max_val = min(np.min(targets), np.min(predictions)), max(np.max(targets), np.max(predictions))
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Ideal (y=x)', 
                      line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # 4. Error vs Target Value
        fig.add_trace(
            go.Scatter(x=targets, y=errors, mode='markers',
                      name='Error vs Target', marker=dict(size=6, opacity=0.6, color='green')),
            row=2, col=2
        )
        
        # Add zero line
        fig.add_trace(
            go.Scatter(x=[np.min(targets), np.max(targets)], y=[0, 0],
                      mode='lines', name='Zero Error',
                      line=dict(color='red', dash='dash')),
            row=2, col=2
        )
        
        # 5. Residuals vs Predicted
        fig.add_trace(
            go.Scatter(x=predictions, y=errors, mode='markers',
                      name='Residuals', marker=dict(size=6, opacity=0.6, color='purple')),
            row=3, col=1
        )
        
        # 6. Absolute Error Percentiles
        percentiles = np.arange(0, 101, 5)
        error_percentiles = np.percentile(abs_errors, percentiles)
        
        fig.add_trace(
            go.Scatter(x=percentiles, y=error_percentiles, mode='lines+markers',
                      name='Error Percentiles', line=dict(color='red', width=3)),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title=f'Comprehensive Error Analysis - {model_name}',
            showlegend=True,
            font=dict(size=12)
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Error (°C)", row=1, col=1)
        fig.update_xaxes(title_text="Absolute Error (°C)", row=1, col=2)
        fig.update_xaxes(title_text="Actual Tg (°C)", row=2, col=1)
        fig.update_xaxes(title_text="Actual Tg (°C)", row=2, col=2)
        fig.update_xaxes(title_text="Predicted Tg (°C)", row=3, col=1)
        fig.update_xaxes(title_text="Percentile", row=3, col=2)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Predicted Tg (°C)", row=2, col=1)
        fig.update_yaxes(title_text="Error (°C)", row=2, col=2)
        fig.update_yaxes(title_text="Residual (°C)", row=3, col=1)
        fig.update_yaxes(title_text="Absolute Error (°C)", row=3, col=2)
        
        # Save plot if requested
        if save_plots:
            plot_path = self.results_dir / f"{model_name}_failure_analysis.html"
            fig.write_html(str(plot_path))
            logger.info(f"Error analysis plot saved to {plot_path}")
        
        return {
            'plot': fig,
            'error_stats': {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'mean_abs_error': np.mean(abs_errors),
                'median_abs_error': np.median(abs_errors),
                'max_abs_error': np.max(abs_errors),
                'r2': data['metrics']['r2']
            }
        }
    
    def generate_failure_report(self, model_name: str) -> str:
        """
        Generate a comprehensive failure analysis report.
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Formatted report string
        """
        if model_name not in self.predictions:
            raise ValueError(f"Model {model_name} not loaded.")
        
        # Run all analyses
        error_analysis = self.analyze_error_distribution(model_name)
        failure_cases = self.identify_failure_cases(model_name, top_n=10)
        structural_patterns = self.analyze_structural_patterns(model_name, failure_cases)
        
        data = self.predictions[model_name]
        
        # Generate report
        report = f"""
# Failure Analysis Report: {model_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Performance Summary
- **Total Samples**: {error_analysis['total_samples']}
- **R² Score**: {data['metrics']['r2']:.4f}
- **RMSE**: {data['metrics']['rmse']:.2f}°C
- **MAE**: {data['metrics']['mae']:.2f}°C
- **Max Absolute Error**: {data['metrics']['max_abs_error']:.2f}°C

## Error Distribution Analysis
- **Mean Error**: {error_analysis['error_stats']['mean_error']:.2f}°C
- **Median Error**: {error_analysis['error_stats']['median_error']:.2f}°C
- **Standard Deviation**: {error_analysis['error_stats']['std_error']:.2f}°C
- **Skewness**: {error_analysis['error_stats']['skewness']:.3f}
- **Kurtosis**: {error_analysis['error_stats']['kurtosis']:.3f}

## Error Categories
- **Excellent (<10°C)**: {error_analysis['error_percentages']['excellent']:.1f}% ({error_analysis['error_categories']['excellent']} samples)
- **Good (10-25°C)**: {error_analysis['error_percentages']['good']:.1f}% ({error_analysis['error_categories']['good']} samples)
- **Moderate (25-50°C)**: {error_analysis['error_percentages']['moderate']:.1f}% ({error_analysis['error_categories']['moderate']} samples)  
- **Poor (50-100°C)**: {error_analysis['error_percentages']['poor']:.1f}% ({error_analysis['error_categories']['poor']} samples)
- **Terrible (>100°C)**: {error_analysis['error_percentages']['terrible']:.1f}% ({error_analysis['error_categories']['terrible']} samples)

## Top 10 Worst Prediction Failures
"""
        
        for i, (_, row) in enumerate(failure_cases.head(10).iterrows()):
            report += f"""
### Failure Case #{i+1}
- **SMILES**: {row['smiles']}
- **Actual Tg**: {row['target']:.1f}°C
- **Predicted Tg**: {row['prediction']:.1f}°C
- **Absolute Error**: {row['abs_error']:.1f}°C
- **Relative Error**: {row['relative_error_pct']:.1f}%
- **Category**: {row['failure_category']}
"""
        
        # Add structural pattern analysis
        report += f"""
## Structural Failure Patterns
From analysis of worst {len(structural_patterns['feature_analysis'])} failure cases:
- **Aromatic-rich structures**: {structural_patterns['structural_patterns']['aromatic_failures']} cases
- **Highly branched polymers**: {structural_patterns['structural_patterns']['branched_failures']} cases
- **Heteroatom-rich polymers**: {structural_patterns['structural_patterns']['heteroatom_failures']} cases
- **Complex structures**: {structural_patterns['structural_patterns']['complex_failures']} cases

## Error Percentiles
- **90th percentile**: {error_analysis['error_stats']['q90_abs_error']:.1f}°C
- **95th percentile**: {error_analysis['error_stats']['q95_abs_error']:.1f}°C
- **99th percentile**: {error_analysis['error_stats']['q99_abs_error']:.1f}°C

## Recommendations
Based on the failure analysis:

1. **High Error Cases (>100°C)**: {error_analysis['error_categories']['terrible']} samples show catastrophic failures
   - May indicate missing structural features or extreme polymer architectures
   - Consider expanding training data for these polymer types

2. **Error Distribution**: 
   - Skewness of {error_analysis['error_stats']['skewness']:.3f} indicates {'right-skewed (overestimation)' if error_analysis['error_stats']['skewness'] > 0 else 'left-skewed (underestimation)'} errors
   - Consider bias correction techniques

3. **Structural Patterns**:
   - Focus on improving representation for complex/aromatic structures
   - Consider additional descriptors for branched polymers

---
*Analysis completed with PolymerFailureAnalyzer*
"""
        
        return report
    
    def save_failure_report(self, model_name: str) -> str:
        """
        Save the failure analysis report to file.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to saved report
        """
        report = self.generate_failure_report(model_name)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.results_dir / f"{model_name}_failure_analysis_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Failure analysis report saved to {report_path}")
        return str(report_path)


def main():
    """Run comprehensive failure analysis on trained models."""
    print("🔍 POLYMER GNN FAILURE ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PolymerFailureAnalyzer()
    
    # List of models to analyze
    models_to_analyze = [
        'tg_gcn_enhanced',
        'tg_gcn_baseline',
        'tg_gcn_optimized'
    ]
    
    for model_name in models_to_analyze:
        try:
            print(f"\n📊 Analyzing {model_name}...")
            
            # Load model and generate predictions
            analyzer.load_model_and_predictions(model_name)
            
            # Create visualizations
            analyzer.create_error_visualizations(model_name, save_plots=True)
            
            # Save comprehensive report
            report_path = analyzer.save_failure_report(model_name)
            print(f"✅ Report saved: {report_path}")
            
        except Exception as e:
            print(f"❌ Failed to analyze {model_name}: {e}")
            continue
    
    print(f"\n🎉 Failure analysis completed! Check the results directory for detailed reports and visualizations.")


if __name__ == "__main__":
    main() 