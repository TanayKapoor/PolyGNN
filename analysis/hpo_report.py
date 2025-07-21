#!/usr/bin/env python3
"""
HPO Results Analysis and Reporting

Generate comprehensive analysis reports from hyperparameter optimization results.
Creates visualizations, parameter importance analysis, and summary PDFs.

Usage:
    python analysis/hpo_report.py --hpo_dir results/hpo/hpo_20231215_143022
    python analysis/hpo_report.py --results_dir results --generate_all
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
from matplotlib.backends.backend_pdf import PdfPages
import warnings

# Setup style and warnings
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HPOAnalyzer:
    """Comprehensive analysis of hyperparameter optimization results."""
    
    def __init__(self, hpo_dir: Path):
        """
        Initialize the HPO analyzer.
        
        Args:
            hpo_dir: Directory containing HPO results
        """
        self.hpo_dir = Path(hpo_dir)
        self.results = None
        self.df = None
        self.best_params = None
        self.success_criteria = {'r2': 0.5, 'rmse': 50.0, 'mae': 30.0}
        
        # Load results
        self._load_results()
    
    def _load_results(self):
        """Load HPO results from files."""
        # Load summary JSON
        summary_file = self.hpo_dir / "hpo_summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"HPO summary not found: {summary_file}")
        
        with open(summary_file, 'r') as f:
            self.results = json.load(f)
        
        self.best_params = self.results.get('best_params', {})
        
        # Load CSV results if available
        csv_file = self.hpo_dir / "hpo_results.csv"
        if csv_file.exists():
            self.df = pd.read_csv(csv_file)
            # Parse params column from string to dict-like
            self.df['parsed_params'] = self.df['params'].apply(self._parse_params_string)
            logger.info(f"Loaded {len(self.df)} trial results")
        else:
            logger.warning("CSV results not found, limited analysis available")
    
    def _parse_params_string(self, params_str: str) -> dict:
        """Parse parameter string back to dictionary."""
        try:
            # Simple parsing - could be improved with ast.literal_eval
            return eval(params_str) if params_str else {}
        except:
            return {}
    
    def generate_comprehensive_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Generate a comprehensive PDF report with all analyses.
        
        Args:
            output_path: Path for output PDF
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = self.hpo_dir / "hpo_comprehensive_report.pdf"
        
        logger.info(f"Generating comprehensive report: {output_path}")
        
        with PdfPages(output_path) as pdf:
            # Title page
            self._create_title_page(pdf)
            
            # Summary page
            self._create_summary_page(pdf)
            
            # Performance analysis
            self._create_performance_analysis(pdf)
            
            # Parameter importance
            self._create_parameter_importance(pdf)
            
            # Best configuration analysis
            self._create_best_config_analysis(pdf)
            
            # Learning curves
            self._create_learning_curves(pdf)
            
            # Parameter correlation analysis
            self._create_correlation_analysis(pdf)
            
            # Recommendations
            self._create_recommendations_page(pdf)
        
        logger.info(f"Report generated: {output_path}")
        return output_path
    
    def _create_title_page(self, pdf):
        """Create title page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, 'Polymer GCN\nHyperparameter Optimization Report', 
                ha='center', va='center', fontsize=24, fontweight='bold',
                transform=ax.transAxes)
        
        # HPO info
        hpo_info = f"""
HPO ID: {self.results.get('hpo_id', 'N/A')}
Method: {self.results.get('method', 'N/A').upper()}
Total Trials: {self.results.get('total_trials', 'N/A')}
Successful Trials: {self.results.get('successful_trials', 'N/A')}
Total Time: {self.results.get('total_time', 0) / 3600:.2f} hours
Primary Metric: {self.results.get('primary_metric', 'N/A').upper()}
Best Score: {self.results.get('best_score', 'N/A'):.4f}
        """
        
        ax.text(0.5, 0.5, hpo_info, ha='center', va='center', fontsize=12,
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Success criteria
        success_text = """
Success Criteria:
• R² ≥ 0.5
• RMSE ≤ 50.0
• MAE ≤ 30.0
        """
        ax.text(0.5, 0.2, success_text, ha='center', va='center', fontsize=12,
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_summary_page(self, pdf):
        """Create summary page with key statistics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('HPO Summary Statistics', fontsize=16, fontweight='bold')
        
        if self.df is not None and len(self.df) > 0:
            # Filter successful trials
            successful_df = self.df[self.df['status'] == 'success'].copy()
            
            if len(successful_df) > 0:
                # Trial performance distribution
                ax1.hist(successful_df['cv_r2_mean'], bins=20, alpha=0.7, edgecolor='black')
                ax1.axvline(self.success_criteria['r2'], color='red', linestyle='--', label='Target')
                ax1.set_xlabel('Cross-Validation R²')
                ax1.set_ylabel('Frequency')
                ax1.set_title('R² Score Distribution')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Performance progression
                ax2.plot(successful_df['trial_id'], successful_df['cv_r2_mean'], 
                        marker='o', markersize=3, alpha=0.7)
                ax2.axhline(self.success_criteria['r2'], color='red', linestyle='--', label='Target')
                ax2.set_xlabel('Trial Number')
                ax2.set_ylabel('Cross-Validation R²')
                ax2.set_title('Performance Progression')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Success rate by metric
                r2_success = (successful_df['cv_r2_mean'] >= self.success_criteria['r2']).mean()
                rmse_success = (successful_df['cv_rmse_mean'] <= self.success_criteria['rmse']).mean()
                mae_success = (successful_df['cv_mae_mean'] <= self.success_criteria['mae']).mean()
                
                metrics = ['R²', 'RMSE', 'MAE']
                success_rates = [r2_success, rmse_success, mae_success]
                colors = ['green' if rate >= 0.5 else 'orange' if rate >= 0.2 else 'red' 
                         for rate in success_rates]
                
                bars = ax3.bar(metrics, success_rates, color=colors, alpha=0.7, edgecolor='black')
                ax3.set_ylabel('Success Rate')
                ax3.set_title('Success Rate by Metric')
                ax3.set_ylim(0, 1)
                ax3.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, rate in zip(bars, success_rates):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{rate:.2%}', ha='center', va='bottom')
                
                # Best trials table
                top_trials = successful_df.nlargest(5, 'cv_r2_mean')[
                    ['trial_id', 'cv_r2_mean', 'cv_rmse_mean', 'cv_mae_mean']
                ].round(4)
                
                ax4.axis('tight')
                ax4.axis('off')
                table = ax4.table(cellText=top_trials.values,
                                colLabels=['Trial ID', 'R²', 'RMSE', 'MAE'],
                                cellLoc='center',
                                loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                ax4.set_title('Top 5 Trials by R²', pad=20)
            else:
                # No successful trials
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No Successful Trials', ha='center', va='center',
                           transform=ax.transAxes, fontsize=14, fontweight='bold', color='red')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_performance_analysis(self, pdf):
        """Create performance analysis plots."""
        if self.df is None or len(self.df) == 0:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Performance Analysis', fontsize=16, fontweight='bold')
        
        successful_df = self.df[self.df['status'] == 'success'].copy()
        
        if len(successful_df) > 0:
            # Performance scatter plots
            ax1.scatter(successful_df['cv_r2_mean'], successful_df['cv_rmse_mean'], 
                       alpha=0.6, s=30)
            ax1.axvline(self.success_criteria['r2'], color='red', linestyle='--', alpha=0.7)
            ax1.axhline(self.success_criteria['rmse'], color='red', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Cross-Validation R²')
            ax1.set_ylabel('Cross-Validation RMSE')
            ax1.set_title('R² vs RMSE Trade-off')
            ax1.grid(True, alpha=0.3)
            
            ax2.scatter(successful_df['cv_r2_mean'], successful_df['cv_mae_mean'], 
                       alpha=0.6, s=30)
            ax2.axvline(self.success_criteria['r2'], color='red', linestyle='--', alpha=0.7)
            ax2.axhline(self.success_criteria['mae'], color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Cross-Validation R²')
            ax2.set_ylabel('Cross-Validation MAE')
            ax2.set_title('R² vs MAE Trade-off')
            ax2.grid(True, alpha=0.3)
            
            # Performance statistics
            stats_data = {
                'Metric': ['R²', 'RMSE', 'MAE'],
                'Mean': [
                    successful_df['cv_r2_mean'].mean(),
                    successful_df['cv_rmse_mean'].mean(),
                    successful_df['cv_mae_mean'].mean()
                ],
                'Std': [
                    successful_df['cv_r2_std'].mean(),
                    successful_df['cv_rmse_std'].mean(),
                    successful_df['cv_mae_std'].mean()
                ],
                'Best': [
                    successful_df['cv_r2_mean'].max(),
                    successful_df['cv_rmse_mean'].min(),
                    successful_df['cv_mae_mean'].min()
                ],
                'Target': [
                    self.success_criteria['r2'],
                    self.success_criteria['rmse'],
                    self.success_criteria['mae']
                ]
            }
            
            stats_df = pd.DataFrame(stats_data).round(4)
            
            ax3.axis('tight')
            ax3.axis('off')
            table = ax3.table(cellText=stats_df.values,
                            colLabels=stats_df.columns,
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax3.set_title('Performance Statistics', pad=20)
            
            # Trials over time
            successful_df['cumulative_best_r2'] = successful_df['cv_r2_mean'].expanding().max()
            ax4.plot(successful_df['trial_id'], successful_df['cumulative_best_r2'], 
                    marker='o', markersize=3, linewidth=2, label='Best R² So Far')
            ax4.axhline(self.success_criteria['r2'], color='red', linestyle='--', 
                       alpha=0.7, label='Target')
            ax4.set_xlabel('Trial Number')
            ax4.set_ylabel('Best R² Score')
            ax4.set_title('Optimization Progress')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_parameter_importance(self, pdf):
        """Create parameter importance analysis."""
        if self.df is None or len(self.df) == 0:
            return
        
        successful_df = self.df[self.df['status'] == 'success'].copy()
        if len(successful_df) == 0:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Parameter Importance Analysis', fontsize=16, fontweight='bold')
        
        # Extract parameter values for analysis
        param_data = []
        for _, row in successful_df.iterrows():
            params = row['parsed_params']
            if params:
                params['r2_score'] = row['cv_r2_mean']
                params['rmse_score'] = row['cv_rmse_mean']
                params['mae_score'] = row['cv_mae_mean']
                param_data.append(params)
        
        if param_data:
            param_df = pd.DataFrame(param_data)
            
            # Learning rate vs performance
            if 'learning_rate' in param_df.columns:
                lr_groups = param_df.groupby('learning_rate')['r2_score'].agg(['mean', 'std', 'count'])
                lr_groups = lr_groups[lr_groups['count'] >= 2]  # Only show groups with multiple trials
                
                if not lr_groups.empty:
                    x_pos = range(len(lr_groups))
                    ax1.bar(x_pos, lr_groups['mean'], yerr=lr_groups['std'], 
                           capsize=5, alpha=0.7, edgecolor='black')
                    ax1.set_xlabel('Learning Rate')
                    ax1.set_ylabel('Mean R² Score')
                    ax1.set_title('Learning Rate Impact')
                    ax1.set_xticks(x_pos)
                    ax1.set_xticklabels([f'{lr:.0e}' for lr in lr_groups.index], rotation=45)
                    ax1.grid(True, alpha=0.3)
            
            # Hidden dimensions impact
            if 'hidden_dims' in param_df.columns:
                # Convert hidden_dims to string for grouping
                param_df['hidden_dims_str'] = param_df['hidden_dims'].astype(str)
                hd_groups = param_df.groupby('hidden_dims_str')['r2_score'].agg(['mean', 'count'])
                hd_groups = hd_groups[hd_groups['count'] >= 2].nlargest(8, 'mean')
                
                if not hd_groups.empty:
                    x_pos = range(len(hd_groups))
                    bars = ax2.bar(x_pos, hd_groups['mean'], alpha=0.7, edgecolor='black')
                    ax2.set_xlabel('Hidden Dimensions')
                    ax2.set_ylabel('Mean R² Score')
                    ax2.set_title('Architecture Impact (Top 8)')
                    ax2.set_xticks(x_pos)
                    # Shorten labels
                    labels = [hd.replace(' ', '').replace('[', '').replace(']', '') 
                             for hd in hd_groups.index]
                    ax2.set_xticklabels(labels, rotation=45, ha='right')
                    ax2.grid(True, alpha=0.3)
            
            # Dropout rate impact
            if 'dropout_rate' in param_df.columns:
                dropout_groups = param_df.groupby('dropout_rate')['r2_score'].agg(['mean', 'std', 'count'])
                dropout_groups = dropout_groups[dropout_groups['count'] >= 2]
                
                if not dropout_groups.empty:
                    x_pos = range(len(dropout_groups))
                    ax3.bar(x_pos, dropout_groups['mean'], yerr=dropout_groups['std'],
                           capsize=5, alpha=0.7, edgecolor='black')
                    ax3.set_xlabel('Dropout Rate')
                    ax3.set_ylabel('Mean R² Score')
                    ax3.set_title('Regularization Impact')
                    ax3.set_xticks(x_pos)
                    ax3.set_xticklabels(dropout_groups.index)
                    ax3.grid(True, alpha=0.3)
            
            # Best parameter combination
            best_trial = successful_df.loc[successful_df['cv_r2_mean'].idxmax()]
            best_params = best_trial['parsed_params']
            
            if best_params:
                ax4.axis('tight')
                ax4.axis('off')
                
                param_text = []
                for key, value in best_params.items():
                    if key in ['r2_score', 'rmse_score', 'mae_score']:
                        continue
                    param_text.append([key, str(value)])
                
                if param_text:
                    table = ax4.table(cellText=param_text,
                                    colLabels=['Parameter', 'Best Value'],
                                    cellLoc='center',
                                    loc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1.2, 1.5)
                    ax4.set_title(f'Best Configuration (R² = {best_trial["cv_r2_mean"]:.4f})', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_best_config_analysis(self, pdf):
        """Create best configuration detailed analysis."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Best Configuration Analysis', ha='center', va='top',
               fontsize=18, fontweight='bold', transform=ax.transAxes)
        
        if self.best_params:
            # Best parameters table
            y_pos = 0.85
            ax.text(0.5, y_pos, 'Optimal Hyperparameters', ha='center', va='top',
                   fontsize=14, fontweight='bold', transform=ax.transAxes)
            
            y_pos -= 0.05
            param_text = []
            for key, value in self.best_params.items():
                if isinstance(value, (list, tuple)):
                    value_str = str(value).replace(' ', '')
                elif isinstance(value, float):
                    value_str = f"{value:.6g}"
                else:
                    value_str = str(value)
                param_text.append(f"• {key}: {value_str}")
            
            param_string = '\n'.join(param_text)
            ax.text(0.1, y_pos, param_string, ha='left', va='top',
                   fontsize=10, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            
            # Performance summary
            y_pos = 0.5
            ax.text(0.5, y_pos, 'Best Performance', ha='center', va='top',
                   fontsize=14, fontweight='bold', transform=ax.transAxes)
            
            y_pos -= 0.05
            best_score = self.results.get('best_score', 'N/A')
            primary_metric = self.results.get('primary_metric', 'N/A')
            
            perf_text = f"""
Best {primary_metric.upper()}: {best_score}

This represents the best cross-validation performance
achieved during hyperparameter optimization.
            """
            
            ax.text(0.1, y_pos, perf_text, ha='left', va='top',
                   fontsize=11, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_learning_curves(self, pdf):
        """Create learning curves placeholder."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.5, 'Learning Curves\n\n(Would require training history data\nfrom individual trials)', 
               ha='center', va='center', fontsize=16,
               transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_correlation_analysis(self, pdf):
        """Create parameter correlation analysis."""
        if self.df is None or len(self.df) == 0:
            return
        
        successful_df = self.df[self.df['status'] == 'success'].copy()
        if len(successful_df) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle('Parameter Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Extract numeric parameters for correlation
        numeric_params = []
        for _, row in successful_df.iterrows():
            params = row['parsed_params']
            if params:
                numeric_row = {'r2_score': row['cv_r2_mean']}
                for key, value in params.items():
                    if isinstance(value, (int, float)) and key not in ['r2_score', 'rmse_score', 'mae_score']:
                        numeric_row[key] = value
                numeric_params.append(numeric_row)
        
        if numeric_params and len(numeric_params) > 5:
            param_df = pd.DataFrame(numeric_params)
            
            # Correlation with R² score
            correlations = param_df.corr()['r2_score'].drop('r2_score').sort_values(key=abs, ascending=False)
            
            if len(correlations) > 0:
                ax1.barh(range(len(correlations)), correlations.values, 
                        color=['green' if x > 0 else 'red' for x in correlations.values],
                        alpha=0.7)
                ax1.set_yticks(range(len(correlations)))
                ax1.set_yticklabels(correlations.index)
                ax1.set_xlabel('Correlation with R²')
                ax1.set_title('Parameter Correlation with Performance')
                ax1.grid(True, alpha=0.3)
                ax1.axvline(0, color='black', linewidth=0.5)
            
            # Parameter correlation heatmap (if enough parameters)
            if param_df.shape[1] > 3:
                corr_matrix = param_df.corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, ax=ax2)
                ax2.set_title('Parameter Correlation Matrix')
        else:
            ax1.text(0.5, 0.5, 'Insufficient Data\nfor Correlation Analysis', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'Insufficient Data\nfor Correlation Analysis', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_recommendations_page(self, pdf):
        """Create recommendations page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Recommendations & Next Steps', ha='center', va='top',
               fontsize=18, fontweight='bold', transform=ax.transAxes)
        
        recommendations = """
HYPERPARAMETER OPTIMIZATION RECOMMENDATIONS

1. MODEL ARCHITECTURE
   • Use the identified best architecture configuration
   • Consider ensemble methods if single model doesn't meet criteria
   • Experiment with advanced architectures (GAT, GraphSAGE)

2. TRAINING STRATEGY  
   • Use optimized learning rate and weight decay
   • Implement learning rate scheduling for final training
   • Increase training epochs for final model

3. FEATURE ENGINEERING
   • Analyze importance of polymer vs molecular features
   • Consider additional polymer descriptors
   • Experiment with graph augmentation techniques

4. DATA QUALITY
   • Review outlier samples that may affect performance
   • Consider data augmentation for small datasets
   • Validate SMILES quality and graph construction

5. ADVANCED TECHNIQUES
   • Try Bayesian optimization for more efficient search
   • Consider multi-objective optimization
   • Implement uncertainty quantification

6. PERFORMANCE IMPROVEMENT
   • If targets not met, try:
     - Larger/deeper models
     - Transfer learning from related datasets  
     - Ensemble of top-performing models
     - Domain-specific feature engineering
        """
        
        ax.text(0.05, 0.85, recommendations, ha='left', va='top',
               fontsize=10, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def save_parameter_summary(self, output_path: Optional[Path] = None) -> Path:
        """Save a CSV summary of parameter performance."""
        if output_path is None:
            output_path = self.hpo_dir / "parameter_summary.csv"
        
        if self.df is not None and len(self.df) > 0:
            successful_df = self.df[self.df['status'] == 'success'].copy()
            
            if len(successful_df) > 0:
                # Create parameter summary
                param_data = []
                for _, row in successful_df.iterrows():
                    params = row['parsed_params']
                    if params:
                        summary_row = {
                            'trial_id': row['trial_id'],
                            'cv_r2_mean': row['cv_r2_mean'],
                            'cv_rmse_mean': row['cv_rmse_mean'],
                            'cv_mae_mean': row['cv_mae_mean'],
                            **params
                        }
                        param_data.append(summary_row)
                
                summary_df = pd.DataFrame(param_data)
                summary_df.to_csv(output_path, index=False)
                logger.info(f"Parameter summary saved: {output_path}")
        
        return output_path


def find_hpo_directories(results_dir: Path) -> List[Path]:
    """Find all HPO directories in results."""
    hpo_base = results_dir / "hpo"
    if not hpo_base.exists():
        return []
    
    hpo_dirs = [d for d in hpo_base.iterdir() if d.is_dir() and d.name.startswith('hpo_')]
    return sorted(hpo_dirs, key=lambda x: x.name, reverse=True)


def main():
    parser = argparse.ArgumentParser(description='Generate HPO analysis reports')
    parser.add_argument('--hpo_dir', type=str, help='Specific HPO directory to analyze')
    parser.add_argument('--results_dir', type=str, default='results', 
                       help='Results directory containing HPO runs')
    parser.add_argument('--generate_all', action='store_true',
                       help='Generate reports for all HPO runs in results_dir')
    parser.add_argument('--output_dir', type=str, help='Output directory for reports')
    
    args = parser.parse_args()
    
    if args.hpo_dir:
        # Analyze specific HPO run
        hpo_dir = Path(args.hpo_dir)
        if not hpo_dir.exists():
            logger.error(f"HPO directory not found: {hpo_dir}")
            return 1
        
        analyzer = HPOAnalyzer(hpo_dir)
        report_path = analyzer.generate_comprehensive_report()
        analyzer.save_parameter_summary()
        
        logger.info(f"Analysis complete! Report: {report_path}")
        
    elif args.generate_all:
        # Analyze all HPO runs
        results_dir = Path(args.results_dir)
        hpo_dirs = find_hpo_directories(results_dir)
        
        if not hpo_dirs:
            logger.error(f"No HPO directories found in {results_dir}")
            return 1
        
        logger.info(f"Found {len(hpo_dirs)} HPO runs to analyze")
        
        for hpo_dir in hpo_dirs:
            logger.info(f"Analyzing: {hpo_dir.name}")
            try:
                analyzer = HPOAnalyzer(hpo_dir)
                analyzer.generate_comprehensive_report()
                analyzer.save_parameter_summary()
            except Exception as e:
                logger.error(f"Failed to analyze {hpo_dir.name}: {e}")
        
        logger.info("All analyses complete!")
        
    else:
        logger.error("Please specify --hpo_dir or --generate_all")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 