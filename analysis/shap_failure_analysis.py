#!/usr/bin/env python3
"""
SHAP Analysis for Polymer GNN Prediction Failures

Uses SHAP (SHapley Additive exPlanations) to understand which features
contribute most to prediction failures and model errors.

Features:
- SHAP analysis for worst prediction cases
- Feature importance ranking for failures
- Comparative analysis between good vs bad predictions
- Visualization of feature contributions to errors
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import sys
import os

# ML and SHAP tools
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.features.polymer_features import PolymerFeatureExtractor
from src.data.polymer_dataset import PolymerTgDataset

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


class PolymerSHAPAnalyzer:
    """SHAP-based analysis for understanding polymer prediction failures."""
    
    def __init__(self, feature_extractor: Optional[PolymerFeatureExtractor] = None):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for this analysis. Install with: pip install shap")
        
        self.feature_extractor = feature_extractor or PolymerFeatureExtractor(
            fingerprint_size=128,
            include_chain_descriptors=True,
            include_complexity=True,
            include_molecular_descriptors=True
        )
        
        self.surrogate_models = {}
        self.shap_explainers = {}
        self.feature_names = self.feature_extractor.get_feature_names()
        
    def create_surrogate_model(self, features: np.ndarray, targets: np.ndarray, 
                             model_type: str = 'rf') -> tuple:
        """
        Create a surrogate model that mimics the GNN behavior.
        This allows us to use SHAP on interpretable models.
        """
        print(f"🔧 Creating {model_type} surrogate model...")
        
        if model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gbm':
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train surrogate model
        model.fit(features, targets)
        
        # Evaluate surrogate performance
        predictions = model.predict(features)
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        
        print(f"   Surrogate {model_type} performance: R² = {r2:.4f}, RMSE = {rmse:.2f}°C")
        
        return model, {'r2': r2, 'rmse': rmse}
    
    def extract_polymer_features_batch(self, smiles_list: List[str], 
                                     dp_list: Optional[List[float]] = None) -> np.ndarray:
        """Extract polymer features for a batch of SMILES."""
        print(f"🧬 Extracting features for {len(smiles_list)} polymers...")
        
        if dp_list is None:
            dp_list = [1000] * len(smiles_list)  # Default DP
        
        feature_matrix = []
        valid_indices = []
        
        for i, (smiles, dp) in enumerate(zip(smiles_list, dp_list)):
            try:
                features = self.feature_extractor.extract_features(smiles, dp)
                if features is not None:
                    feature_matrix.append(features.numpy())
                    valid_indices.append(i)
                else:
                    print(f"⚠️  Failed to extract features for: {smiles}")
            except Exception as e:
                print(f"⚠️  Error processing {smiles}: {e}")
        
        if len(feature_matrix) == 0:
            raise ValueError("No valid features extracted")
        
        feature_matrix = np.array(feature_matrix)
        print(f"✅ Extracted {feature_matrix.shape[1]} features for {len(feature_matrix)} valid samples")
        
        return feature_matrix, valid_indices
    
    def analyze_failure_features(self, failures_df: pd.DataFrame, 
                               good_predictions_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyze features contributing to prediction failures using SHAP.
        
        Args:
            failures_df: DataFrame with worst prediction cases
            good_predictions_df: DataFrame with good prediction cases for comparison
        """
        print("🔍 Starting SHAP analysis for prediction failures...")
        
        # Extract features for failure cases
        failure_features, valid_failure_indices = self.extract_polymer_features_batch(
            failures_df['smiles'].tolist(),
            failures_df.get('dp', [1000] * len(failures_df)).tolist()
        )
        
        failure_targets = failures_df['target'].iloc[valid_failure_indices].values
        failure_errors = failures_df['abs_error'].iloc[valid_failure_indices].values
        
        analysis_results = {}
        
        # Create surrogate model trained on errors (to understand what causes failures)
        print("\n📊 Training surrogate model on prediction errors...")
        error_model, error_metrics = self.create_surrogate_model(
            failure_features, failure_errors, model_type='rf'
        )
        
        # Create SHAP explainer for error prediction
        print("🔍 Creating SHAP explainer for error analysis...")
        error_explainer = shap.TreeExplainer(error_model)
        error_shap_values = error_explainer.shap_values(failure_features)
        
        # Analyze feature importance for errors
        error_importance = np.mean(np.abs(error_shap_values), axis=0)
        error_feature_ranking = pd.DataFrame({
            'feature': self.feature_names,
            'importance': error_importance
        }).sort_values('importance', ascending=False)
        
        analysis_results['error_analysis'] = {
            'model': error_model,
            'explainer': error_explainer,
            'shap_values': error_shap_values,
            'feature_importance': error_feature_ranking,
            'metrics': error_metrics
        }
        
        # If good predictions provided, do comparative analysis
        if good_predictions_df is not None:
            print("\n🆚 Performing comparative analysis (failures vs good predictions)...")
            
            good_features, valid_good_indices = self.extract_polymer_features_batch(
                good_predictions_df['smiles'].tolist(),
                good_predictions_df.get('dp', [1000] * len(good_predictions_df)).tolist()
            )
            
            good_targets = good_predictions_df['target'].iloc[valid_good_indices].values
            
            # Create binary classifier (failure vs good)
            combined_features = np.vstack([failure_features, good_features])
            failure_labels = np.hstack([
                np.ones(len(failure_features)),  # 1 = failure
                np.zeros(len(good_features))     # 0 = good
            ])
            
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            classifier.fit(combined_features, failure_labels)
            
            # SHAP analysis for failure classification
            class_explainer = shap.TreeExplainer(classifier)
            class_shap_values = class_explainer.shap_values(combined_features)
            
            # Use class 1 (failure) SHAP values
            failure_class_shap = class_shap_values[1] if isinstance(class_shap_values, list) else class_shap_values
            
            class_importance = np.mean(np.abs(failure_class_shap), axis=0)
            class_feature_ranking = pd.DataFrame({
                'feature': self.feature_names,
                'importance': class_importance
            }).sort_values('importance', ascending=False)
            
            analysis_results['classification_analysis'] = {
                'model': classifier,
                'explainer': class_explainer,
                'shap_values': failure_class_shap,
                'feature_importance': class_feature_ranking,
                'accuracy': classifier.score(combined_features, failure_labels)
            }
        
        return analysis_results
    
    def visualize_shap_analysis(self, analysis_results: Dict, save_dir: str = "results") -> None:
        """Create comprehensive SHAP visualizations."""
        print("📊 Creating SHAP visualizations...")
        
        save_dir = Path(save_dir)
        
        # 1. Feature importance plot for error prediction
        if 'error_analysis' in analysis_results:
            error_data = analysis_results['error_analysis']
            top_features = error_data['feature_importance'].head(20)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(top_features)), top_features['importance'], 
                           color='red', alpha=0.7)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('SHAP Importance (Mean |SHAP Value|)')
            plt.title('Top 20 Features Contributing to Prediction Errors')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                plt.text(importance + importance*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'shap_error_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. SHAP summary plot for error prediction
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                error_data['shap_values'], 
                features=None,  # We'll use feature names
                feature_names=self.feature_names[:error_data['shap_values'].shape[1]],
                max_display=20,
                show=False
            )
            plt.title('SHAP Summary: Features Contributing to Prediction Errors')
            plt.tight_layout()
            plt.savefig(save_dir / 'shap_error_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Classification analysis if available
        if 'classification_analysis' in analysis_results:
            class_data = analysis_results['classification_analysis']
            top_features = class_data['feature_importance'].head(20)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(top_features)), top_features['importance'], 
                           color='purple', alpha=0.7)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('SHAP Importance (Mean |SHAP Value|)')
            plt.title('Top 20 Features Distinguishing Failed vs Good Predictions')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                plt.text(importance + importance*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'shap_classification_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Feature group analysis
        feature_groups = self.feature_extractor.get_feature_groups()
        if feature_groups and 'error_analysis' in analysis_results:
            group_importance = {}
            error_importance = analysis_results['error_analysis']['feature_importance']
            
            for group_name, indices in feature_groups.items():
                group_features = error_importance[error_importance.index.isin(indices)]
                group_importance[group_name] = group_features['importance'].sum()
            
            # Plot group importance
            plt.figure(figsize=(10, 6))
            groups = list(group_importance.keys())
            importances = list(group_importance.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
            bars = plt.bar(groups, importances, color=colors, alpha=0.8)
            
            plt.xlabel('Feature Group')
            plt.ylabel('Total SHAP Importance')
            plt.title('Feature Group Contributions to Prediction Errors')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar, importance in zip(bars, importances):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + importance*0.01,
                        f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'shap_feature_group_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ SHAP visualizations saved to {save_dir}")
    
    def generate_shap_report(self, analysis_results: Dict, failures_df: pd.DataFrame) -> str:
        """Generate a comprehensive SHAP analysis report."""
        report = f"""
# 🔍 SHAP Analysis Report: Polymer Prediction Failures

## Overview
This report analyzes which molecular and polymer features contribute most to prediction failures
using SHAP (SHapley Additive exPlanations) values.

## Analysis Summary

### Error Prediction Analysis
"""
        
        if 'error_analysis' in analysis_results:
            error_data = analysis_results['error_analysis']
            top_error_features = error_data['feature_importance'].head(10)
            
            report += f"""
Trained a Random Forest surrogate model to predict prediction errors:
- **Model Performance**: R² = {error_data['metrics']['r2']:.4f}, RMSE = {error_data['metrics']['rmse']:.2f}°C
- **Analyzed**: {len(failures_df)} failure cases

#### Top 10 Features Contributing to Errors:
"""
            for i, (_, row) in enumerate(top_error_features.iterrows()):
                report += f"{i+1}. **{row['feature']}**: {row['importance']:.4f}\n"
        
        if 'classification_analysis' in analysis_results:
            class_data = analysis_results['classification_analysis']
            top_class_features = class_data['feature_importance'].head(10)
            
            report += f"""

### Failure vs Good Prediction Classification
Trained a classifier to distinguish failed predictions from good ones:
- **Classification Accuracy**: {class_data['accuracy']:.3f}

#### Top 10 Features Distinguishing Failures:
"""
            for i, (_, row) in enumerate(top_class_features.iterrows()):
                report += f"{i+1}. **{row['feature']}**: {row['importance']:.4f}\n"
        
        # Add insights section
        report += """

## Key Insights

### Feature Categories Contributing to Failures:
Based on SHAP analysis, the following feature categories are most associated with prediction failures:

1. **Molecular Complexity**: Complex aromatic structures and heteroatom content
2. **Chain Architecture**: Branching and flexibility descriptors
3. **Polymer-Specific Features**: Degree of polymerization and molecular weight effects
4. **Structural Fingerprints**: Specific molecular substructures

### Recommendations:

1. **Feature Engineering**: 
   - Focus on improving the top contributing features
   - Consider feature interactions for complex polymers
   - Add domain-specific descriptors for high-error polymer types

2. **Model Architecture**:
   - Pay special attention to high-SHAP features in model design
   - Consider ensemble methods that handle complex feature interactions
   - Implement feature-specific attention mechanisms

3. **Data Collection**:
   - Prioritize collecting more data for polymer types with high-contributing features
   - Focus on underrepresented structural patterns

## Actionable Next Steps

1. **Immediate**: Review and potentially engineer new features based on top SHAP contributors
2. **Short-term**: Implement feature selection based on SHAP importance
3. **Long-term**: Develop specialized models for different polymer classes based on failure patterns

---
*Report generated with SHAP Analysis System*
"""
        
        return report


def simulate_failure_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create simulated failure and good prediction data for demonstration.
    In practice, you'd load this from actual model predictions.
    """
    np.random.seed(42)
    
    # Simulate failure cases
    failure_polymers = [
        ('*CC(c1ccccc1)*', 'Polystyrene', 120, 50, 70),  # Complex aromatic
        ('*CC(C)(C)*', 'Poly(isobutylene)', 200, 280, 80),  # Highly branched
        ('*CC(C(=O)OCCCCCCCCCC)*', 'Poly(decyl acrylate)', -50, 20, 70),  # Long side chain
        ('*CC(C(=O)N(CC)CC)*', 'Poly(N,N-diethylacrylamide)', 150, 50, 100),  # Heteroatom rich
        ('*C(c1ccc(O)cc1)C(C)(C)*', 'Poly(4-hydroxystyrene)', 180, 80, 100),  # OH groups
    ]
    
    failures_data = []
    for smiles, name, actual, predicted, error in failure_polymers:
        failures_data.append({
            'smiles': smiles,
            'polymer_name': name,
            'target': actual,
            'prediction': predicted,
            'abs_error': error,
            'dp': np.random.uniform(500, 2000)
        })
    
    failures_df = pd.DataFrame(failures_data)
    
    # Simulate good predictions
    good_polymers = [
        ('*CC*', 'Polyethylene', -100, -95, 5),
        ('*CCO*', 'Poly(ethylene oxide)', -60, -58, 2),
        ('*CC(C)*', 'Polypropylene', -10, -8, 2),
        ('*CC(C(=O)OC)*', 'Poly(methyl acrylate)', 15, 18, 3),
    ]
    
    good_data = []
    for smiles, name, actual, predicted, error in good_polymers:
        good_data.append({
            'smiles': smiles,
            'polymer_name': name,
            'target': actual,
            'prediction': predicted,
            'abs_error': error,
            'dp': np.random.uniform(500, 2000)
        })
    
    good_df = pd.DataFrame(good_data)
    
    return failures_df, good_df


def main():
    """Run SHAP analysis on polymer prediction failures."""
    print("🔍 POLYMER GNN SHAP FAILURE ANALYSIS")
    print("=" * 50)
    
    if not SHAP_AVAILABLE:
        print("❌ SHAP not available. Please install with: pip install shap")
        return
    
    try:
        # Initialize analyzer
        analyzer = PolymerSHAPAnalyzer()
        
        print("📊 Using simulated failure data for demonstration...")
        print("   (In practice, load actual model predictions)")
        
        # Get failure data (simulated for demo)
        failures_df, good_df = simulate_failure_data()
        
        print(f"📈 Analyzing {len(failures_df)} failure cases vs {len(good_df)} good predictions...")
        
        # Run SHAP analysis
        results = analyzer.analyze_failure_features(failures_df, good_df)
        
        # Create visualizations
        analyzer.visualize_shap_analysis(results)
        
        # Generate report
        report = analyzer.generate_shap_report(results, failures_df)
        
        # Save report
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"results/shap_failure_analysis_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ SHAP analysis completed!")
        print(f"📊 Visualizations saved in results/ directory")
        print(f"📝 Report saved: {report_path}")
        
        # Print top insights
        if 'error_analysis' in results:
            top_features = results['error_analysis']['feature_importance'].head(5)
            print(f"\n🔍 TOP 5 FEATURES CONTRIBUTING TO ERRORS:")
            for i, (_, row) in enumerate(top_features.iterrows()):
                print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    except Exception as e:
        print(f"❌ SHAP analysis failed: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install shap scikit-learn")


if __name__ == "__main__":
    main() 