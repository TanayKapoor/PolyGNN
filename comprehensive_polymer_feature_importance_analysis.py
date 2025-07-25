#!/usr/bin/env python3
"""
Comprehensive SHAP Feature Importance Analysis for 147 Polymer Features

This script provides detailed feature importance analysis using SHAP on the
147 polymer-specific features for glass transition temperature (Tg) prediction.
"""

import sys
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.features.polymer_features import PolymerFeatureExtractor
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data():
    """Load polymer dataset and extract features."""
    print('📊 Loading polymer dataset...')
    df = pd.read_csv('data/processed/filtered_tg_dataset.csv')
    print(f'   Initial dataset size: {len(df)} samples')
    
    feature_extractor = PolymerFeatureExtractor()
    print(f'   Feature extractor: {feature_extractor.get_feature_dim()} features')
    
    features_list = []
    targets = []
    valid_smiles = []
    failed_count = 0
    
    for idx, row in df.iterrows():
        try:
            smiles = row['processed_smiles']
            target = row['Tg']
            
            if pd.notna(smiles) and pd.notna(target):
                features = feature_extractor.extract_features(smiles)
                features_list.append(features.numpy())
                targets.append(target)
                valid_smiles.append(smiles)
        except Exception as e:
            failed_count += 1
            continue
    
    print(f'   Successfully processed: {len(features_list)} samples')
    print(f'   Failed processing: {failed_count} samples')
    
    return np.array(features_list), np.array(targets), valid_smiles, feature_extractor

def train_surrogate_model(X, y):
    """Train Random Forest surrogate model for SHAP analysis."""
    print('🔧 Training surrogate model...')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f'✅ Surrogate model performance:')
    print(f'   Train R²: {train_r2:.4f}')
    print(f'   Test R²: {test_r2:.4f}')
    print(f'   Test RMSE: {test_rmse:.2f}°C')
    print(f'   Test MAE: {test_mae:.2f}°C')
    
    return rf_model, X_train, X_test, y_train, y_test

def compute_shap_analysis(model, X_train, X_test, feature_extractor):
    """Compute comprehensive SHAP analysis."""
    print('🔍 Computing SHAP values...')
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for test set (sample for efficiency)
    sample_size = min(100, len(X_test))
    X_sample = X_test[:sample_size]
    shap_values = explainer.shap_values(X_sample)
    
    print(f'✅ SHAP values computed for {sample_size} samples')
    print(f'   SHAP values shape: {shap_values.shape}')
    
    # Get feature names
    feature_names = feature_extractor.get_feature_names()
    
    # Compute feature importance metrics
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_std = np.abs(shap_values).std(axis=0)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance_mean': feature_importance,
        'importance_std': feature_std,
        'abs_importance': feature_importance
    }).sort_values('abs_importance', ascending=False)
    
    return shap_values, importance_df, X_sample, explainer

def analyze_feature_groups(importance_df):
    """Analyze importance by feature groups."""
    print('🔬 Analyzing feature groups...')
    
    # Define feature groups based on naming patterns
    feature_groups = {
        'Molecular Weight': lambda x: 'molecular_weight' in x.lower() or 'mw' in x.lower(),
        'Degree of Polymerization': lambda x: 'degree_polymerization' in x.lower() or 'dp' in x.lower(),
        'Morgan Fingerprint': lambda x: 'morgan_fp' in x.lower() or 'fp_bit' in x.lower(),
        'Chain Descriptors': lambda x: 'chain' in x.lower() or 'flexibility' in x.lower(),
        'Molecular Descriptors': lambda x: any(desc in x.lower() for desc in 
                                               ['logp', 'tpsa', 'hbd', 'hba', 'rotatable', 'aromatic']),
        'Complexity': lambda x: 'complexity' in x.lower() or 'bertz' in x.lower(),
        'Other': lambda x: True  # Default category
    }
    
    # Assign features to groups
    importance_df['feature_group'] = 'Other'
    for group_name, condition in feature_groups.items():
        if group_name != 'Other':
            mask = importance_df['feature_name'].apply(condition)
            importance_df.loc[mask, 'feature_group'] = group_name
    
    # Calculate group importance
    group_importance = importance_df.groupby('feature_group').agg({
        'importance_mean': ['sum', 'mean', 'count'],
        'importance_std': 'mean'
    }).round(4)
    
    group_importance.columns = ['total_importance', 'mean_importance', 'feature_count', 'mean_std']
    group_importance = group_importance.sort_values('total_importance', ascending=False)
    
    print('📊 Feature Group Importance:')
    for group, data in group_importance.iterrows():
        print(f'   {group}: {data["total_importance"]:.4f} (n={data["feature_count"]})')
    
    return group_importance

def create_visualizations(shap_values, importance_df, X_sample, feature_extractor, explainer):
    """Create comprehensive visualizations."""
    print('📈 Creating visualizations...')
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 25))
    
    # 1. Top 20 Feature Importance Bar Plot
    ax1 = plt.subplot(4, 2, 1)
    top_20 = importance_df.head(20)
    bars = ax1.barh(range(len(top_20)), top_20['importance_mean'], 
                    xerr=top_20['importance_std'], capsize=3)
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                         for name in top_20['feature_name']], fontsize=8)
    ax1.set_xlabel('Mean |SHAP value|')
    ax1.set_title('Top 20 Most Important Features', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Color bars by importance
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 2. Feature Group Importance
    ax2 = plt.subplot(4, 2, 2)
    group_importance = importance_df.groupby('feature_group')['importance_mean'].sum().sort_values(ascending=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(group_importance)))
    bars = ax2.barh(range(len(group_importance)), group_importance.values)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    ax2.set_yticks(range(len(group_importance)))
    ax2.set_yticklabels(group_importance.index, fontsize=10)
    ax2.set_xlabel('Total Group Importance')
    ax2.set_title('Feature Group Importance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. SHAP Summary Plot (top features)
    ax3 = plt.subplot(4, 1, 3)
    feature_names = feature_extractor.get_feature_names()
    top_features_idx = importance_df.head(20).index
    shap.summary_plot(shap_values[:, top_features_idx], 
                     X_sample[:, top_features_idx],
                     feature_names=[feature_names[i] for i in top_features_idx],
                     show=False, max_display=20)
    plt.title('SHAP Summary Plot - Top 20 Features', fontsize=12, fontweight='bold')
    
    # 4. Feature Importance Distribution
    ax4 = plt.subplot(4, 2, 7)
    ax4.hist(importance_df['importance_mean'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Feature Importance')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Feature Importance', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Cumulative Importance
    ax5 = plt.subplot(4, 2, 8)
    cumulative_importance = np.cumsum(importance_df['importance_mean']) / np.sum(importance_df['importance_mean'])
    ax5.plot(range(len(cumulative_importance)), cumulative_importance, 'b-', linewidth=2)
    ax5.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% threshold')
    ax5.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    ax5.set_xlabel('Number of Features (ranked)')
    ax5.set_ylabel('Cumulative Importance')
    ax5.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    plot_path = results_dir / f'polymer_147_features_shap_analysis_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'📊 Visualizations saved to: {plot_path}')
    
    return plot_path

def generate_report(importance_df, group_importance, model_performance):
    """Generate comprehensive analysis report."""
    print('📝 Generating comprehensive report...')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    csv_path = results_dir / f'polymer_147_features_importance_{timestamp}.csv'
    importance_df.to_csv(csv_path, index=False)
    
    # Generate summary report
    report_path = results_dir / f'polymer_147_features_analysis_report_{timestamp}.md'
    
    report_content = f"""# Polymer Feature Importance Analysis Report
## SHAP Analysis of 147 Polymer-Specific Features for Tg Prediction

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This analysis evaluated the importance of 147 polymer-specific features for predicting glass transition temperature (Tg) using SHAP (SHapley Additive exPlanations) values.

## Dataset Summary
- **Total samples processed:** {len(importance_df)} features analyzed
- **Target property:** Glass transition temperature (Tg)
- **Feature extraction:** 147 polymer-specific features including molecular weight, degree of polymerization, Morgan fingerprints, and molecular descriptors

## Model Performance
- **Algorithm:** Random Forest Regressor (200 trees)
- **Test R² Score:** {model_performance.get('test_r2', 'N/A'):.4f}
- **Test RMSE:** {model_performance.get('test_rmse', 'N/A'):.2f}°C
- **Test MAE:** {model_performance.get('test_mae', 'N/A'):.2f}°C

## Top 10 Most Important Features

| Rank | Feature Name | SHAP Importance | Std Dev | Feature Group |
|------|-------------|----------------|---------|---------------|
"""
    
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        report_content += f"| {i} | {row['feature_name']} | {row['importance_mean']:.4f} | {row['importance_std']:.4f} | {row['feature_group']} |\n"
    
    report_content += f"""
## Feature Group Analysis

| Feature Group | Total Importance | Average Importance | Feature Count |
|---------------|-----------------|-------------------|---------------|
"""
    
    for group, data in group_importance.iterrows():
        report_content += f"| {group} | {data['total_importance']:.4f} | {data['mean_importance']:.4f} | {int(data['feature_count'])} |\n"
    
    # Feature selection recommendations
    features_80_pct = np.where(np.cumsum(importance_df['importance_mean']) / np.sum(importance_df['importance_mean']) >= 0.8)[0][0] + 1
    features_90_pct = np.where(np.cumsum(importance_df['importance_mean']) / np.sum(importance_df['importance_mean']) >= 0.9)[0][0] + 1
    
    report_content += f"""
## Feature Selection Recommendations

### Dimensionality Reduction Opportunities
- **80% of importance** captured by top **{features_80_pct}** features ({features_80_pct/147*100:.1f}% of total)
- **90% of importance** captured by top **{features_90_pct}** features ({features_90_pct/147*100:.1f}% of total)

### Key Insights
1. **Most Predictive Group:** {group_importance.index[0]} (highest total importance)
2. **Feature Efficiency:** Top {features_80_pct} features provide 80% of predictive power
3. **Redundancy:** {147 - features_90_pct} features contribute <10% to predictions

### Actionable Recommendations
1. **For production models:** Use top {features_80_pct} features for computational efficiency
2. **For research:** Focus on {group_importance.index[0]} and {group_importance.index[1]} feature groups
3. **Feature engineering:** Investigate combinations of top-performing features

## Technical Details
- **SHAP Method:** TreeExplainer with Random Forest
- **Validation:** 80/20 train-test split
- **Feature Extraction:** Polymer-specific descriptors including:
  - Molecular weight of repeating units
  - Degree of polymerization encoding
  - Morgan fingerprints (128-bit)
  - Chain flexibility descriptors
  - Molecular complexity measures

## Files Generated
- Feature importance CSV: `{csv_path.name}`
- Visualization plots: `polymer_147_features_shap_analysis_{timestamp}.png`
- This report: `{report_path.name}`

---
*Analysis conducted using SHAP v{shap.__version__} and scikit-learn*
"""
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f'📄 Report saved to: {report_path}')
    print(f'📊 Data saved to: {csv_path}')
    
    return report_path, csv_path

def main():
    """Main analysis pipeline."""
    print('🚀 Starting Comprehensive SHAP Analysis of 147 Polymer Features')
    print('=' * 70)
    
    # Load and process data
    X, y, smiles, feature_extractor = load_and_process_data()
    
    # Train surrogate model
    model, X_train, X_test, y_train, y_test = train_surrogate_model(X, y)
    
    # Store model performance
    y_pred_test = model.predict(X_test)
    model_performance = {
        'test_r2': r2_score(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    
    # Compute SHAP analysis
    shap_values, importance_df, X_sample, explainer = compute_shap_analysis(
        model, X_train, X_test, feature_extractor
    )
    
    # Analyze feature groups
    group_importance = analyze_feature_groups(importance_df)
    
    # Create visualizations
    plot_path = create_visualizations(
        shap_values, importance_df, X_sample, feature_extractor, explainer
    )
    
    # Generate report
    report_path, csv_path = generate_report(
        importance_df, group_importance, model_performance
    )
    
    print('\n✅ Analysis Complete!')
    print(f'📊 Key findings:')
    print(f'   • Top feature: {importance_df.iloc[0]["feature_name"]}')
    print(f'   • Most important group: {group_importance.index[0]}')
    print(f'   • 80% importance captured by top {np.where(np.cumsum(importance_df["importance_mean"]) / np.sum(importance_df["importance_mean"]) >= 0.8)[0][0] + 1} features')
    print(f'📁 Results saved in results/ directory')

if __name__ == '__main__':
    main()