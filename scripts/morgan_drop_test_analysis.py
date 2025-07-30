#!/usr/bin/env python3
"""
Morgan Fingerprint Drop Test Analysis

Tests the impact of excluding Morgan fingerprint features (128 features) on 
polymer Tg prediction performance and SHAP feature importance rankings.
"""

import sys

sys.path.insert(0, "src")
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.features.polymer_features import PolymerFeatureExtractor

warnings.filterwarnings("ignore")


def run_morgan_drop_test():
    """Run analysis without Morgan fingerprint features."""
    print("🧪 Morgan Fingerprint Drop Test Analysis")
    print("=" * 50)

    # Load dataset
    print("📊 Loading polymer dataset...")
    df = pd.read_csv("data/processed/filtered_tg_dataset.csv")
    print(f"   Dataset size: {len(df)} samples")

    # Initialize feature extractor (Morgan fingerprints already excluded)
    feature_extractor = PolymerFeatureExtractor()
    print(
        f"   Slim feature set: {feature_extractor.get_feature_dim()} features (Morgan FPs excluded)"
    )

    # Extract features
    print("🔧 Extracting features from SMILES...")
    features_list = []
    targets = []
    valid_smiles = []
    failed_count = 0

    for idx, row in df.iterrows():
        try:
            smiles = row["processed_smiles"]
            target = row["Tg"]

            if pd.notna(smiles) and pd.notna(target):
                features = feature_extractor.extract_features(smiles)
                features_list.append(features.numpy())
                targets.append(target)
                valid_smiles.append(smiles)
        except Exception as e:
            failed_count += 1
            continue

    features_array = np.array(features_list)
    targets_array = np.array(targets)

    print(f"✅ Processed: {len(features_array)} valid samples")
    print(f"   Failed: {failed_count} samples")
    print(f"   Feature matrix shape: {features_array.shape}")

    # Train/test split (same random state as original)
    X_train, X_test, y_train, y_test = train_test_split(
        features_array, targets_array, test_size=0.2, random_state=42
    )

    # Train Random Forest (200 trees, same as original)
    print("🌲 Training Random Forest (200 trees)...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # Evaluate performance
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print("📊 Morgan Drop Test Performance:")
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.2f}°C")
    print(f"   Test MAE: {test_mae:.2f}°C")

    # Compare with full run results (from previous analysis)
    full_run_r2 = 0.6319
    full_run_rmse = 70.12
    full_run_mae = 53.66

    print("\n📈 Comparison vs Full Run (with Morgan FPs):")
    print(
        f"   R² change: {test_r2:.4f} vs {full_run_r2:.4f} = {test_r2 - full_run_r2:+.4f}"
    )
    print(
        f"   RMSE change: {test_rmse:.2f} vs {full_run_rmse:.2f} = {test_rmse - full_run_rmse:+.2f}°C"
    )
    print(
        f"   MAE change: {test_mae:.2f} vs {full_run_mae:.2f} = {test_mae - full_run_mae:+.2f}°C"
    )

    # SHAP Analysis
    print("🔍 Computing SHAP values for slim feature set...")
    explainer = shap.TreeExplainer(rf_model)

    # Use same sample size as original
    sample_size = min(96, len(X_test))
    X_sample = X_test[:sample_size]
    shap_values = explainer.shap_values(X_sample)

    print(f"✅ SHAP values computed for {sample_size} samples")
    print(f"   SHAP values shape: {shap_values.shape}")

    # Feature importance analysis
    feature_names = feature_extractor.get_feature_names()
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_std = np.abs(shap_values).std(axis=0)

    # Create importance DataFrame
    importance_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance_mean": feature_importance,
            "importance_std": feature_std,
            "abs_importance": feature_importance,
        }
    ).sort_values("abs_importance", ascending=False)

    # Assign feature groups (without Morgan fingerprints)
    importance_df["feature_group"] = "Other"
    for idx, row in importance_df.iterrows():
        name = row["feature_name"].lower()
        if "chain" in name or "flexibility" in name:
            importance_df.at[idx, "feature_group"] = "Chain Descriptors"
        elif "molecular_weight" in name or "mw" in name:
            importance_df.at[idx, "feature_group"] = "Molecular Weight"
        elif "degree_polymerization" in name or "dp" in name:
            importance_df.at[idx, "feature_group"] = "Degree of Polymerization"
        elif any(
            desc in name
            for desc in ["logp", "tpsa", "hbd", "hba", "rotatable", "aromatic"]
        ):
            importance_df.at[idx, "feature_group"] = "Molecular Descriptors"
        elif "complexity" in name or "bertz" in name:
            importance_df.at[idx, "feature_group"] = "Complexity"

    # Group analysis
    group_importance = (
        importance_df.groupby("feature_group")
        .agg({"importance_mean": ["sum", "mean", "count"], "importance_std": "mean"})
        .round(4)
    )

    group_importance.columns = [
        "total_importance",
        "mean_importance",
        "feature_count",
        "mean_std",
    ]
    group_importance = group_importance.sort_values("total_importance", ascending=False)

    print("\n🔬 Feature Group Importance (Morgan FPs excluded):")
    for group, data in group_importance.iterrows():
        print(
            f'   {group}: {data["total_importance"]:.4f} (n={int(data["feature_count"])})'
        )

    print("\n🏆 Top 10 Features (Morgan FPs excluded):")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(
            f'   {i:2d}. {row["feature_name"]:25s} | {row["importance_mean"]:8.4f} | {row["feature_group"]}'
        )

    # Calculate efficiency metrics
    total_importance = importance_df["importance_mean"].sum()
    cumulative_importance = (
        np.cumsum(importance_df["importance_mean"]) / total_importance
    )
    features_80_pct = np.where(cumulative_importance >= 0.8)[0][0] + 1
    features_90_pct = np.where(cumulative_importance >= 0.9)[0][0] + 1

    print(f"\n💡 Feature Selection Insights:")
    print(
        f"   80% importance captured by top {features_80_pct} features ({features_80_pct/20*100:.1f}% of slim set)"
    )
    print(
        f"   90% importance captured by top {features_90_pct} features ({features_90_pct/20*100:.1f}% of slim set)"
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save detailed CSV
    csv_path = results_dir / f"morgan_drop_test_importance_{timestamp}.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"\n📁 Results saved to: {csv_path}")

    # Generate summary report
    report_path = results_dir / f"morgan_drop_test_report_{timestamp}.md"

    report_content = f"""# Morgan Fingerprint Drop Test Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Summary
Excluded 128 Morgan fingerprint features to test their contribution to Tg prediction.

## Performance Impact
| Metric | Full Run (148 features) | Drop Test (20 features) | Change |
|--------|------------------------|-------------------------|---------|
| Test R² | {full_run_r2:.4f} | {test_r2:.4f} | {test_r2 - full_run_r2:+.4f} |
| Test RMSE | {full_run_rmse:.2f}°C | {test_rmse:.2f}°C | {test_rmse - full_run_rmse:+.2f}°C |
| Test MAE | {full_run_mae:.2f}°C | {test_mae:.2f}°C | {test_mae - full_run_mae:+.2f}°C |

## Key Findings

### Performance Assessment
- **R² Impact**: {"Minimal degradation" if abs(test_r2 - full_run_r2) < 0.05 else "Significant degradation"}
- **Morgan FP Value**: {"Low" if abs(test_r2 - full_run_r2) < 0.05 else "High"} - collectively contributed {'<5%' if abs(test_r2 - full_run_r2) < 0.05 else '>5%'} to predictive power

### Feature Group Dominance (without Morgan FPs)
"""

    for i, (group, data) in enumerate(group_importance.iterrows(), 1):
        report_content += f"{i}. **{group}**: {data['total_importance']:.4f} total importance ({int(data['feature_count'])} features)\\n"

    report_content += f"""

### Top Features (without Morgan FPs)
"""

    for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
        report_content += f"{i}. **{row['feature_name']}**: {row['importance_mean']:.4f} ({row['feature_group']})\\n"

    # Analyze group behavior changes
    chain_desc_before = 60.5324  # From full run
    chain_desc_after = (
        group_importance.loc["Chain Descriptors", "total_importance"]
        if "Chain Descriptors" in group_importance.index
        else 0
    )

    report_content += f"""

## Group Behavior Analysis
- **Chain Descriptors**: {"Spiked" if chain_desc_after > chain_desc_before * 1.1 else "Stable"} from {chain_desc_before:.2f} to {chain_desc_after:.2f}
- **Feature Efficiency**: Top {features_80_pct} features capture 80% of importance in slim set

## Conclusion
Morgan fingerprints proved to be {"mostly fluff" if abs(test_r2 - full_run_r2) < 0.05 else "valuable contributors"} - 
removing 128 features caused {"minimal" if abs(test_r2 - full_run_r2) < 0.05 else "significant"} performance degradation.

**Recommendation**: {"Focus on core polymer physics features" if abs(test_r2 - full_run_r2) < 0.05 else "Retain Morgan fingerprints for optimal performance"}
"""

    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"📄 Report saved to: {report_path}")

    return {
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "importance_df": importance_df,
        "group_importance": group_importance,
    }


if __name__ == "__main__":
    results = run_morgan_drop_test()
