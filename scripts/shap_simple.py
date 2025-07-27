#!/usr/bin/env python3
"""
Simplified SHAP Analysis for Polymer Features
Focuses on analyzing the 148 polymer features for Tg prediction

Usage:
    python shap_simple.py --n_samples 200
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# SHAP imports
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePolymerSHAP:
    """Simplified SHAP analysis using polymer features directly"""

    def __init__(self, dataset_path: str = "data/processed/full_feats.csv"):
        self.dataset_path = Path(dataset_path)
        self.results_dir = Path("results/shap_simple")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        self.df = pd.read_csv(self.dataset_path)
        self.features, self.targets, self.feature_names = self._prepare_data()

        logger.info(f"🔍 Simple SHAP initialized")
        logger.info(f"📊 Dataset: {len(self.df)} samples")
        logger.info(f"🧪 Features: {len(self.feature_names)} polymer features")
        logger.info(f"🎯 Target: Tg prediction")

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare polymer features and targets"""

        logger.info(f"📊 Dataset columns: {list(self.df.columns)[:10]}...")
        logger.info(f"📊 Dataset shape: {self.df.shape}")

        # Skip SMILES and target columns, focus on engineered features
        skip_cols = [
            "canonical_smiles",
            "smiles",
            "Tg",
            "Tm",
            "Density",
            "FFV",
            "Tc",
            "Rg",
            "source",
        ]
        feature_cols = [col for col in self.df.columns if col not in skip_cols]

        logger.info(f"🧪 Found {len(feature_cols)} feature columns")
        logger.info(f"🔬 Sample features: {feature_cols[:5]}")

        # Get targets first
        targets = self.df["Tg"].values
        valid_target_mask = ~pd.isna(targets)

        logger.info(f"🎯 Valid targets: {valid_target_mask.sum()}/{len(targets)}")

        # Get features and convert to numeric
        feature_df = self.df[feature_cols].copy()

        # Convert to numeric, replacing non-numeric with NaN
        numeric_features = feature_df.apply(pd.to_numeric, errors="coerce")

        # Check for valid features
        logger.info(
            f"📊 Numeric conversion: {numeric_features.notna().sum().sum()} valid values"
        )

        features = numeric_features.values

        # Remove rows with NaN values in either features or targets
        valid_feature_mask = ~np.isnan(features).any(axis=1)
        valid_mask = valid_target_mask & valid_feature_mask

        features = features[valid_mask]
        targets = targets[valid_mask]

        logger.info(f"✅ Prepared {len(features)} valid samples")
        if len(features) > 0:
            logger.info(f"🧪 Feature shape: {features.shape}")
            logger.info(
                f"📊 Feature range: {features.min():.3f} to {features.max():.3f}"
            )
            logger.info(
                f"🎯 Target range: {targets.min():.1f}°C to {targets.max():.1f}°C"
            )

        return features, targets, feature_cols

    def train_surrogate_model(self) -> RandomForestRegressor:
        """Train a Random Forest as surrogate model for SHAP analysis"""

        logger.info("🌳 Training Random Forest surrogate model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.targets, test_size=0.2, random_state=42
        )

        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        # Evaluate
        train_pred = rf_model.predict(X_train)
        test_pred = rf_model.predict(X_test)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        logger.info(f"✅ Random Forest trained:")
        logger.info(f"   Train R²: {train_r2:.4f}")
        logger.info(f"   Test R²: {test_r2:.4f}")
        logger.info(f"   Test RMSE: {test_rmse:.2f}°C")

        return rf_model

    def run_shap_analysis(
        self, model: RandomForestRegressor, n_samples: int = 200
    ) -> Dict:
        """Run SHAP analysis on the surrogate model"""

        logger.info(f"🔍 Starting SHAP analysis with {n_samples} samples...")
        start_time = time.time()

        # Sample data for SHAP (for efficiency)
        n_total = len(self.features)
        if n_samples < n_total:
            sample_indices = np.random.choice(n_total, n_samples, replace=False)
            sample_features = self.features[sample_indices]
            sample_targets = self.targets[sample_indices]
        else:
            sample_features = self.features
            sample_targets = self.targets
            sample_indices = np.arange(n_total)

        logger.info(f"📊 Using {len(sample_features)} samples for SHAP analysis")

        # Create SHAP explainer
        logger.info("🔬 Creating TreeExplainer (fast for Random Forest)...")
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values
        logger.info("⚡ Computing SHAP values...")
        shap_values = explainer.shap_values(sample_features)

        analysis_time = time.time() - start_time
        logger.info(f"⏱️  SHAP analysis completed in {analysis_time:.1f} seconds")

        # Analyze results
        results = self._analyze_shap_results(
            shap_values, sample_features, sample_targets, model
        )

        # Add metadata
        results["metadata"] = {
            "n_samples": len(sample_features),
            "n_features": len(self.feature_names),
            "analysis_time_seconds": analysis_time,
            "model_type": "RandomForestRegressor",
            "feature_names": self.feature_names,
        }

        # Save results
        results_file = self.results_dir / "shap_simple_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 SHAP results saved to {results_file}")

        return results, shap_values, sample_features

    def _analyze_shap_results(
        self, shap_values: np.ndarray, features: np.ndarray, targets: np.ndarray, model
    ) -> Dict:
        """Analyze SHAP results"""

        logger.info("📊 Analyzing SHAP results...")

        # Feature importance analysis
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feature_importance = list(zip(self.feature_names, mean_abs_shap))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        # Top features
        top_n = 30
        top_features = feature_importance[:top_n]

        logger.info(f"🏆 Top {min(10, top_n)} most important features:")
        for i, (feat_name, importance) in enumerate(top_features[:10]):
            logger.info(f"   {i+1:2d}. {feat_name:<35} | Importance: {importance:.4f}")

        # Feature category analysis
        categories = self._categorize_features(top_features)

        # Model performance check
        predictions = model.predict(features)
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))

        results = {
            "feature_importance": {
                "top_features": [
                    {"name": name, "importance": float(imp)}
                    for name, imp in top_features
                ],
                "all_features": [
                    {"name": name, "importance": float(imp)}
                    for name, imp in feature_importance
                ],
            },
            "feature_categories": categories,
            "model_performance": {
                "r2": float(r2),
                "rmse": float(rmse),
                "n_samples": len(features),
            },
            "shap_statistics": {
                "mean_shap_magnitude": float(np.mean(np.abs(shap_values))),
                "max_shap_value": float(np.max(shap_values)),
                "min_shap_value": float(np.min(shap_values)),
                "shap_std": float(np.std(shap_values)),
            },
        }

        return results

    def _categorize_features(self, top_features: List[Tuple[str, float]]) -> Dict:
        """Categorize features by type"""

        categories = {
            "molecular_descriptors": [],
            "fingerprint_bits": [],
            "polymer_properties": [],
            "structural_features": [],
            "other": [],
        }

        for feat_name, importance in top_features:
            name_lower = feat_name.lower()

            if any(
                keyword in name_lower
                for keyword in ["molecular", "logp", "tpsa", "hbd", "hba", "rotatable"]
            ):
                categories["molecular_descriptors"].append(
                    {"name": feat_name, "importance": importance}
                )
            elif any(
                keyword in name_lower
                for keyword in ["fp_bit", "fingerprint", "ecfp", "morgan"]
            ):
                categories["fingerprint_bits"].append(
                    {"name": feat_name, "importance": importance}
                )
            elif any(
                keyword in name_lower
                for keyword in ["dp", "mw", "polymer", "chain", "flex", "degree"]
            ):
                categories["polymer_properties"].append(
                    {"name": feat_name, "importance": importance}
                )
            elif any(
                keyword in name_lower
                for keyword in ["aromatic", "ring", "bond", "atom"]
            ):
                categories["structural_features"].append(
                    {"name": feat_name, "importance": importance}
                )
            else:
                categories["other"].append(
                    {"name": feat_name, "importance": importance}
                )

        return categories

    def create_visualizations(
        self, shap_values: np.ndarray, features: np.ndarray, results: Dict
    ):
        """Create SHAP visualizations"""

        logger.info("🎨 Creating SHAP visualizations...")

        # Set style
        plt.style.use("default")
        sns.set_palette("viridis")

        # 1. Feature importance bar plot
        top_features = results["feature_importance"]["top_features"][:20]

        plt.figure(figsize=(12, 8))
        names = [f["name"] for f in top_features]
        importances = [f["importance"] for f in top_features]

        # Truncate long names for better display
        display_names = [
            name[:30] + "..." if len(name) > 30 else name for name in names
        ]

        bars = plt.barh(range(len(display_names)), importances, alpha=0.8)
        plt.yticks(range(len(display_names)), display_names)
        plt.xlabel("Mean |SHAP Value|")
        plt.title(
            "Top 20 Features by SHAP Importance for Tg Prediction", fontsize=14, pad=20
        )
        plt.gca().invert_yaxis()

        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(importances)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.tight_layout()
        importance_path = self.results_dir / "feature_importance.png"
        plt.savefig(importance_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"📊 Feature importance plot saved to {importance_path}")

        # 2. Summary plot using SHAP
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values,
            features,
            feature_names=self.feature_names,
            max_display=20,
            show=False,
        )
        plt.title(
            "SHAP Summary Plot - Polymer Features for Tg Prediction",
            fontsize=14,
            pad=20,
        )
        plt.tight_layout()
        summary_path = self.results_dir / "shap_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"📊 SHAP summary plot saved to {summary_path}")

        # 3. Feature category pie chart
        categories = results["feature_categories"]
        category_counts = {k: len(v) for k, v in categories.items() if v}

        if category_counts:
            plt.figure(figsize=(10, 8))
            labels = list(category_counts.keys())
            sizes = list(category_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

            plt.pie(
                sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
            )
            plt.title(
                "Distribution of Top 30 Features by Category", fontsize=14, pad=20
            )
            plt.axis("equal")

            category_path = self.results_dir / "feature_categories.png"
            plt.savefig(category_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"📊 Category distribution plot saved to {category_path}")

        # 4. SHAP values distribution
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(
            shap_values.flatten(),
            bins=50,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        plt.xlabel("SHAP Values")
        plt.ylabel("Frequency")
        plt.title("Distribution of SHAP Values")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        shap_magnitude = np.sum(np.abs(shap_values), axis=1)
        plt.hist(
            shap_magnitude, bins=30, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        plt.xlabel("Total |SHAP| per Sample")
        plt.ylabel("Frequency")
        plt.title("Distribution of SHAP Magnitude")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        dist_path = self.results_dir / "shap_distribution.png"
        plt.savefig(dist_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"📊 SHAP distribution plots saved to {dist_path}")

    def print_analysis_summary(self, results: Dict):
        """Print comprehensive analysis summary"""

        print("\n" + "=" * 80)
        print("🔍 POLYMER FEATURE SHAP ANALYSIS SUMMARY")
        print("=" * 80)

        metadata = results["metadata"]
        performance = results["model_performance"]
        categories = results["feature_categories"]

        print(f"\n📊 ANALYSIS OVERVIEW:")
        print(f"   Samples analyzed: {metadata['n_samples']}")
        print(f"   Total features: {metadata['n_features']}")
        print(f"   Model: {metadata['model_type']}")
        print(f"   Analysis time: {metadata['analysis_time_seconds']:.1f} seconds")

        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"   R²: {performance['r2']:.4f}")
        print(f"   RMSE: {performance['rmse']:.2f}°C")

        top_features = results["feature_importance"]["top_features"]
        print(f"\n🏆 TOP 15 MOST IMPORTANT FEATURES:")
        for i, feat in enumerate(top_features[:15]):
            print(f"   {i+1:2d}. {feat['name']:<40} | {feat['importance']:.4f}")

        print(f"\n🧪 FEATURE CATEGORIES IN TOP 30:")
        for category, features in categories.items():
            if features:
                category_name = category.replace("_", " ").title()
                print(f"   {category_name}: {len(features)} features")
                if len(features) <= 3:
                    examples = [f["name"] for f in features]
                else:
                    examples = [f["name"] for f in features[:3]] + ["..."]
                print(f"      Examples: {', '.join(examples)}")

        # Scientific insights
        print(f"\n💡 SCIENTIFIC INSIGHTS:")

        # Check for specific important feature types
        molecular_features = categories.get("molecular_descriptors", [])
        polymer_features = categories.get("polymer_properties", [])
        fingerprint_features = categories.get("fingerprint_bits", [])

        if molecular_features:
            print(f"   🧬 Molecular descriptors are key drivers of Tg prediction")
            print(f"      • {len(molecular_features)} molecular features in top 30")

        if polymer_features:
            print(f"   🔗 Polymer-specific properties significantly influence Tg")
            print(f"      • {len(polymer_features)} polymer features in top 30")

        if fingerprint_features:
            print(f"   🏗️  Structural fingerprints capture important molecular patterns")
            print(f"      • {len(fingerprint_features)} fingerprint features in top 30")

        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   • Focus feature engineering on top-ranked feature types")
        print(f"   • Validate findings with domain knowledge about polymer Tg")
        print(f"   • Use SHAP insights to guide future model improvements")
        print(f"   • Consider feature selection based on SHAP importance")

        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Simple SHAP Analysis for Polymer Features"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/full_feats.csv",
        help="Path to dataset with polymer features",
    )
    parser.add_argument(
        "--n_samples", type=int, default=200, help="Number of samples for SHAP analysis"
    )

    args = parser.parse_args()

    # Validate dataset
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        sys.exit(1)

    # Create analyzer
    analyzer = SimplePolymerSHAP(args.dataset)

    # Train surrogate model
    model = analyzer.train_surrogate_model()

    # Run SHAP analysis
    results, shap_values, features = analyzer.run_shap_analysis(model, args.n_samples)

    # Create visualizations
    analyzer.create_visualizations(shap_values, features, results)

    # Print summary
    analyzer.print_analysis_summary(results)

    return results


if __name__ == "__main__":
    main()
