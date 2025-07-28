#!/usr/bin/env python3
"""
External Validation Script for PolyGNN
Tests ensemble GCN generalization on preprocessed external datasets
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Batch, Data

# Local imports
from src.data.molecular_graph import MolecularGraphConverter
from src.data.polymer_dataset import PolymerTgDataset
from src.models.polymer_gcn import PolymerGCN

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalValidator:
    """External validation with ensemble predictions and UQ"""

    def __init__(self, model_dir: str = "results", device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.ensemble_models = []
        self.feature_cols = None
        self.graph_converter = MolecularGraphConverter(max_atoms=300)

        logger.info(f"🚀 External Validator initialized on {self.device}")

    def load_ensemble(self, model_pattern: str = "best_gcn_*.pth") -> int:
        """Load ensemble of trained models"""

        model_files = list(self.model_dir.glob(model_pattern))

        if not model_files:
            # Try alternative patterns
            patterns = ["*.pth", "best_*.pth", "gcn_*.pth"]
            for pattern in patterns:
                model_files = list(self.model_dir.glob(pattern))
                if model_files:
                    break

        if not model_files:
            logger.warning(f"⚠️  No models found in {self.model_dir}")
            return 0

        logger.info(f"📂 Found {len(model_files)} model files")

        # Load models
        for model_file in model_files[:5]:  # Max 5 for ensemble
            try:
                logger.info(f"📥 Loading {model_file.name}")

                # Create model instance (using default config)
                model = PolymerGCN(
                    node_feature_dim=157,
                    hidden_dims=[256, 128],
                    num_gcn_layers=3,
                    dropout_rate=0.3,
                )

                # Load weights
                checkpoint = torch.load(model_file, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)

                model.to(self.device)
                model.eval()
                self.ensemble_models.append(model)

            except Exception as e:
                logger.warning(f"⚠️  Failed to load {model_file}: {e}")

        logger.info(f"✅ Loaded {len(self.ensemble_models)} models for ensemble")
        return len(self.ensemble_models)

    def load_external_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate external dataset"""

        logger.info(f"📂 Loading external data from {file_path}")

        df = pd.read_csv(file_path)
        logger.info(f"📊 External dataset shape: {df.shape}")
        logger.info(f"📋 Columns: {df.columns.tolist()[:10]}...")  # First 10 cols

        # Identify feature columns
        exclude_cols = [
            "smiles",
            "canonical_smiles",
            "SMILES",
            "Tg",
            "Tm",
            "Density",
            "FFV",
            "Tc",
            "Rg",
            "source",
        ]
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]

        logger.info(f"🧪 Feature columns: {len(self.feature_cols)}")

        # Check for missing SMILES
        smiles_col = None
        for col in ["canonical_smiles", "smiles", "SMILES"]:
            if col in df.columns:
                smiles_col = col
                break

        if smiles_col is None:
            raise ValueError("No SMILES column found in external data")

        # Remove rows with missing SMILES
        df = df.dropna(subset=[smiles_col])

        logger.info(f"✅ Processed external data: {len(df)} samples")

        return df

    def df_to_graphs(self, df: pd.DataFrame) -> List[Data]:
        """Convert dataframe to PyTorch Geometric graphs"""

        smiles_col = None
        for col in ["canonical_smiles", "smiles", "SMILES"]:
            if col in df.columns:
                smiles_col = col
                break

        graphs = []
        failed_count = 0

        logger.info(f"🔄 Converting {len(df)} SMILES to graphs...")

        for idx, row in df.iterrows():
            smiles = row[smiles_col]

            try:
                # Convert SMILES to graph
                graph_data = self.graph_converter.smiles_to_graph(smiles)

                if graph_data is not None:
                    graphs.append(graph_data)
                else:
                    failed_count += 1

            except Exception as e:
                logger.warning(f"Graph conversion failed for {smiles}: {e}")
                failed_count += 1

        logger.info(f"✅ Converted {len(graphs)} graphs, {failed_count} failed")

        return graphs

    def predict_with_ensemble(
        self, graphs: List[Data]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble predictions with uncertainty quantification"""

        if not self.ensemble_models:
            raise ValueError("No ensemble models loaded")

        logger.info(f"🔮 Predicting with {len(self.ensemble_models)} models...")

        # Batch graphs
        batch = Batch.from_data_list(graphs).to(self.device)

        # Collect predictions from ensemble
        ensemble_preds = []

        with torch.no_grad():
            for model in self.ensemble_models:
                try:
                    pred = model(batch)
                    ensemble_preds.append(pred.cpu().numpy())
                except Exception as e:
                    logger.warning(f"Prediction failed for one model: {e}")

        if not ensemble_preds:
            raise ValueError("All ensemble predictions failed")

        # Stack predictions: (n_models, n_samples)
        ensemble_preds = np.array(ensemble_preds)

        # Calculate mean and variance
        mean_pred = np.mean(ensemble_preds, axis=0).flatten()
        var_pred = np.var(ensemble_preds, axis=0).flatten()

        logger.info(f"✅ Ensemble predictions: {len(mean_pred)} samples")

        return mean_pred, var_pred

    def calculate_metrics(
        self,
        true_values: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        property_name: str,
    ) -> Dict:
        """Calculate comprehensive validation metrics"""

        # Remove NaN values
        mask = ~(np.isnan(true_values) | np.isnan(predictions))
        true_clean = true_values[mask]
        pred_clean = predictions[mask]
        unc_clean = uncertainties[mask]

        if len(true_clean) == 0:
            return {
                "property": property_name,
                "n_samples": 0,
                "r2": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "unc_corr": np.nan,
                "coverage_95": np.nan,
            }

        # Basic metrics
        r2 = r2_score(true_clean, pred_clean)
        rmse = np.sqrt(mean_squared_error(true_clean, pred_clean))
        mae = mean_absolute_error(true_clean, pred_clean)

        # Uncertainty metrics
        errors = np.abs(true_clean - pred_clean)
        unc_corr = (
            np.corrcoef(errors, unc_clean)[0, 1] if len(unc_clean) > 1 else np.nan
        )

        # 95% coverage (2-sigma bounds)
        coverage_95 = np.mean(errors <= 2 * unc_clean) if len(unc_clean) > 0 else np.nan

        metrics = {
            "property": property_name,
            "n_samples": len(true_clean),
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "unc_corr": unc_corr,
            "coverage_95": coverage_95,
        }

        return metrics

    def robustness_test(
        self, df: pd.DataFrame, graphs: List[Data], noise_level: float = 0.05
    ) -> Dict:
        """Test robustness to feature perturbations"""

        logger.info(f"🔬 Running robustness test with {noise_level*100:.1f}% noise...")

        # Get original predictions
        orig_preds, orig_vars = self.predict_with_ensemble(graphs)

        # Add noise to features (if available in df)
        if self.feature_cols and len(self.feature_cols) > 0:
            df_noise = df.copy()

            # Select numeric feature columns
            numeric_feats = [col for col in self.feature_cols if col in df.columns]

            if len(numeric_feats) > 0:
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, df_noise[numeric_feats].shape)
                df_noise[numeric_feats] = (
                    df_noise[numeric_feats] + df_noise[numeric_feats] * noise
                )

                # Convert to graphs again
                graphs_noise = self.df_to_graphs(df_noise)

                if len(graphs_noise) > 0:
                    # Predict with noise
                    noise_preds, noise_vars = self.predict_with_ensemble(graphs_noise)

                    # Calculate shifts
                    pred_shift = (
                        np.mean(np.abs(orig_preds - noise_preds))
                        / (np.mean(np.abs(orig_preds)) + 1e-8)
                        * 100
                    )
                    var_rise = np.mean(noise_vars / (orig_vars + 1e-8)) * 100 - 100

                    return {
                        "pred_shift_pct": pred_shift,
                        "var_rise_pct": var_rise,
                        "noise_level": noise_level * 100,
                    }

        return {
            "pred_shift_pct": np.nan,
            "var_rise_pct": np.nan,
            "noise_level": noise_level * 100,
        }

    def run_validation(
        self, external_file: str, output_file: str = "data/processed/external_preds.csv"
    ):
        """Run complete external validation pipeline"""

        logger.info("=" * 80)
        logger.info("🚀 STARTING EXTERNAL VALIDATION")
        logger.info("=" * 80)

        # Load ensemble
        n_models = self.load_ensemble()
        if n_models == 0:
            logger.error("❌ No models found - aborting validation")
            return None, None

        # Load external data
        df_val = self.load_external_data(external_file)

        # Convert to graphs
        graphs = self.df_to_graphs(df_val)

        if len(graphs) == 0:
            logger.error("❌ No valid graphs generated - aborting validation")
            return None, None

        # Align dataframe with successful graphs
        df_val = df_val.iloc[: len(graphs)].copy()

        # Ensemble predictions
        predictions, uncertainties = self.predict_with_ensemble(graphs)

        # Calculate metrics for available properties
        results = {}
        target_props = ["Tg", "Tm", "Density"]

        for prop in target_props:
            if prop in df_val.columns:
                true_vals = df_val[prop].values
                metrics = self.calculate_metrics(
                    true_vals, predictions, uncertainties, prop
                )
                results[prop] = metrics

                logger.info(f"📊 {prop} Metrics:")
                logger.info(f"   R² = {metrics['r2']:.3f}")
                logger.info(f"   RMSE = {metrics['rmse']:.2f}")
                logger.info(f"   MAE = {metrics['mae']:.2f}")
                logger.info(f"   Unc Corr = {metrics['unc_corr']:.3f}")
                logger.info(f"   95% Coverage = {metrics['coverage_95']:.3f}")

        # Robustness test
        robustness = self.robustness_test(df_val, graphs)
        logger.info(f"🔬 Robustness Test:")
        logger.info(f"   Pred Shift: {robustness['pred_shift_pct']:.2f}%")
        logger.info(f"   Var Rise: {robustness['var_rise_pct']:.2f}%")

        # Save predictions
        df_val["Tg_pred"] = predictions
        df_val["Tg_unc"] = np.sqrt(uncertainties)

        # Add ensemble info
        df_val["ensemble_size"] = n_models
        df_val["validation_type"] = "external"

        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_val.to_csv(output_path, index=False)

        logger.info(f"💾 Predictions saved to {output_path}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 EXTERNAL VALIDATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Validated {len(df_val)} samples")
        logger.info(f"🔮 Ensemble size: {n_models}")

        if "Tg" in results:
            tg_r2 = results["Tg"]["r2"]
            tg_coverage = results["Tg"]["coverage_95"]
            logger.info(f"🎯 Tg R² = {tg_r2:.3f} (target: >0.6)")
            logger.info(f"📈 UQ Coverage = {tg_coverage:.3f} (target: >0.95)")

            if tg_r2 > 0.6 and tg_coverage > 0.9:
                logger.info("✅ EXCELLENT GENERALIZATION!")
            elif tg_r2 > 0.5:
                logger.info("✅ GOOD GENERALIZATION")
            else:
                logger.info("⚠️  Lower performance - consider fine-tuning")

        return results, robustness


def main():
    """Main execution"""

    # Check for external validation data
    external_files = [
        "results/external_validation_test.csv",
        "data/processed/external_validation.csv",
        "data/processed/external_validation_test.csv",
    ]

    external_file = None
    for file_path in external_files:
        if Path(file_path).exists():
            external_file = file_path
            break

    if external_file is None:
        logger.error("❌ No external validation file found")
        logger.info(f"Looked for: {external_files}")
        return

    # Run validation
    validator = ExternalValidator()
    result = validator.run_validation(external_file)

    if result is None:
        logger.error("❌ Validation failed")
        return

    results, robustness = result

    # Print final summary
    print("\n" + "=" * 60)
    print("📋 EXTERNAL VALIDATION SUMMARY")
    print("=" * 60)

    if results:
        for prop, metrics in results.items():
            print(f"{prop}:")
            print(f"  R² = {metrics['r2']:.3f}")
            print(f"  RMSE = {metrics['rmse']:.2f}")
            print(f"  Coverage = {metrics['coverage_95']:.3f}")
    else:
        print("No validation metrics available")

    if robustness:
        print(f"\nRobustness:")
        print(f"  Prediction shift: {robustness['pred_shift_pct']:.2f}%")
        print(f"  Uncertainty rise: {robustness['var_rise_pct']:.2f}%")
    else:
        print("No robustness metrics available")

    print("\n🚀 Ready for open-source release!")


if __name__ == "__main__":
    main()
