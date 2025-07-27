#!/usr/bin/env python3
"""
Uncertainty Quantification Analysis for PolyGNN
Practical implementation using existing trained models and datasets

Usage:
    python run_uq_analysis_fixed.py --ensemble_size 3 --device cpu
"""

import argparse
import json
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
import torch.nn as nn
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.polymer_dataset import PolymerTgDataset
from src.models.polymer_gcn import PolymerGCN
from src.training.gcn_trainer import PolymerGCNTrainer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleUQ:
    """Simplified uncertainty quantification"""

    def __init__(self, dataset_path: str = "data/processed/full_feats.csv"):
        self.dataset_path = Path(dataset_path)
        self.device = torch.device("cpu")  # Force CPU for stability
        self.results_dir = Path("results/uq_simple")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🎯 Simple UQ initialized")

    def train_single_model(self, seed: int = 42) -> nn.Module:
        """Train a single model for testing"""

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create datasets
        train_dataset = PolymerTgDataset(
            root=str(self.results_dir / f"model_{seed}"),
            csv_file=str(self.dataset_path),
            smiles_column="canonical_smiles",
            target_column="Tg",
            split_type="train",
            split_ratios=(0.7, 0.15, 0.15),
            polymer_feature_kwargs={"fingerprint_size": 128, "fp_radius": 2},
            random_state=seed,
        )

        val_dataset = PolymerTgDataset(
            root=str(self.results_dir / f"model_{seed}"),
            csv_file=str(self.dataset_path),
            smiles_column="canonical_smiles",
            target_column="Tg",
            split_type="val",
            split_ratios=(0.7, 0.15, 0.15),
            polymer_feature_kwargs={"fingerprint_size": 128, "fp_radius": 2},
            random_state=seed,
        )

        # Create model
        model = PolymerGCN(
            node_feature_dim=157,
            molecular_feature_dim=13,
            hidden_dims=[256, 256, 256],
            output_dim=1,
            num_layers=3,
            dropout_rate=0.2,
            pooling_method="mean",
            use_molecular_features=True,
            use_polymer_features=True,
            polymer_feature_dim=148,
            activation="relu",
        ).to(self.device)

        # Create trainer
        trainer = PolymerGCNTrainer(
            model=model,
            device=self.device,
            results_dir=str(self.results_dir / f"model_{seed}"),
        )

        # Train model
        results = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=32,
            max_epochs=30,  # Reduced for speed
            learning_rate=0.001,
            weight_decay=1e-4,
            patience=8,
            verbose=False,
        )

        return model

    def create_ensemble(self, n_models: int = 3) -> List[nn.Module]:
        """Create ensemble with different seeds"""

        models = []
        for i in range(n_models):
            seed = 42 + i * 17
            logger.info(f"🎭 Training model {i+1}/{n_models} with seed {seed}")

            try:
                model = self.train_single_model(seed)
                models.append(model)
                logger.info(f"✅ Model {i+1} trained successfully")
            except Exception as e:
                logger.error(f"❌ Failed to train model {i+1}: {e}")
                continue

        return models

    def ensemble_predict(
        self, models: List[nn.Module], test_dataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ensemble predictions"""

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        all_predictions = []
        targets = None

        for i, model in enumerate(models):
            model.eval()
            model_preds = []
            model_targets = []

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    pred = model(batch)
                    model_preds.append(pred.cpu().numpy())

                    if i == 0:  # Only collect targets once
                        model_targets.append(batch.y.cpu().numpy())

            predictions = np.concatenate(model_preds, axis=0)
            all_predictions.append(predictions)

            if i == 0:
                targets = np.concatenate(model_targets, axis=0)

        # Calculate ensemble statistics
        predictions = np.array(all_predictions)
        mean_pred = np.mean(predictions, axis=0).flatten()
        std_pred = np.std(predictions, axis=0).flatten()
        targets = targets.flatten()

        return mean_pred, std_pred, targets

    def analyze_uncertainty(
        self, predictions: np.ndarray, uncertainties: np.ndarray, targets: np.ndarray
    ) -> Dict:
        """Analyze uncertainty quality"""

        errors = np.abs(predictions - targets)

        # Error-uncertainty correlation
        pearson_corr = stats.pearsonr(errors, uncertainties)[0]

        # Simple coverage test (95% confidence)
        z_score = 1.96  # 95% confidence
        lower_bound = predictions - z_score * uncertainties
        upper_bound = predictions + z_score * uncertainties
        coverage = np.mean((targets >= lower_bound) & (targets <= upper_bound))

        return {
            "error_correlation": pearson_corr,
            "coverage_95": coverage,
            "mean_uncertainty": np.mean(uncertainties),
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "r2": r2_score(targets, predictions),
        }

    def run_simple_uq(self, n_models: int = 3) -> Dict:
        """Run simplified UQ analysis"""

        logger.info(f"🚀 Starting Simple UQ with {n_models} models")

        # Create test dataset
        test_dataset = PolymerTgDataset(
            root=str(self.results_dir / "test"),
            csv_file=str(self.dataset_path),
            smiles_column="canonical_smiles",
            target_column="Tg",
            split_type="test",
            split_ratios=(0.7, 0.15, 0.15),
            polymer_feature_kwargs={"fingerprint_size": 128, "fp_radius": 2},
            random_state=42,
        )

        logger.info(f"📊 Test dataset: {len(test_dataset)} samples")

        # Train ensemble
        models = self.create_ensemble(n_models)
        if not models:
            raise ValueError("No models trained successfully")

        logger.info(f"🎭 Ensemble: {len(models)} models")

        # Get predictions
        predictions, uncertainties, targets = self.ensemble_predict(
            models, test_dataset
        )

        # Analyze
        results = self.analyze_uncertainty(predictions, uncertainties, targets)

        # Save results
        results_file = self.results_dir / "simple_uq_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        print("\n" + "=" * 50)
        print("🎯 SIMPLE UQ RESULTS")
        print("=" * 50)
        print(f"🎭 Ensemble Size: {len(models)} models")
        print(f"📊 Test Samples: {len(targets)}")
        print(f"🏆 R²: {results['r2']:.4f}")
        print(f"📈 RMSE: {results['rmse']:.2f}°C")
        print(f"🔗 Error-Uncertainty Correlation: {results['error_correlation']:.3f}")
        print(f"📊 95% Coverage: {results['coverage_95']:.3f} (Target: 0.95)")
        print(f"🎲 Mean Uncertainty: {results['mean_uncertainty']:.2f}")

        # Quality assessment
        if results["error_correlation"] > 0.7:
            print("✅ GOOD uncertainty correlation")
        elif results["error_correlation"] > 0.5:
            print("⚠️  MODERATE uncertainty correlation")
        else:
            print("❌ POOR uncertainty correlation")

        if 0.90 <= results["coverage_95"] <= 0.98:
            print("✅ GOOD calibration")
        else:
            print("⚠️  MODERATE calibration")

        print("=" * 50)

        return results


def main():
    parser = argparse.ArgumentParser(description="Simple UQ Analysis")
    parser.add_argument("--dataset", type=str, default="data/processed/full_feats.csv")
    parser.add_argument("--n_models", type=int, default=3)

    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        sys.exit(1)

    uq = SimpleUQ(args.dataset)
    results = uq.run_simple_uq(args.n_models)

    return results


if __name__ == "__main__":
    main()
