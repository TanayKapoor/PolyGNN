#!/usr/bin/env python3
"""
Week 2 Pipeline: Expanded HPO + Uncertainty Quantification
Complete pipeline for advanced hyperparameter optimization and robust UQ

Pipeline Steps:
1. Expanded HPO with 200 Bayesian trials
2. Train ensemble models from top-k HPO results  
3. Uncertainty quantification with calibration analysis
4. Performance validation targeting R² > 0.5

Usage:
    python run_week2_pipeline.py --gpu --n_trials 200 --ensemble_size 5
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from run_expanded_hpo import ExpandedHPOOptimizer

from src.data.polymer_dataset import PolymerDataset
from src.models.polymer_gcn import PolymerGCN
from src.models.uncertainty_quantification import EnsembleGCN, UncertaintyQuantifier
from src.training.gcn_trainer import PolymerGCNTrainer

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Week2Pipeline:
    """Complete Week 2 pipeline: Expanded HPO + Uncertainty Quantification"""

    def __init__(
        self,
        dataset_path: str = "data/processed/full_feats.csv",
        n_trials: int = 200,
        ensemble_size: int = 5,
        use_gpu: bool = True,
        results_dir: str = "results/week2_pipeline",
    ):

        self.dataset_path = Path(dataset_path)
        self.n_trials = n_trials
        self.ensemble_size = ensemble_size
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline state
        self.hpo_results = None
        self.ensemble_models = []
        self.final_results = {}

        logger.info(f"🚀 Week 2 Pipeline initialized")
        logger.info(f"📊 Device: {self.device}")
        logger.info(f"🎯 Target: R² > 0.5 with robust UQ")
        logger.info(f"🧪 HPO Trials: {n_trials}")
        logger.info(f"🎭 Ensemble Size: {ensemble_size}")

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete Week 2 pipeline"""

        logger.info("=" * 80)
        logger.info("🚀 STARTING WEEK 2 PIPELINE: EXPANDED HPO + UQ")
        logger.info("=" * 80)

        pipeline_start = time.time()

        try:
            # Step 1: Expanded HPO
            logger.info("\n" + "🔍 STEP 1: EXPANDED HYPERPARAMETER OPTIMIZATION")
            logger.info("-" * 50)
            self.hpo_results = self._run_expanded_hpo()

            # Step 2: Train ensemble models
            logger.info("\n" + "🎭 STEP 2: ENSEMBLE MODEL TRAINING")
            logger.info("-" * 50)
            self.ensemble_models = self._train_ensemble_models()

            # Step 3: Uncertainty quantification
            logger.info("\n" + "🎯 STEP 3: UNCERTAINTY QUANTIFICATION")
            logger.info("-" * 50)
            uq_results = self._run_uncertainty_quantification()

            # Step 4: Final validation
            logger.info("\n" + "✅ STEP 4: FINAL VALIDATION")
            logger.info("-" * 50)
            final_metrics = self._final_validation()

            # Compile results
            pipeline_time = time.time() - pipeline_start

            self.final_results = {
                "pipeline_info": {
                    "completion_time": datetime.now().isoformat(),
                    "total_time_hours": pipeline_time / 3600,
                    "target_achieved": final_metrics["best_r2"] >= 0.5,
                    "dataset_path": str(self.dataset_path),
                    "n_trials": self.n_trials,
                    "ensemble_size": self.ensemble_size,
                },
                "hpo_results": self.hpo_results,
                "ensemble_performance": final_metrics,
                "uncertainty_quantification": uq_results,
                "summary": {
                    "best_single_r2": self.hpo_results["best_value"],
                    "ensemble_r2": final_metrics["best_r2"],
                    "uncertainty_correlation": uq_results.get("calibration", {})
                    .get("error_correlation", {})
                    .get("pearson", 0),
                    "calibration_95_coverage": (
                        uq_results.get("calibration", {}).get("coverage", [0, 0, 0, 0])[
                            -1
                        ]
                        if uq_results.get("calibration")
                        else 0
                    ),
                },
            }

            # Save results
            self._save_pipeline_results()

            # Print summary
            self._print_final_summary()

            return self.final_results

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

    def _run_expanded_hpo(self) -> Dict[str, Any]:
        """Run expanded hyperparameter optimization"""

        # Create HPO optimizer
        hpo_optimizer = ExpandedHPOOptimizer(
            dataset_path=str(self.dataset_path),
            n_trials=self.n_trials,
            use_gpu=self.device.type == "cuda",
            multi_task=True,
            results_dir=str(self.results_dir / "hpo"),
        )

        # Run optimization
        hpo_results = hpo_optimizer.optimize()

        logger.info(f"🏆 HPO Best R²: {hpo_results['best_value']:.4f}")
        logger.info(
            f"✅ Target R² ≥ 0.5: {'ACHIEVED' if hpo_results['target_achieved'] else 'APPROACHING'}"
        )

        return hpo_results

    def _train_ensemble_models(self) -> List[nn.Module]:
        """Train ensemble models from top-k HPO results"""

        if not self.hpo_results:
            raise ValueError("HPO results not available. Run HPO first.")

        logger.info(f"🏗️  Training ensemble of {self.ensemble_size} models...")

        # Get top-k parameter sets from HPO
        trial_results = self.hpo_results.get("trial_history", [])
        if not trial_results:
            logger.warning("No trial history found. Using best parameters only.")
            top_trials = [
                {
                    "parameters": self.hpo_results["best_params"],
                    "r2_score": self.hpo_results["best_value"],
                }
            ]
        else:
            # Sort by R² and take top-k
            sorted_trials = sorted(
                trial_results, key=lambda x: x["r2_score"], reverse=True
            )
            top_trials = sorted_trials[: self.ensemble_size]

        ensemble_models = []

        for i, trial in enumerate(top_trials):
            logger.info(
                f"🎭 Training ensemble model {i+1}/{len(top_trials)} (R²: {trial['r2_score']:.4f})"
            )

            try:
                model = self._train_single_model(
                    trial["parameters"], f"ensemble_model_{i}"
                )
                ensemble_models.append(model)
            except Exception as e:
                logger.error(f"❌ Failed to train ensemble model {i}: {e}")
                continue

        logger.info(f"✅ Successfully trained {len(ensemble_models)} ensemble models")
        return ensemble_models

    def _train_single_model(self, params: Dict[str, Any], model_name: str) -> nn.Module:
        """Train a single model with given parameters"""

        # Create dataset
        dataset = PolymerDataset(
            csv_path=str(self.dataset_path),
            smiles_column=(
                "canonical_smiles"
                if "canonical_smiles" in pd.read_csv(self.dataset_path).columns
                else "smiles"
            ),
            target_column="Tg",
            splits={
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "shuffle": True,
            },
            use_polymer_features=True,
            device=self.device,
        )

        train_dataset, val_dataset, test_dataset = dataset.get_splits()

        # Create model with HPO parameters
        model = PolymerGCN(
            node_feature_dim=157,
            molecular_feature_dim=13,
            hidden_dims=[params.get("hidden_dims", 256)]
            * params.get("num_gcn_layers", 3),
            output_dim=1,
            num_layers=params.get("num_gcn_layers", 3),
            dropout_rate=params.get("dropout_rate", 0.2),
            pooling_method=params.get("pooling_method", "mean"),
            use_molecular_features=True,
            use_polymer_features=True,
            polymer_feature_dim=148,
            activation=params.get("activation", "relu"),
        ).to(self.device)

        # Create trainer
        trainer = PolymerGCNTrainer(
            model=model,
            device=self.device,
            results_dir=str(self.results_dir / "ensemble" / model_name),
        )

        # Train model
        results = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=params.get("batch_size", 32),
            max_epochs=100,
            learning_rate=params.get("learning_rate", 0.001),
            weight_decay=params.get("weight_decay", 1e-4),
            patience=15,
            verbose=False,
        )

        return model

    def _run_uncertainty_quantification(self) -> Dict[str, Any]:
        """Run uncertainty quantification analysis"""

        if not self.ensemble_models:
            raise ValueError("No ensemble models available. Train ensemble first.")

        logger.info(
            f"🎲 Running UQ analysis with {len(self.ensemble_models)} models..."
        )

        # Create dataset for evaluation
        dataset = PolymerDataset(
            csv_path=str(self.dataset_path),
            smiles_column=(
                "canonical_smiles"
                if "canonical_smiles" in pd.read_csv(self.dataset_path).columns
                else "smiles"
            ),
            target_column="Tg",
            splits={
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "shuffle": True,
            },
            use_polymer_features=True,
            device=self.device,
        )

        train_dataset, val_dataset, test_dataset = dataset.get_splits()

        # Create validation dataloader
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create UQ system
        uq_system = UncertaintyQuantifier(
            models=self.ensemble_models, device=self.device, mc_samples=20
        )

        # Run UQ evaluation
        uq_results = uq_system.evaluate_uncertainty(
            val_loader,
            method="ensemble",
            save_plots=True,
            results_dir=self.results_dir / "uq_analysis",
        )

        logger.info(f"🎯 UQ Analysis Complete:")
        logger.info(
            f"   Error-Uncertainty Correlation: {uq_results['calibration']['error_correlation']['pearson']:.3f}"
        )
        logger.info(f"   95% Coverage: {uq_results['calibration']['coverage'][-1]:.3f}")

        return uq_results

    def _final_validation(self) -> Dict[str, Any]:
        """Final validation with ensemble predictions"""

        logger.info("📊 Running final ensemble validation...")

        # Create test dataset
        dataset = PolymerDataset(
            csv_path=str(self.dataset_path),
            smiles_column=(
                "canonical_smiles"
                if "canonical_smiles" in pd.read_csv(self.dataset_path).columns
                else "smiles"
            ),
            target_column="Tg",
            splits={
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "shuffle": True,
            },
            use_polymer_features=True,
            device=self.device,
        )

        train_dataset, val_dataset, test_dataset = dataset.get_splits()
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Ensemble prediction
        ensemble = EnsembleGCN(self.ensemble_models, self.device)
        predictions, uncertainties, targets = ensemble.predict(
            test_loader, return_variance=True
        )

        # Calculate metrics
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        if len(targets.shape) > 1:
            targets = targets.flatten()

        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)

        # Individual model performance
        individual_r2s = []
        for i, model in enumerate(self.ensemble_models):
            model_preds = []
            model_targets = []

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    pred = model(batch)
                    model_preds.append(pred.cpu().numpy())
                    if i == 0:  # Only collect targets once
                        if hasattr(batch, "y"):
                            model_targets.append(batch.y.cpu().numpy())

            model_preds = np.concatenate(model_preds).flatten()
            if i == 0:
                model_targets = np.concatenate(model_targets).flatten()

            individual_r2 = r2_score(model_targets, model_preds)
            individual_r2s.append(individual_r2)

        metrics = {
            "ensemble_performance": {
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "n_test_samples": len(targets),
            },
            "individual_model_r2s": individual_r2s,
            "best_r2": r2,
            "mean_individual_r2": np.mean(individual_r2s),
            "r2_improvement": r2 - np.mean(individual_r2s),
            "target_achieved": r2 >= 0.5,
        }

        logger.info(f"🏆 Final Ensemble Performance:")
        logger.info(f"   Ensemble R²: {r2:.4f}")
        logger.info(f"   Individual R² (mean): {np.mean(individual_r2s):.4f}")
        logger.info(f"   Ensemble Improvement: +{r2 - np.mean(individual_r2s):.4f}")
        logger.info(
            f"   Target R² ≥ 0.5: {'✅ ACHIEVED' if r2 >= 0.5 else '⚠️  APPROACHING'}"
        )

        return metrics

    def _save_pipeline_results(self):
        """Save comprehensive pipeline results"""

        # Save main results
        results_file = (
            self.results_dir
            / f"week2_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(self.final_results, f, indent=2, default=str)

        # Save model ensemble
        ensemble_dir = self.results_dir / "final_ensemble"
        ensemble_dir.mkdir(exist_ok=True)

        for i, model in enumerate(self.ensemble_models):
            model_path = ensemble_dir / f"ensemble_model_{i}.pth"
            torch.save(model.state_dict(), model_path)

        logger.info(f"💾 Pipeline results saved to {results_file}")
        logger.info(f"💾 Ensemble models saved to {ensemble_dir}")

    def _print_final_summary(self):
        """Print comprehensive final summary"""

        print("\n" + "=" * 80)
        print("🎉 WEEK 2 PIPELINE COMPLETION SUMMARY")
        print("=" * 80)

        results = self.final_results

        print(f"🏆 PERFORMANCE ACHIEVEMENTS:")
        print(f"   📊 Best Single Model R²: {results['hpo_results']['best_value']:.4f}")
        print(f"   🎭 Ensemble R²: {results['ensemble_performance']['best_r2']:.4f}")
        print(
            f"   🎯 Target R² ≥ 0.5: {'✅ ACHIEVED' if results['pipeline_info']['target_achieved'] else '⚠️  APPROACHING'}"
        )
        print(
            f"   📈 Ensemble Improvement: +{results['ensemble_performance']['r2_improvement']:.4f}"
        )

        print(f"\n🔍 HPO OPTIMIZATION:")
        print(f"   🧪 Total Trials: {results['hpo_results']['n_trials']}")
        print(
            f"   ✅ Completed: {results['hpo_results']['study_stats']['completed_trials']}"
        )
        print(
            f"   ⏱️  Time: {results['hpo_results']['optimization_time_hours']:.2f} hours"
        )

        print(f"\n🎲 UNCERTAINTY QUANTIFICATION:")
        uq_corr = (
            results["uncertainty_quantification"]
            .get("calibration", {})
            .get("error_correlation", {})
            .get("pearson", 0)
        )
        uq_coverage = (
            results["uncertainty_quantification"]
            .get("calibration", {})
            .get("coverage", [0])[-1]
        )
        print(f"   🔗 Error-Uncertainty Correlation: {uq_corr:.3f} (Target: >0.7)")
        print(f"   📊 95% Calibration Coverage: {uq_coverage:.3f} (Target: 0.95)")
        print(f"   🎭 Ensemble Size: {len(self.ensemble_models)} models")

        print(
            f"\n⏱️  TOTAL PIPELINE TIME: {results['pipeline_info']['total_time_hours']:.2f} hours"
        )
        print(f"📁 Results saved to: {self.results_dir}")

        print("\n🚀 READY FOR WEEK 3: External Validation (PoLyInfo subsets)")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Week 2 Pipeline: Expanded HPO + UQ")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/full_feats.csv",
        help="Path to enhanced dataset",
    )
    parser.add_argument(
        "--n_trials", type=int, default=200, help="Number of HPO trials"
    )
    parser.add_argument(
        "--ensemble_size", type=int, default=5, help="Number of models in ensemble"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/week2_pipeline",
        help="Results directory",
    )

    args = parser.parse_args()

    # Validate dataset
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        print("💡 Run the feature engineering pipeline first:")
        print("   python build_full_features.py")
        sys.exit(1)

    # Create and run pipeline
    pipeline = Week2Pipeline(
        dataset_path=args.dataset,
        n_trials=args.n_trials,
        ensemble_size=args.ensemble_size,
        use_gpu=args.gpu,
        results_dir=args.results_dir,
    )

    # Execute complete pipeline
    results = pipeline.run_complete_pipeline()

    return results


if __name__ == "__main__":
    main()
