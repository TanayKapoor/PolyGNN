#!/usr/bin/env python3
"""
Simplified Expanded HPO for PolyGNN
Uses existing training infrastructure with Optuna optimization

Usage:
    python run_hpo_simple.py --n_trials 200 --gpu
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_gcn_training(config_dict: Dict[str, Any]) -> float:
    """Run single GCN training and return validation R²"""

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name

    try:
        # Import and run training
        from train_polymer_gcn import main as train_main

        # Override sys.argv to pass config
        original_argv = sys.argv.copy()
        sys.argv = ["train_polymer_gcn.py", "--config", config_path]

        # Capture training results
        results = train_main()

        # Extract best validation R²
        if results and isinstance(results, dict):
            if "val_metrics" in results:
                val_r2_scores = [m.get("r2", 0) for m in results["val_metrics"]]
                best_r2 = max(val_r2_scores) if val_r2_scores else 0.0
            elif "best_val_r2" in results:
                best_r2 = results["best_val_r2"]
            elif "final_metrics" in results and "r2" in results["final_metrics"]:
                best_r2 = results["final_metrics"]["r2"]
            else:
                best_r2 = 0.0
        else:
            best_r2 = 0.0

        # Restore sys.argv
        sys.argv = original_argv

        return best_r2

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 0.0

    finally:
        # Clean up temp file
        try:
            os.unlink(config_path)
        except:
            pass


class SimpleHPOOptimizer:
    """Simplified HPO using existing training infrastructure"""

    def __init__(
        self,
        n_trials: int = 200,
        multi_task: bool = True,
        results_dir: str = "results/simple_hpo",
    ):

        self.n_trials = n_trials
        self.multi_task = multi_task
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Multi-task weights (focus on Tg)
        self.task_weights = (
            {
                "Tg": 0.6,  # Primary focus
                "Tm": 0.2,  # Secondary
                "Density": 0.2,  # Secondary
            }
            if multi_task
            else {"Tg": 1.0}
        )

        # Base config template
        self.base_config = {
            "experiment": {
                "name": "hpo_trial",
                "description": "HPO trial with enhanced features",
                "random_seed": 42,
            },
            "data": {
                "dataset_path": "data/processed/full_feats.csv",
                "smiles_column": "canonical_smiles",
                "target_column": "Tg",
                "splits": {
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                    "shuffle": True,
                },
                "target_transform": {
                    "enabled": True,
                    "method": "clip",
                    "clip_min": -200,
                    "clip_max": 500,
                },
            },
            "graph": {
                "max_atoms": 200,
                "include_hydrogens": False,
                "use_chirality": True,
                "use_bond_types": False,
            },
            "model": {
                "type": "gcn",
                "node_feature_dim": 157,
                "molecular_feature_dim": 13,
                "pooling_method": "mean",
                "use_molecular_features": True,
                "use_polymer_features": True,
                "polymer_feature_dim": 148,
                "activation": "relu",
            },
            "training": {
                "device": "auto",
                "epochs": 50,
                "early_stopping": {"patience": 8, "min_delta": 1e-4},
            },
            "cross_validation": {"enabled": False},
            "results": {"save_dir": "results", "log_level": "WARNING"},
            "polymer_features": {
                "fingerprint_size": 128,
                "fp_radius": 2,
                "log_scale_dp": True,
                "max_dp": 10000,
                "cache_features": False,
            },
            "advanced": {
                "feature_scaling": {"node_features": False, "molecular_features": True}
            },
        }

        self.best_r2 = 0.0
        self.trial_results = []

        logger.info(f"🚀 Simple HPO initialized with {n_trials} trials")
        logger.info(f"⚖️  Task weights: {self.task_weights}")
        logger.info(f"🎯 Multi-task enabled: {multi_task}")

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function"""

        # Sample hyperparameters
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "num_gcn_layers": trial.suggest_int("num_gcn_layers", 3, 5),
            "hidden_dims": trial.suggest_categorical("hidden_dims", [128, 256, 512]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "pooling_method": trial.suggest_categorical(
                "pooling_method", ["mean", "max", "add"]
            ),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "gelu", "swish"]
            ),
        }

        # Create config for this trial
        config = self.base_config.copy()
        config["experiment"]["name"] = f"hpo_trial_{trial.number}"

        # Update model parameters
        config["model"]["num_gcn_layers"] = params["num_gcn_layers"]
        config["model"]["hidden_dims"] = [params["hidden_dims"]] * params[
            "num_gcn_layers"
        ]
        config["model"]["dropout_rate"] = params["dropout_rate"]
        config["model"]["pooling_method"] = params["pooling_method"]
        config["model"]["activation"] = params["activation"]

        # Update training parameters
        config["training"]["batch_size"] = params["batch_size"]
        config["training"]["learning_rate"] = params["learning_rate"]
        config["training"]["weight_decay"] = params["weight_decay"]
        config["training"]["epochs"] = 50  # Use 'epochs' not 'max_epochs'

        # Update results directory
        config["results"]["model_name"] = f"hpo_trial_{trial.number}"

        try:
            # Run training
            r2_score = run_single_gcn_training(config)

            # Track results
            if r2_score > self.best_r2:
                self.best_r2 = r2_score
                logger.info(f"🎯 New best R²: {r2_score:.4f} (Trial {trial.number})")

            # Store results
            trial_result = {
                "trial_number": trial.number,
                "r2_score": r2_score,
                "parameters": params,
                "timestamp": datetime.now().isoformat(),
            }
            self.trial_results.append(trial_result)

            return r2_score

        except Exception as e:
            logger.error(f"❌ Trial {trial.number} failed: {e}")
            return 0.0

    def optimize(self) -> Dict[str, Any]:
        """Run HPO optimization"""

        logger.info(f"🚀 Starting HPO with {self.n_trials} trials...")

        # Create study
        sampler = TPESampler(
            n_startup_trials=20, n_ei_candidates=24, multivariate=True, seed=42
        )

        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=5)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"simple_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # Add initial good trial
        study.enqueue_trial(
            {
                "learning_rate": 0.001,
                "num_gcn_layers": 3,
                "hidden_dims": 256,
                "dropout_rate": 0.2,
                "batch_size": 32,
                "weight_decay": 1e-4,
                "pooling_method": "mean",
                "activation": "relu",
            }
        )

        start_time = time.time()

        # Run optimization with progress callback
        def progress_callback(study, trial):
            if trial.number % 5 == 0:
                current_val = (
                    f"{trial.value:.4f}" if trial.value is not None else "Failed"
                )
                logger.info(
                    f"📊 Progress: Trial {trial.number}/{self.n_trials} | "
                    f"Best R²: {study.best_value:.4f} | "
                    f"Current: {current_val}"
                )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            callbacks=[progress_callback],
            show_progress_bar=True,
        )

        optimization_time = time.time() - start_time

        # Compile results
        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "optimization_time_hours": optimization_time / 3600,
            "target_achieved": study.best_value >= 0.5,
            "study_stats": {
                "completed_trials": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                ),
                "pruned_trials": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED
                    ]
                ),
                "failed_trials": len(
                    [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
                ),
            },
            "trial_history": self.trial_results,
        }

        # Save results
        results_file = (
            self.results_dir
            / f"hpo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"🎉 HPO completed!")
        logger.info(f"🏆 Best R²: {study.best_value:.4f}")
        logger.info(f"✅ Target achieved: {results['target_achieved']}")
        logger.info(f"⏱️  Time: {optimization_time/3600:.2f} hours")
        logger.info(f"📁 Results saved to: {results_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Simple HPO for PolyGNN")
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of optimization trials"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/simple_hpo",
        help="Results directory",
    )

    args = parser.parse_args()

    # Check if dataset exists
    dataset_path = Path("data/processed/full_feats.csv")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("💡 Run the feature engineering pipeline first:")
        print("   python build_full_features.py")
        sys.exit(1)

    # Create optimizer
    optimizer = SimpleHPOOptimizer(
        n_trials=args.n_trials, multi_task=True, results_dir=args.results_dir
    )

    # Run optimization
    results = optimizer.optimize()

    # Print summary
    print("\n" + "=" * 60)
    print("🎯 SIMPLE HPO RESULTS SUMMARY")
    print("=" * 60)
    print(f"🏆 Best Validation R²: {results['best_value']:.4f}")
    print(
        f"✅ Target R² ≥ 0.5: {'ACHIEVED' if results['target_achieved'] else 'APPROACHING'}"
    )
    print(f"📊 Completed Trials: {results['study_stats']['completed_trials']}")
    print(f"⏱️  Optimization Time: {results['optimization_time_hours']:.2f} hours")
    print("\n🎛️  Best Parameters:")
    for param, value in results["best_params"].items():
        print(f"   {param}: {value}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
