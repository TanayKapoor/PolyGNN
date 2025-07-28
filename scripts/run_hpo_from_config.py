#!/usr/bin/env python3
"""
Configuration-based HPO Runner for Polymer GCN

Runs hyperparameter optimization using YAML configuration files.
Provides easy presets and customization options.

Usage:
    python run_hpo_from_config.py --config configs/hpo_config.yaml
    python run_hpo_from_config.py --config configs/hpo_config.yaml --search_type aggressive_search
    python run_hpo_from_config.py --preset quick  # Use quick preset
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

# Add src to path for imports
sys.path.append("src")

from src.data.polymer_dataset import PolymerTgDataset
from src.models.polymer_gcn import PolymerGCN, create_gcn_model_from_config
from src.training.gcn_trainer import PolymerGCNTrainer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_preset_config(preset: str) -> Dict[str, Any]:
    """Get predefined configuration presets."""
    presets = {
        "quick": {
            "hpo": {
                "method": "random",
                "max_trials": 10,
                "cv_folds": 3,
                "primary_metric": "r2",
                "max_epochs": 20,
                "patience": 5,
                "random_seed": 42,
            },
            "search_type": "conservative_search",
            "data": {
                "csv_file": "data/processed/filtered_tg_dataset.csv",
                "split_ratios": [0.8, 0.1, 0.1],
                "smiles_column": "SMILES",
                "target_column": "Tg",
            },
            "system": {
                "device": "auto",
                "results_dir": "results",
                "save_all_models": False,
            },
            "parameter_grids": {
                "conservative_search": {
                    "learning_rate": [1e-3, 2e-3],
                    "hidden_dims": [[128, 64, 32], [256, 128, 64]],
                    "num_gcn_layers": [3, 4],
                    "weight_decay": [1e-5, 1e-4],
                    "dropout_rate": [0.1, 0.2],
                    "batch_size": [32, 64],
                    "use_molecular_features": [False],  # Disable for debugging
                    "use_polymer_features": [False],  # Disable for debugging
                }
            },
        },
        "standard": {
            "hpo": {
                "method": "random",
                "max_trials": 50,
                "cv_folds": 5,
                "primary_metric": "r2",
                "max_epochs": 50,
                "patience": 8,
                "random_seed": 42,
            },
            "search_type": "random_search",
            "data": {
                "csv_file": "data/processed/filtered_tg_dataset.csv",
                "split_ratios": [0.8, 0.1, 0.1],
                "smiles_column": "SMILES",
                "target_column": "Tg",
            },
            "system": {
                "device": "auto",
                "results_dir": "results",
                "save_all_models": False,
            },
            "parameter_grids": {
                "random_search": {
                    "learning_rate": [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
                    "hidden_dims": [
                        [64, 32],
                        [128, 64, 32],
                        [256, 128, 64],
                        [512, 256, 128],
                    ],
                    "num_gcn_layers": [2, 3, 4, 5],
                    "weight_decay": [1e-6, 1e-5, 1e-4, 1e-3],
                    "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4],
                    "batch_size": [16, 32, 64, 128],
                    "use_molecular_features": [False],
                    "use_polymer_features": [False],
                }
            },
        },
        "comprehensive": {
            "hpo": {
                "method": "random",
                "max_trials": 100,
                "cv_folds": 5,
                "primary_metric": "r2",
                "max_epochs": 100,
                "patience": 15,
                "random_seed": 42,
            },
            "search_type": "aggressive_search",
            "data": {
                "csv_file": "data/processed/filtered_tg_dataset.csv",
                "split_ratios": [0.8, 0.1, 0.1],
                "smiles_column": "SMILES",
                "target_column": "Tg",
            },
            "system": {
                "device": "auto",
                "results_dir": "results",
                "save_all_models": False,
            },
            "parameter_grids": {
                "aggressive_search": {
                    "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
                    "hidden_dims": [
                        [32],
                        [64],
                        [128],
                        [64, 32],
                        [128, 64],
                        [256, 128],
                        [128, 64, 32],
                        [256, 128, 64],
                        [512, 256, 128],
                        [1024, 512, 256, 128],
                    ],
                    "num_gcn_layers": [1, 2, 3, 4, 5, 6],
                    "weight_decay": [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                    "dropout_rate": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
                    "batch_size": [8, 16, 32, 64, 128, 256],
                    "use_molecular_features": [False],
                    "use_polymer_features": [False],
                }
            },
        },
        "enhanced_full": {
            "hpo": {
                "method": "random",
                "max_trials": 30,
                "cv_folds": 5,
                "primary_metric": "r2",
                "max_epochs": 50,
                "patience": 8,
                "random_seed": 42,
            },
            "search_type": "random_search",
            "data": {
                "csv_file": "data/processed/filtered_tg_dataset.csv",
                "split_ratios": [0.8, 0.1, 0.1],
                "smiles_column": "SMILES",
                "target_column": "Tg",
            },
            "system": {
                "device": "auto",
                "results_dir": "results",
                "save_all_models": False,
            },
            "parameter_grids": {
                "random_search": {
                    # Architecture optimized for 147 polymer features
                    "hidden_dims": [
                        [512, 256, 128],  # Larger for more features
                        [256, 128, 64],
                        [384, 192, 96],  # Alternative sizing
                        [128, 64, 32],  # Smaller option
                    ],
                    "num_gcn_layers": [3, 4, 5],
                    "learning_rate": [0.0005, 0.001, 0.002, 0.003],
                    "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4],
                    "dropout": [0.1, 0.2, 0.3, 0.4],
                    "batch_size": [16, 32, 64],
                    # Enhanced polymer features (ALL ENABLED)
                    "use_polymer_features": [True],
                    "polymer_feature_dim": [147],  # All enhanced features
                    "fingerprint_size": [128],
                    "include_chain_descriptors": [True],  # 🆕 NEW
                    "include_complexity": [True],  # 🆕 NEW
                    "include_molecular_descriptors": [True],  # 🆕 NEW
                }
            },
        },
    }

    if preset not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    return presets[preset]


def setup_logging(config: Dict[str, Any]):
    """Setup logging based on configuration."""
    log_config = config.get("logging", {})

    level = getattr(logging, log_config.get("level", "INFO").upper())
    handlers = []

    if log_config.get("console_output", True):
        handlers.append(logging.StreamHandler())

    if "log_file" in log_config:
        handlers.append(logging.FileHandler(log_config["log_file"]))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def get_search_space(config: Dict[str, Any], search_type: str) -> Dict[str, list]:
    """Get parameter search space from configuration."""
    parameter_grids = config.get("parameter_grids", {})

    if search_type not in parameter_grids:
        available = list(parameter_grids.keys())
        raise ValueError(f"Unknown search type '{search_type}'. Available: {available}")

    return parameter_grids[search_type]


def load_dataset_from_config(config: Dict[str, Any]) -> tuple:
    """Load dataset based on configuration."""
    data_config = config["data"]

    # Get polymer features from data config or parameter grid
    polymer_features = data_config.get("polymer_features", {})

    # If no polymer features in data config, check parameter grid
    if not polymer_features:
        parameter_grids = config.get("parameter_grids", {})
        search_type = config.get("search_type", "random_search")

        if search_type in parameter_grids:
            grid = parameter_grids[search_type]

            # Extract polymer feature settings from parameter grid
            if grid.get("use_polymer_features", [False])[
                0
            ]:  # If using polymer features
                polymer_features = {
                    "fingerprint_size": grid.get("fingerprint_size", [128])[0],
                    "log_scale_dp": True,
                    "include_chain_descriptors": grid.get(
                        "include_chain_descriptors", [True]
                    )[0],
                    "include_complexity": grid.get("include_complexity", [True])[0],
                    "include_molecular_descriptors": grid.get(
                        "include_molecular_descriptors", [True]
                    )[0],
                    "cache_features": True,
                }
                logger.info(
                    f"Using polymer features from parameter grid: {polymer_features}"
                )

    split_ratios = data_config.get("split_ratios", [0.8, 0.1, 0.1])
    csv_file = data_config["csv_file"]
    smiles_column = data_config.get("smiles_column", "smiles")
    target_column = data_config.get("target_column", "tg")

    logger.info(f"Loading dataset from: {csv_file}")
    logger.info(f"SMILES column: '{smiles_column}', Target column: '{target_column}'")

    # Create train+val dataset for HPO
    train_val_dataset = PolymerTgDataset(
        root="data",
        csv_file=csv_file,
        smiles_column=smiles_column,
        target_column=target_column,
        split_type="train",
        split_ratios=split_ratios,
        polymer_feature_kwargs=polymer_features if polymer_features else None,
    )

    # Create validation dataset
    val_dataset = PolymerTgDataset(
        root="data",
        csv_file=csv_file,
        smiles_column=smiles_column,
        target_column=target_column,
        split_type="val",
        split_ratios=split_ratios,
        polymer_feature_kwargs=polymer_features if polymer_features else None,
    )

    # Create test dataset
    test_dataset = PolymerTgDataset(
        root="data",
        csv_file=csv_file,
        smiles_column=smiles_column,
        target_column=target_column,
        split_type="test",
        split_ratios=split_ratios,
        polymer_feature_kwargs=polymer_features if polymer_features else None,
    )

    # Combine train and val for HPO
    combined_data = []
    for i in range(len(train_val_dataset)):
        combined_data.append(train_val_dataset[i])
    for i in range(len(val_dataset)):
        combined_data.append(val_dataset[i])

    logger.info(
        f"Dataset loaded - Train+Val: {len(combined_data)}, Test: {len(test_dataset)}"
    )

    return combined_data, test_dataset


def run_hpo_from_config(config: Dict[str, Any], search_type: str = None):
    """Run HPO based on configuration."""

    # Setup logging
    setup_logging(config)

    logger.info("=" * 80)
    logger.info("CONFIGURATION-BASED HPO FOR POLYMER GCN")
    logger.info("=" * 80)

    # Get HPO settings
    hpo_config = config["hpo"]
    system_config = config["system"]

    # Determine search type
    if search_type is None:
        search_type = config.get("search_type", "random_search")

    logger.info(f"Search type: {search_type}")
    logger.info(f"Method: {hpo_config['method']}")
    logger.info(f"Max trials: {hpo_config['max_trials']}")

    # Set random seeds
    torch.manual_seed(hpo_config["random_seed"])
    np.random.seed(hpo_config["random_seed"])

    # Load data
    train_val_data, test_data = load_dataset_from_config(config)

    # Get search space
    param_grid = get_search_space(config, search_type)

    # Create initial model
    sample = train_val_data[0] if train_val_data else None
    if sample is None:
        raise ValueError("Empty dataset!")

    node_feature_dim = sample.x.shape[1] if hasattr(sample, "x") else 157
    logger.info(f"Dataset node feature dimension: {node_feature_dim}")

    # Get polymer feature settings from parameter grid
    use_polymer_features = param_grid.get("use_polymer_features", [False])[0]
    polymer_feature_dim = (
        param_grid.get("polymer_feature_dim", [0])[0] if use_polymer_features else 0
    )

    logger.info(f"Using polymer features: {use_polymer_features}")
    if use_polymer_features:
        logger.info(f"Polymer feature dimension: {polymer_feature_dim}")

    initial_model = PolymerGCN(
        node_feature_dim=node_feature_dim,
        hidden_dims=[128, 64, 32],
        num_gcn_layers=3,
        dropout_rate=0.2,
        use_polymer_features=use_polymer_features,
        polymer_feature_dim=polymer_feature_dim,
    )

    # Create trainer
    trainer = PolymerGCNTrainer(
        model=initial_model,
        device=system_config["device"],
        results_dir=system_config["results_dir"],
    )

    # Run HPO
    logger.info("Starting hyperparameter optimization...")
    hpo_results = trainer.hyperparam_optimize(
        dataset=train_val_data,
        param_grid=param_grid,
        method=hpo_config["method"],
        n_trials=hpo_config["max_trials"],
        cv_folds=hpo_config["cv_folds"],
        max_epochs=hpo_config["max_epochs"],
        patience=hpo_config["patience"],
        primary_metric=hpo_config["primary_metric"],
        minimize_primary=hpo_config["primary_metric"] in ["rmse", "mae"],
        save_all_models=system_config.get("save_all_models", False),
        random_seed=hpo_config["random_seed"],
    )

    # Final training
    final_config = config.get("final_training", {})
    logger.info("Retraining with best parameters...")
    final_results = trainer.retrain_with_best_params(
        hpo_results=hpo_results,
        full_dataset=train_val_data,
        test_dataset=test_data,
        max_epochs=final_config.get("max_epochs", 100),
        patience=final_config.get("patience", 15),
        save_final_model=final_config.get("save_model", True),
    )

    # Check success criteria
    success_config = config.get("success_criteria", {})
    if "test_metrics" in final_results and success_config:
        test_metrics = final_results["test_metrics"]

        logger.info("=" * 60)
        logger.info("SUCCESS CRITERIA EVALUATION")
        logger.info("=" * 60)

        r2_target = success_config.get("r2_target", 0.5)
        rmse_target = success_config.get("rmse_target", 50.0)
        mae_target = success_config.get("mae_target", 30.0)

        r2_success = test_metrics["r2"] >= r2_target
        rmse_success = test_metrics["rmse"] <= rmse_target
        mae_success = test_metrics["mae"] <= mae_target

        all_success = r2_success and rmse_success and mae_success

        logger.info(
            f"R² ≥ {r2_target}: {'✅' if r2_success else '❌'} ({test_metrics['r2']:.4f})"
        )
        logger.info(
            f"RMSE ≤ {rmse_target}: {'✅' if rmse_success else '❌'} ({test_metrics['rmse']:.4f})"
        )
        logger.info(
            f"MAE ≤ {mae_target}: {'✅' if mae_success else '❌'} ({test_metrics['mae']:.4f})"
        )
        logger.info(
            f"Overall: {'🎉 SUCCESS!' if all_success else '⚠️  Partially successful'}"
        )

    logger.info("Configuration-based HPO completed!")
    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Configuration-based HPO for Polymer GCN"
    )

    # Configuration options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to configuration YAML file")
    group.add_argument(
        "--preset",
        type=str,
        choices=["quick", "standard", "comprehensive", "enhanced_full"],
        help="Use predefined configuration preset",
    )

    # Override options
    parser.add_argument(
        "--search_type",
        type=str,
        choices=[
            "grid_search",
            "random_search",
            "conservative_search",
            "aggressive_search",
        ],
        help="Override search type from config",
    )
    parser.add_argument("--max_trials", type=int, help="Override max trials")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        help="Override device selection",
    )
    parser.add_argument("--results_dir", type=str, help="Override results directory")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        if not Path(args.config).exists():
            logger.error(f"Configuration file not found: {args.config}")
            return 1
        config = load_config(args.config)
        search_type = args.search_type
    else:
        config = get_preset_config(args.preset)
        search_type = args.search_type or config.get("search_type", "random_search")

    # Apply command-line overrides
    if args.max_trials:
        config["hpo"]["max_trials"] = args.max_trials
    if args.device:
        config["system"]["device"] = args.device
    if args.results_dir:
        config["system"]["results_dir"] = args.results_dir

    # Run HPO
    try:
        final_results = run_hpo_from_config(config, search_type)
        logger.info("HPO completed successfully!")

        # Generate report if analysis module is available
        try:
            from analysis.hpo_report import HPOAnalyzer

            # Find the latest HPO directory
            results_dir = Path(config["system"]["results_dir"])
            hpo_dirs = [
                d
                for d in (results_dir / "hpo").iterdir()
                if d.is_dir() and d.name.startswith("hpo_")
            ]

            if hpo_dirs:
                latest_hpo_dir = max(hpo_dirs, key=lambda x: x.name)
                analyzer = HPOAnalyzer(latest_hpo_dir)
                report_path = analyzer.generate_comprehensive_report()
                logger.info(f"Analysis report generated: {report_path}")

        except ImportError:
            logger.info("Analysis module not available, skipping report generation")
        except Exception as e:
            logger.warning(f"Failed to generate analysis report: {e}")

        return 0

    except Exception as e:
        logger.error(f"HPO failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
