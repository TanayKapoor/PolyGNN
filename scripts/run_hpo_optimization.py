#!/usr/bin/env python3
"""
Hyperparameter Optimization for Polymer GCN

Main script to run comprehensive hyperparameter optimization for the PolymerGCN model.
Aims to achieve target performance: R² ≥ 0.5, RMSE ≤ 50.0, MAE ≤ 30.0

Usage:
    python run_hpo_optimization.py --method grid --max_trials 100
    python run_hpo_optimization.py --method random --max_trials 50
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path for imports
sys.path.append("src")

from src.data.polymer_dataset import PolymerTgDataset
from src.models.polymer_gcn import PolymerGCN, create_gcn_model_from_config
from src.training.gcn_trainer import PolymerGCNTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("hpo_optimization.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def define_search_space(method: str = "grid"):
    """
    Define the hyperparameter search space.

    Args:
        method: 'grid' for smaller focused search, 'random' for broader exploration

    Returns:
        Dictionary of parameter lists for optimization
    """
    if method == "grid":
        # Focused grid search - most critical parameters
        param_grid = {
            "learning_rate": [5e-4, 1e-3, 2e-3, 5e-3],
            "hidden_dims": [
                [64, 32, 16],
                [128, 64, 32],
                [256, 128, 64],
                [512, 256, 128],
            ],
            "num_gcn_layers": [2, 3, 4, 5],
            "weight_decay": [0, 1e-5, 1e-4, 1e-3],
            "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.5],
            "batch_size": [16, 32, 64, 128],
        }
    else:  # random search - broader exploration
        param_grid = {
            "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
            "hidden_dims": [
                [32, 16],
                [64, 32],
                [64, 32, 16],
                [128, 64],
                [128, 64, 32],
                [256, 128],
                [256, 128, 64],
                [512, 256],
                [512, 256, 128],
                [1024, 512, 256],
            ],
            "num_gcn_layers": [2, 3, 4, 5, 6],
            "weight_decay": [0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            "dropout_rate": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6],
            "batch_size": [8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
            "pooling_method": ["mean", "max", "sum"],
            "activation": ["relu", "gelu", "tanh"],
            # Feature usage variations
            "use_molecular_features": [True, False],
            "use_polymer_features": [True, False],
        }

    logger.info(f"Search space defined for {method} search:")
    total_combinations = 1
    for param, values in param_grid.items():
        logger.info(f"  {param}: {len(values)} options")
        total_combinations *= len(values)

    if method == "grid":
        logger.info(f"Total combinations: {total_combinations:,}")
        if total_combinations > 1000:
            logger.warning(
                f"Large search space ({total_combinations:,} combinations)! Consider using random search."
            )

    return param_grid


def load_data(data_path: str = "data/processed/filtered_tg_dataset.csv") -> tuple:
    """
    Load and prepare the dataset for HPO.

    Args:
        data_path: Path to the dataset CSV file

    Returns:
        Tuple of (train_val_dataset, test_dataset)
    """
    logger.info("Loading dataset...")

    # Check if data file exists
    if not Path(data_path).exists():
        # Try alternative paths
        alternative_paths = [
            "data/raw/polymer_tg_data.csv",
            "data/polymer_tg_data.csv",
            "../data/processed/polymer_tg_data.csv",
        ]

        data_path = None
        for alt_path in alternative_paths:
            if Path(alt_path).exists():
                data_path = alt_path
                break

        if data_path is None:
            raise FileNotFoundError(
                "Dataset not found. Please ensure polymer_tg_data.csv exists in data/processed/ or data/raw/"
            )

    logger.info(f"Using dataset: {data_path}")

    # Create train+val dataset for HPO
    train_val_dataset = PolymerTgDataset(
        root="data",
        csv_file=data_path,
        smiles_column="SMILES",
        target_column="Tg",
        split_type="train",  # This will get both train and val
        split_ratios=(0.8, 0.1, 0.1),  # 80% train+val for HPO, 10% test
        polymer_feature_kwargs={
            "use_fingerprints": True,
            "use_molecular_weight": True,
            "use_degree_of_polymerization": True,
            "fingerprint_type": "morgan",
            "fingerprint_radius": 3,
            "fingerprint_bits": 2048,
        },
    )

    # Create test dataset
    test_dataset = PolymerTgDataset(
        root="data",
        csv_file=data_path,
        smiles_column="SMILES",
        target_column="Tg",
        split_type="test",
        split_ratios=(0.8, 0.1, 0.1),
        polymer_feature_kwargs={
            "use_fingerprints": True,
            "use_molecular_weight": True,
            "use_degree_of_polymerization": True,
            "fingerprint_type": "morgan",
            "fingerprint_radius": 3,
            "fingerprint_bits": 2048,
        },
    )

    # Combine train and val for HPO (test remains separate)
    val_dataset = PolymerTgDataset(
        root="data",
        csv_file=data_path,
        smiles_column="SMILES",
        target_column="Tg",
        split_type="val",
        split_ratios=(0.8, 0.1, 0.1),
        polymer_feature_kwargs={
            "use_fingerprints": True,
            "use_molecular_weight": True,
            "use_degree_of_polymerization": True,
            "fingerprint_type": "morgan",
            "fingerprint_radius": 3,
            "fingerprint_bits": 2048,
        },
    )

    # Combine train and val datasets for HPO
    combined_data = []
    for i in range(len(train_val_dataset)):
        combined_data.append(train_val_dataset[i])
    for i in range(len(val_dataset)):
        combined_data.append(val_dataset[i])

    logger.info(f"Dataset loaded:")
    logger.info(f"  Train+Val samples: {len(combined_data)}")
    logger.info(f"  Test samples: {len(test_dataset)}")

    # Log feature information
    if len(combined_data) > 0:
        sample = combined_data[0]
        logger.info(
            f"  Node features: {sample.x.shape[1] if hasattr(sample, 'x') else 'N/A'}"
        )
        logger.info(
            f"  Molecular features: {sample.mol_features.shape[0] if hasattr(sample, 'mol_features') else 'N/A'}"
        )
        logger.info(
            f"  Polymer features: {sample.polymer_features.shape[0] if hasattr(sample, 'polymer_features') else 'N/A'}"
        )

    return combined_data, test_dataset


def run_hpo(args):
    """Run the hyperparameter optimization process."""
    logger.info("=" * 80)
    logger.info("POLYMER GCN HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Method: {args.method}")
    logger.info(f"Max trials: {args.max_trials}")
    logger.info(f"CV folds: {args.cv_folds}")
    logger.info(f"Device: {args.device}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    train_val_data, test_data = load_data(args.data_path)

    # Define search space
    param_grid = define_search_space(args.method)

    # Create dummy model to initialize trainer
    sample = train_val_data[0] if train_val_data else None
    if sample is None:
        raise ValueError("Empty dataset!")

    node_feature_dim = sample.x.shape[1] if hasattr(sample, "x") else 157

    # Create initial model (will be replaced during HPO)
    initial_model = PolymerGCN(
        node_feature_dim=node_feature_dim,
        hidden_dims=[128, 64, 32],
        num_gcn_layers=3,
        dropout_rate=0.2,
    )

    # Create trainer
    trainer = PolymerGCNTrainer(
        model=initial_model,
        device=args.device,
        results_dir=args.results_dir,
        model_name="polymer_gcn_hpo",
    )

    # Run HPO
    logger.info("Starting hyperparameter optimization...")
    hpo_results = trainer.hyperparam_optimize(
        dataset=train_val_data,
        param_grid=param_grid,
        method=args.method,
        n_trials=args.max_trials,
        cv_folds=args.cv_folds,
        max_epochs=args.max_epochs,
        patience=args.patience,
        primary_metric=args.primary_metric,
        minimize_primary=args.primary_metric in ["rmse", "mae"],
        save_all_models=args.save_all_models,
        random_seed=args.seed,
    )

    # Retrain with best parameters
    logger.info("Retraining with best parameters...")
    final_results = trainer.retrain_with_best_params(
        hpo_results=hpo_results,
        full_dataset=train_val_data,
        test_dataset=test_data,
        max_epochs=args.final_epochs,
        patience=args.final_patience,
        save_final_model=True,
    )

    # Summary
    logger.info("=" * 80)
    logger.info("HPO COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total HPO time: {hpo_results['total_time']/3600:.2f} hours")
    logger.info(
        f"Successful trials: {hpo_results['successful_trials']}/{hpo_results['total_trials']}"
    )
    logger.info(f"Best CV {args.primary_metric}: {hpo_results['best_score']:.4f}")

    if "test_metrics" in final_results:
        test_metrics = final_results["test_metrics"]
        logger.info(f"Final test metrics:")
        logger.info(f"  R²: {test_metrics['r2']:.4f}")
        logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  MAE: {test_metrics['mae']:.4f}")

        if "success_criteria" in final_results:
            success = final_results["success_criteria"]["success"]
            logger.info(f"Success criteria:")
            logger.info(
                f"  R² ≥ 0.5: {'✅' if success['r2'] else '❌'} ({test_metrics['r2']:.4f})"
            )
            logger.info(
                f"  RMSE ≤ 50.0: {'✅' if success['rmse'] else '❌'} ({test_metrics['rmse']:.4f})"
            )
            logger.info(
                f"  MAE ≤ 30.0: {'✅' if success['mae'] else '❌'} ({test_metrics['mae']:.4f})"
            )
            logger.info(
                f"  Overall: {'🎉 SUCCESS!' if success['overall'] else '⚠️  Partially successful'}"
            )

    logger.info(f"Results saved to: {args.results_dir}")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization for Polymer GCN"
    )

    # HPO parameters
    parser.add_argument(
        "--method",
        type=str,
        default="random",
        choices=["grid", "random"],
        help="HPO method: grid or random search",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=50,
        help="Maximum number of trials (for random search)",
    )
    parser.add_argument(
        "--cv_folds", type=int, default=5, help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--primary_metric",
        type=str,
        default="r2",
        choices=["r2", "rmse", "mae"],
        help="Primary metric for optimization",
    )

    # Training parameters
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum epochs per HPO trial"
    )
    parser.add_argument(
        "--patience", type=int, default=8, help="Early stopping patience for HPO trials"
    )
    parser.add_argument(
        "--final_epochs",
        type=int,
        default=100,
        help="Maximum epochs for final training",
    )
    parser.add_argument(
        "--final_patience",
        type=int,
        default=15,
        help="Early stopping patience for final training",
    )

    # Data and system parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/filtered_tg_dataset.csv",
        help="Path to dataset CSV file",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use: cuda, cpu, or auto"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Advanced options
    parser.add_argument(
        "--save_all_models",
        action="store_true",
        help="Save all trial models (warning: storage intensive)",
    )

    args = parser.parse_args()

    # Run HPO
    try:
        final_results = run_hpo(args)
        logger.info("Hyperparameter optimization completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"HPO failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
