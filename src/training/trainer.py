"""
Trainer for Polymer Fingerprint Baseline Models

Handles training, validation, cross-validation, and evaluation
of polymer property prediction models.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PolymerBaselineTrainer:
    """
    Trainer class for polymer fingerprint baseline models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        results_dir: str = "results",
        model_name: str = "polymer_baseline",
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            device: Device to train on ('cuda' or 'cpu')
            results_dir: Directory to save results
            model_name: Name for saving models and results
        """
        # Handle device auto-detection
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        self.results_dir.mkdir(exist_ok=True)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_model_state = None

        logger.info(f"Initialized trainer on device: {device}")
        logger.info(f"Results will be saved to: {self.results_dir}")

    def train_epoch(
        self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        predictions = []
        targets = []

        for batch in tqdm(train_loader, desc="Training", leave=False):
            # Move batch to device
            fingerprints = batch["fingerprint"].to(self.device)
            chain_features = batch["chain_features"].to(self.device)
            target = batch["target"].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            pred = self.model(fingerprints, chain_features).squeeze()
            loss = criterion(pred, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions.extend(pred.detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(np.array(predictions), np.array(targets))
        metrics["loss"] = avg_loss

        return metrics

    def validate_epoch(
        self, val_loader: DataLoader, criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                # Move batch to device
                fingerprints = batch["fingerprint"].to(self.device)
                chain_features = batch["chain_features"].to(self.device)
                target = batch["target"].to(self.device)

                # Forward pass
                pred = self.model(fingerprints, chain_features).squeeze()
                loss = criterion(pred, target)

                # Track metrics
                total_loss += loss.item()
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())

        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(np.array(predictions), np.array(targets))
        metrics["loss"] = avg_loss

        return metrics

    def _calculate_metrics(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "mae": mean_absolute_error(targets, predictions),
            "r2": r2_score(targets, predictions),
        }

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        val_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (if None, will split from train_dataset)
            batch_size: Batch size
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            val_split: Validation split ratio if val_dataset is None

        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

        # Create validation dataset if not provided
        if val_dataset is None:
            train_size = int((1 - val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )
            logger.info(
                f"Split dataset: {train_size} train, {val_size} validation samples"
            )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup training components
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        no_improve_count = 0

        for epoch in range(epochs):
            # Train and validate
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            val_metrics = self.validate_epoch(val_loader, criterion)

            # Track history
            self.train_losses.append(train_metrics["loss"])
            self.val_losses.append(val_metrics["loss"])
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)

            # Learning rate scheduling
            scheduler.step(val_metrics["loss"])

            # Check for improvement
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = self.model.state_dict().copy()
                no_improve_count = 0

                # Save best model
                torch.save(
                    self.best_model_state,
                    self.results_dir / f"{self.model_name}_best.pth",
                )
                logger.info(
                    f"New best model saved (val_loss: {val_metrics['loss']:.4f})"
                )
            else:
                no_improve_count += 1

            # Logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val R²: {val_metrics['r2']:.4f}"
                )

            # Early stopping
            if no_improve_count >= patience:
                logger.info(
                    f"Early stopping triggered after {patience} epochs without improvement"
                )
                break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model state")

        # Final evaluation
        final_val_metrics = self.validate_epoch(val_loader, criterion)

        # Prepare results
        results = {
            "final_metrics": final_val_metrics,
            "best_val_loss": self.best_val_loss,
            "total_epochs": epoch + 1,
            "train_history": {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_metrics": self.train_metrics,
                "val_metrics": self.val_metrics,
            },
        }

        # Save training results
        self._save_results(results)
        self._plot_training_curves()

        return results

    def cross_validate(
        self,
        dataset: torch.utils.data.Dataset,
        n_folds: int = 5,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross validation.

        Args:
            dataset: Full dataset
            n_folds: Number of folds
            batch_size: Batch size
            epochs: Number of epochs per fold
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience

        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {n_folds}-fold cross validation")

        # Convert dataset to lists for sklearn compatibility
        smiles_list = [dataset[i]["smiles"] for i in range(len(dataset))]
        targets = [dataset[i]["target"].item() for i in range(len(dataset))]

        # Create KFold splitter
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_results = []
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(smiles_list)):
            logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")

            # Create fold datasets
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            # Reset model for each fold
            self.model.apply(self._reset_weights)

            # Train on fold
            self.train_losses = []
            self.val_losses = []
            self.train_metrics = []
            self.val_metrics = []
            self.best_val_loss = float("inf")

            fold_result = self.train(
                train_dataset=train_subset,
                val_dataset=val_subset,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=patience,
            )

            fold_results.append(fold_result)
            fold_metrics.append(fold_result["final_metrics"])

            logger.info(
                f"Fold {fold + 1} Results: "
                f"R²: {fold_result['final_metrics']['r2']:.4f}, "
                f"RMSE: {fold_result['final_metrics']['rmse']:.4f}, "
                f"MAE: {fold_result['final_metrics']['mae']:.4f}"
            )

        # Calculate cross-validation statistics
        cv_results = self._calculate_cv_statistics(fold_metrics)
        cv_results["fold_results"] = fold_results

        # Save cross-validation results (convert float32 to float for JSON serialization)
        def convert_float32(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_float32(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_float32(item) for item in obj]
            return obj

        serializable_cv_results = convert_float32(cv_results)
        with open(self.results_dir / f"{self.model_name}_cv_results.json", "w") as f:
            json.dump(serializable_cv_results, f, indent=2)

        logger.info(f"\nCross-Validation Results:")
        logger.info(f"R² = {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
        logger.info(
            f"RMSE = {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}"
        )
        logger.info(f"MAE = {cv_results['mean_mae']:.4f} ± {cv_results['std_mae']:.4f}")

        return cv_results

    def _reset_weights(self, m):
        """Reset model weights."""
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def _calculate_cv_statistics(
        self, fold_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate cross-validation statistics."""
        r2_scores = [m["r2"] for m in fold_metrics]
        rmse_scores = [m["rmse"] for m in fold_metrics]
        mae_scores = [m["mae"] for m in fold_metrics]

        return {
            "mean_r2": float(np.mean(r2_scores)),
            "std_r2": float(np.std(r2_scores)),
            "mean_rmse": float(np.mean(rmse_scores)),
            "std_rmse": float(np.std(rmse_scores)),
            "mean_mae": float(np.mean(mae_scores)),
            "std_mae": float(np.std(mae_scores)),
            "fold_scores": {"r2": r2_scores, "rmse": rmse_scores, "mae": mae_scores},
        }

    def _save_results(self, results: Dict[str, Any]):
        """Save training results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        # Handle list of dicts (metrics history)
                        serializable_results[key][k] = [
                            {mk: float(mv) for mk, mv in metric_dict.items()}
                            for metric_dict in v
                        ]
                    elif isinstance(v, list):
                        serializable_results[key][k] = [float(item) for item in v]
                    else:
                        serializable_results[key][k] = (
                            float(v) if isinstance(v, (int, float, np.number)) else v
                        )
            else:
                serializable_results[key] = (
                    float(value)
                    if isinstance(value, (int, float, np.number))
                    else value
                )

        with open(self.results_dir / f"{self.model_name}_results.json", "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(
            f"Results saved to {self.results_dir / f'{self.model_name}_results.json'}"
        )

    def _plot_training_curves(self):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{self.model_name.title()} Training Results")

        epochs = range(1, len(self.train_losses) + 1)

        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, label="Train Loss", color="blue")
        axes[0, 0].plot(epochs, self.val_losses, label="Val Loss", color="red")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # R² scores
        train_r2 = [m["r2"] for m in self.train_metrics]
        val_r2 = [m["r2"] for m in self.val_metrics]
        axes[0, 1].plot(epochs, train_r2, label="Train R²", color="blue")
        axes[0, 1].plot(epochs, val_r2, label="Val R²", color="red")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("R² Score")
        axes[0, 1].set_title("R² Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # RMSE
        train_rmse = [m["rmse"] for m in self.train_metrics]
        val_rmse = [m["rmse"] for m in self.val_metrics]
        axes[1, 0].plot(epochs, train_rmse, label="Train RMSE", color="blue")
        axes[1, 0].plot(epochs, val_rmse, label="Val RMSE", color="red")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("RMSE")
        axes[1, 0].set_title("Root Mean Square Error")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # MAE
        train_mae = [m["mae"] for m in self.train_metrics]
        val_mae = [m["mae"] for m in self.val_metrics]
        axes[1, 1].plot(epochs, train_mae, label="Train MAE", color="blue")
        axes[1, 1].plot(epochs, val_mae, label="Val MAE", color="red")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("MAE")
        axes[1, 1].set_title("Mean Absolute Error")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(
            self.results_dir / f"{self.model_name}_training_curves.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(
            f"Training curves saved to {self.results_dir / f'{self.model_name}_training_curves.png'}"
        )
