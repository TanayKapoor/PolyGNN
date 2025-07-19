"""
GCN Trainer for Polymer Property Prediction

Specialized trainer for Graph Convolutional Network models using PyTorch Geometric.
Handles graph data loading, training, validation, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class PolymerGCNTrainer:
    """
    Trainer class for polymer GCN models using PyTorch Geometric.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 results_dir: str = 'results',
                 model_name: str = 'polymer_gcn'):
        """
        Initialize the GCN trainer.
        
        Args:
            model: PyTorch GCN model to train
            device: Device to train on ('cuda', 'cpu', or 'auto')
            results_dir: Directory to save results
            model_name: Name for saving models and results
        """
        # Handle device auto-detection
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized GCN trainer on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train(self,
              train_dataset,
              val_dataset,
              batch_size: int = 32,
              epochs: int = 100,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4,
              patience: int = 10,
              min_delta: float = 1e-4) -> Dict[str, Any]:
        """
        Train the GCN model.
        
        Args:
            train_dataset: Training dataset (PolymerTgDataset or PolymerGCNDataset)
            val_dataset: Validation dataset
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization weight
            patience: Early stopping patience
            min_delta: Minimum change for improvement
            
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting GCN training...")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Training tracking
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss, train_metric = self._train_epoch(train_loader, optimizer, criterion)
            train_losses.append(train_loss)
            train_metrics.append(train_metric)
            
            # Validation phase
            val_loss, val_metric = self._validate_epoch(val_loader, criterion)
            val_losses.append(val_loss)
            val_metrics.append(val_metric)
            
            # Log progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}:")
                logger.info(f"  Train Loss: {train_loss:.4f}, R²: {train_metric['r2']:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}, R²: {val_metric['r2']:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Training summary
        final_epoch = epoch + 1
        final_metrics = val_metrics[-1]
        
        results = {
            'final_metrics': final_metrics,
            'best_val_loss': best_val_loss,
            'total_epochs': final_epoch,
            'train_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
        }
        
        # Save results
        self._save_results(results)
        self._plot_training_curves(train_losses, val_losses, train_metrics, val_metrics)
        
        logger.info(f"Training completed in {final_epoch} epochs")
        logger.info(f"Best validation R²: {max(m['r2'] for m in val_metrics):.4f}")
        
        return results
    
    def _train_epoch(self, train_loader, optimizer, criterion) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(batch)
            loss = criterion(output.squeeze(), batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            predictions.extend(output.squeeze().detach().cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(predictions, targets, avg_loss)
        
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader, criterion) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                batch = batch.to(self.device)
                
                # Forward pass
                output = self.model(batch)
                loss = criterion(output.squeeze(), batch.y)
                
                # Accumulate metrics
                total_loss += loss.item()
                predictions.extend(output.squeeze().cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(predictions, targets, avg_loss)
        
        return avg_loss, metrics
    
    def validate_epoch(self, data_loader, criterion) -> Dict[str, float]:
        """Public method for validation (for compatibility with existing code)."""
        _, metrics = self._validate_epoch(data_loader, criterion)
        return metrics
    
    def _calculate_metrics(self, predictions: List[float], targets: List[float], loss: float) -> Dict[str, float]:
        """Calculate regression metrics."""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'loss': float(loss)
        }
    
    def predict(self, dataset, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a dataset.
        
        Args:
            dataset: Dataset to predict on
            batch_size: Batch size for prediction
            
        Returns:
            Tuple of (predictions, targets)
        """
        self.model.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting", leave=False):
                batch = batch.to(self.device)
                output = self.model(batch)
                
                predictions.extend(output.squeeze().cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        return np.array(predictions), np.array(targets)
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_path = self.results_dir / f"{self.model_name}_best.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results to JSON."""
        results_path = self.results_dir / f"{self.model_name}_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def _plot_training_curves(self, train_losses, val_losses, train_metrics, val_metrics):
        """Plot and save training curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # R² curves
        train_r2 = [m['r2'] for m in train_metrics]
        val_r2 = [m['r2'] for m in val_metrics]
        ax2.plot(epochs, train_r2, 'b-', label='Train R²')
        ax2.plot(epochs, val_r2, 'r-', label='Validation R²')
        ax2.set_title('R² Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R²')
        ax2.legend()
        ax2.grid(True)
        
        # RMSE curves
        train_rmse = [m['rmse'] for m in train_metrics]
        val_rmse = [m['rmse'] for m in val_metrics]
        ax3.plot(epochs, train_rmse, 'b-', label='Train RMSE')
        ax3.plot(epochs, val_rmse, 'r-', label='Validation RMSE')
        ax3.set_title('Root Mean Square Error')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('RMSE')
        ax3.legend()
        ax3.grid(True)
        
        # MAE curves
        train_mae = [m['mae'] for m in train_metrics]
        val_mae = [m['mae'] for m in val_metrics]
        ax4.plot(epochs, train_mae, 'b-', label='Train MAE')
        ax4.plot(epochs, val_mae, 'r-', label='Validation MAE')
        ax4.set_title('Mean Absolute Error')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MAE')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"{self.model_name}_training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {plot_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {checkpoint_path}")
        return checkpoint.get('val_loss', None)
    
    def cross_validate(self,
                      dataset,
                      n_folds: int = 5,
                      batch_size: int = 32,
                      epochs: int = 50,
                      learning_rate: float = 0.001,
                      weight_decay: float = 1e-4,
                      patience: int = 8) -> Dict[str, Any]:
        """
        Perform cross-validation on the dataset.
        
        Args:
            dataset: Full dataset for cross-validation
            n_folds: Number of folds
            batch_size: Batch size for training
            epochs: Maximum epochs per fold
            learning_rate: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {n_folds}-fold cross-validation...")
        
        # Convert dataset to list for splitting
        data_list = [dataset[i] for i in range(len(dataset))]
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(data_list)):
            logger.info(f"Training fold {fold + 1}/{n_folds}...")
            
            # Create fold datasets
            train_fold = [data_list[i] for i in train_idx]
            val_fold = [data_list[i] for i in val_idx]
            
            # Reset model weights
            self.model.apply(self._reset_weights)
            
            # Train on fold
            results = self.train(
                train_dataset=train_fold,
                val_dataset=val_fold,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=patience
            )
            
            fold_results.append(results['final_metrics'])
        
        # Calculate cross-validation statistics
        cv_results = self._calculate_cv_statistics(fold_results)
        
        # Save CV results
        cv_path = self.results_dir / f"{self.model_name}_cv_results.json"
        with open(cv_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        logger.info("Cross-validation completed!")
        logger.info(f"Mean R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
        
        return cv_results
    
    def _reset_weights(self, module):
        """Reset model weights for cross-validation."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.reset_parameters()
    
    def _calculate_cv_statistics(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Calculate cross-validation statistics."""
        metrics = ['r2', 'rmse', 'mae']
        cv_results = {}
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results]
            cv_results[f'mean_{metric}'] = float(np.mean(values))
            cv_results[f'std_{metric}'] = float(np.std(values))
        
        cv_results['fold_scores'] = {
            metric: [fold[metric] for fold in fold_results]
            for metric in metrics
        }
        
        return cv_results 