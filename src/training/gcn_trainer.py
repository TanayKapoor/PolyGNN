"""
GCN Trainer for Polymer Property Prediction

Specialized trainer for Graph Convolutional Network models using PyTorch Geometric.
Handles graph data loading, training, validation, evaluation, and hyperparameter optimization.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
from itertools import product
from pathlib import Path

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback to colorama for basic colors
    try:
        from colorama import init, Fore, Back, Style
        init()
        COLORAMA_AVAILABLE = True
    except ImportError:
        COLORAMA_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configure matplotlib for non-interactive backend
plt.switch_backend('Agg')

logger = logging.getLogger(__name__)

class BeautifulOutput:
    """Handle beautiful console output with fallbacks."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        
    def print_header(self, text: str, style: str = "bold blue"):
        """Print a beautiful header."""
        if RICH_AVAILABLE:
            self.console.print(Panel(Text(text, style=style), expand=False))
        elif COLORAMA_AVAILABLE:
            print(f"\n{Fore.BLUE}{'='*60}")
            print(f"{Fore.CYAN}{text}")
            print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        else:
            print(f"\n{'='*60}")
            print(text)
            print('='*60)
    
    def print_section(self, text: str):
        """Print a section header."""
        if RICH_AVAILABLE:
            self.console.print(f"\n[bold cyan]{text}[/bold cyan]")
        elif COLORAMA_AVAILABLE:
            print(f"\n{Fore.CYAN}{text}{Style.RESET_ALL}")
        else:
            print(f"\n{text}")
    
    def print_success(self, text: str):
        """Print success message."""
        if RICH_AVAILABLE:
            self.console.print(f"✅ [green]{text}[/green]")
        elif COLORAMA_AVAILABLE:
            print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")
        else:
            print(f"✅ {text}")
    
    def print_warning(self, text: str):
        """Print warning message."""
        if RICH_AVAILABLE:
            self.console.print(f"⚠️  [yellow]{text}[/yellow]")
        elif COLORAMA_AVAILABLE:
            print(f"{Fore.YELLOW}⚠️  {text}{Style.RESET_ALL}")
        else:
            print(f"⚠️ {text}")
    
    def print_error(self, text: str):
        """Print error message."""
        if RICH_AVAILABLE:
            self.console.print(f"❌ [red]{text}[/red]")
        elif COLORAMA_AVAILABLE:
            print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")
        else:
            print(f"❌ {text}")
    
    def print_info(self, text: str):
        """Print info message."""
        if RICH_AVAILABLE:
            self.console.print(f"ℹ️  [blue]{text}[/blue]")
        elif COLORAMA_AVAILABLE:
            print(f"{Fore.BLUE}ℹ️  {text}{Style.RESET_ALL}")
        else:
            print(f"ℹ️ {text}")
    
    def print_metrics_table(self, metrics: Dict[str, float], title: str = "Performance Metrics"):
        """Print metrics in a beautiful table."""
        if RICH_AVAILABLE:
            table = Table(title=title)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Status", style="yellow")
            
            for metric, value in metrics.items():
                if metric.lower() == 'r²':
                    status = "✅ Excellent" if value >= 0.7 else "✅ Good" if value >= 0.5 else "⚠️ Poor"
                elif 'rmse' in metric.lower():
                    status = "✅ Excellent" if value <= 40 else "✅ Good" if value <= 60 else "⚠️ High"
                elif 'mae' in metric.lower():
                    status = "✅ Excellent" if value <= 25 else "✅ Good" if value <= 40 else "⚠️ High"
                else:
                    status = ""
                
                table.add_row(metric, f"{value:.4f}", status)
            
            self.console.print(table)
        else:
            print(f"\n{title}:")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"{metric:15}: {value:.4f}")
            print("-" * 40)

# Initialize beautiful output
beautiful = BeautifulOutput()

class PolymerGCNTrainer:
    """
    Enhanced trainer for Polymer GCN models with beautiful output and comprehensive HPO.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'auto',
                 results_dir: str = 'results',
                 random_seed: int = 42,
                 model_name: str = "polymer_gcn"):
        """
        Initialize the GCN trainer.
        
        Args:
            model: The GCN model to train
            device: Device to use ('cpu', 'cuda', or 'auto')
            results_dir: Directory to save results
            random_seed: Random seed for reproducibility
            model_name: Name for model files and results
        """
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.model_name = model_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # HPO results subdirectory
        self.hpo_results_dir = self.results_dir / "hpo"
        self.hpo_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Beautiful startup message
        beautiful.print_header("🧬 POLYMER GCN TRAINER INITIALIZED")
        beautiful.print_info(f"Device: {self.device}")
        beautiful.print_info(f"Model parameters: {total_params:,}")
        beautiful.print_info(f"Results directory: {self.results_dir}")
        
        logger.info(f"Initialized GCN trainer on {self.device}")
        logger.info(f"Model parameters: {total_params:,}")
    
    def _create_epoch_display(self, epoch: int, max_epochs: int, 
                            train_metrics: Dict[str, float], 
                            val_metrics: Dict[str, float],
                            elapsed_time: float, 
                            eta: float) -> str:
        """Create beautiful epoch display."""
        if RICH_AVAILABLE:
            # Create a rich layout for epoch info
            progress_bar = "█" * int(20 * epoch / max_epochs) + "░" * (20 - int(20 * epoch / max_epochs))
            
            layout = f"""
[bold blue]Epoch {epoch}/{max_epochs}[/bold blue] [{progress_bar}] {epoch/max_epochs*100:.1f}%

[cyan]Training:[/cyan]
  Loss: [green]{train_metrics.get('loss', 0):.4f}[/green] | R²: [green]{train_metrics.get('r2', 0):.4f}[/green]

[cyan]Validation:[/cyan]
  Loss: [yellow]{val_metrics.get('loss', 0):.4f}[/yellow] | R²: [yellow]{val_metrics.get('r2', 0):.4f}[/yellow]

[dim]Time: {elapsed_time:.1f}s | ETA: {eta:.1f}s[/dim]
"""
            return layout
        else:
            progress_bar = "█" * int(20 * epoch / max_epochs) + "░" * (20 - int(20 * epoch / max_epochs))
            return f"""
Epoch {epoch}/{max_epochs} [{progress_bar}] {epoch/max_epochs*100:.1f}%
Training  - Loss: {train_metrics.get('loss', 0):.4f} | R²: {train_metrics.get('r2', 0):.4f}
Validation- Loss: {val_metrics.get('loss', 0):.4f} | R²: {val_metrics.get('r2', 0):.4f}
Time: {elapsed_time:.1f}s | ETA: {eta:.1f}s
"""

    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch with beautiful progress display."""
        self.model.train()
        total_loss = 0.0
        predictions = []
        targets = []
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            out = self.model(batch)
            loss = criterion(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(out.squeeze().detach().cpu().numpy())
            targets.extend(batch.y.detach().cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        r2 = r2_score(targets, predictions)
        
        return avg_loss, r2
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                output = self.model(batch)
                loss = criterion(output.squeeze(), batch.y)
                
                total_loss += loss.item()
                predictions.extend(output.squeeze().cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        
        return {
            'loss': avg_loss,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'targets': targets
        }
    
    def train(self, train_dataset, val_dataset, 
              learning_rate: float = 0.001,
              batch_size: int = 32,
              max_epochs: int = 100,
              patience: int = 15,
              weight_decay: float = 1e-4,
              save_name: str = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model with beautiful progress display and detailed metrics.
        """
        # Beautiful training header
        beautiful.print_header("🚀 STARTING POLYMER GCN TRAINING")
        
        # Training info
        train_info = {
            'Training samples': len(train_dataset),
            'Validation samples': len(val_dataset),
            'Batch size': batch_size,
            'Learning rate': learning_rate,
            'Max epochs': max_epochs,
            'Patience': patience,
            'Weight decay': weight_decay
        }
        
        if RICH_AVAILABLE:
            table = Table(title="Training Configuration")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")
            
            for param, value in train_info.items():
                table.add_row(param, str(value))
            
            beautiful.console.print(table)
        else:
            beautiful.print_section("Training Configuration")
            for param, value in train_info.items():
                print(f"  {param}: {value}")
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Check for polymer features
        sample = train_dataset[0]
        has_polymer_features = hasattr(sample, 'polymer_features')
        
        if has_polymer_features:
            beautiful.print_info(f"Using polymer features: dimension {sample.polymer_features.shape[0]}")
        else:
            beautiful.print_info("No polymer features detected")
        
        # Setup training
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_r2': [],
            'val_loss': [],
            'val_r2': [],
            'val_rmse': [],
            'val_mae': []
        }
        
        best_val_r2 = -float('inf')
        patience_counter = 0
        start_time = time.time()
        
        beautiful.print_section("Training Progress")
        
        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss, train_r2 = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader, criterion)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_r2'].append(train_r2)
            history['val_loss'].append(val_metrics['loss'])
            history['val_r2'].append(val_metrics['r2'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_mae'].append(val_metrics['mae'])
            
            # Calculate timing
            epoch_time = time.time() - epoch_start
            elapsed_total = time.time() - start_time
            avg_epoch_time = elapsed_total / epoch
            eta = avg_epoch_time * (max_epochs - epoch)
            
            # Beautiful epoch display every 10 epochs or significant milestones
            if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
                train_metrics = {'loss': train_loss, 'r2': train_r2}
                val_display_metrics = {'loss': val_metrics['loss'], 'r2': val_metrics['r2']}
                
                if RICH_AVAILABLE:
                    display = self._create_epoch_display(
                        epoch, max_epochs, train_metrics, val_display_metrics, 
                        elapsed_total, eta
                    )
                    beautiful.console.print(Panel(display, title=f"[bold]Epoch {epoch}[/bold]"), end="")
                else:
                    print(self._create_epoch_display(
                        epoch, max_epochs, train_metrics, val_display_metrics, 
                        elapsed_total, eta
                    ), end="")
            
            # Early stopping check
            if val_metrics['r2'] > best_val_r2:
                # Only show significant improvements (>1% or every 10 epochs)
                improvement = val_metrics['r2'] - best_val_r2
                if epoch > 1 and (improvement > 0.01 or epoch % 10 == 0):
                    beautiful.print_success(f"New best validation R²: {val_metrics['r2']:.4f}")
                
                best_val_r2 = val_metrics['r2']
                patience_counter = 0
                
                # Save best model
                if save_name:
                    model_path = self.results_dir / f"{save_name}_best_model.pth"
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_val_r2': best_val_r2,
                        'history': history
                    }, model_path)
                
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                beautiful.print_warning(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Training completion summary
        total_time = time.time() - start_time
        
        beautiful.print_header("📊 TRAINING COMPLETED")
        
        # Metrics for cross-validation statistics (with expected keys)
        final_metrics = {
            'r2': best_val_r2,
            'rmse': history['val_rmse'][-1], 
            'mae': history['val_mae'][-1]
        }
        
        # Display metrics for the beautiful table
        display_metrics = {
            'Final Train R²': history['train_r2'][-1],
            'Best Val R²': best_val_r2,
            'Final Val RMSE': history['val_rmse'][-1],
            'Final Val MAE': history['val_mae'][-1],
            'Epochs Trained': len(history['train_loss']),
            'Training Time (min)': total_time / 60
        }
        
        beautiful.print_metrics_table(display_metrics, "Training Summary")
        
        # Save training results
        if save_name:
            results = {
                'history': history,
                'final_metrics': final_metrics,  # For CV statistics
                'display_metrics': display_metrics,  # For display/saving
                'best_val_r2': float(best_val_r2),
                'total_epochs': len(history['train_loss']),
                'training_time_seconds': total_time
            }
            
            results_path = self.results_dir / f"{save_name}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            beautiful.print_success(f"Results saved to {results_path}")
            
            # Save training curves
            self._save_training_curves(history, save_name)
            beautiful.print_success(f"Training curves saved to {self.results_dir / f'{save_name}_training_curves.png'}")
        
        logger.info(f"Training completed in {len(history['train_loss'])} epochs")
        logger.info(f"Best validation R²: {best_val_r2:.4f}")
        
        # Always return complete results including final_metrics
        results = {
            'history': history,
            'final_metrics': final_metrics,  # For CV statistics
            'display_metrics': display_metrics,  # For display/saving
            'best_val_r2': float(best_val_r2),
            'total_epochs': len(history['train_loss']),
            'training_time_seconds': total_time
        }
        
        return results
    
    def _save_training_curves(self, history: Dict[str, List[float]], save_name: str):
        """Save training curves plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R² curves
        ax2.plot(epochs, history['train_r2'], 'b-', label='Training R²')
        ax2.plot(epochs, history['val_r2'], 'r-', label='Validation R²')
        ax2.set_title('Training and Validation R²')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R²')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # RMSE curves
        ax3.plot(epochs, history['val_rmse'], 'g-', label='Validation RMSE')
        ax3.set_title('Validation RMSE')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('RMSE')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # MAE curves
        ax4.plot(epochs, history['val_mae'], 'm-', label='Validation MAE')
        ax4.set_title('Validation MAE')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MAE')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f'{save_name}_training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
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
            for batch in val_loader:
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
            for batch in data_loader:
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
                max_epochs=epochs,
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

    def hyperparam_optimize(self,
                           dataset,
                           param_grid: Dict[str, List],
                           method: str = 'grid',
                           n_trials: int = 50,
                           cv_folds: int = 5,
                           max_epochs: int = 50,
                           patience: int = 8,
                           primary_metric: str = 'r2',
                           minimize_primary: bool = False,
                           save_all_models: bool = False,
                           parallel: bool = False,
                           random_seed: int = 42) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using grid search or random search.
        
        Args:
            dataset: Full dataset for cross-validation
            param_grid: Dictionary of parameter lists to search over
            method: 'grid' for exhaustive search, 'random' for random sampling
            n_trials: Number of trials for random search (ignored for grid search)
            cv_folds: Number of cross-validation folds
            max_epochs: Maximum epochs per trial
            patience: Early stopping patience
            primary_metric: Primary metric for ranking ('r2', 'rmse', 'mae')
            minimize_primary: Whether to minimize primary metric (True for rmse/mae)
            save_all_models: Whether to save all trial models (warning: storage intensive)
            parallel: Whether to use parallel processing (experimental)
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing HPO results and best configuration
        """
        beautiful.print_header("🎯 HYPERPARAMETER OPTIMIZATION")
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Setup HPO tracking
        start_time = time.time()
        hpo_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        hpo_dir = self.hpo_results_dir / f"hpo_{hpo_id}"
        hpo_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate parameter combinations
        if method == 'grid':
            param_combinations = self._generate_grid_combinations(param_grid)
            beautiful.print_info(f"Grid search: {len(param_combinations)} total combinations")
        elif method == 'random':
            param_combinations = self._generate_random_combinations(param_grid, n_trials, random_seed)
            beautiful.print_info(f"Random search: {n_trials} trials")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'grid' or 'random'")
        
        # Show HPO configuration
        if RICH_AVAILABLE:
            config_table = Table(title="HPO Configuration")
            config_table.add_column("Parameter", style="cyan")
            config_table.add_column("Value", style="green")
            
            config_items = {
                'Method': method,
                'Total Trials': len(param_combinations),
                'CV Folds': cv_folds,
                'Max Epochs': max_epochs,
                'Patience': patience,
                'Primary Metric': primary_metric,
                'Random Seed': random_seed
            }
            
            for param, value in config_items.items():
                config_table.add_row(param, str(value))
            
            beautiful.console.print(config_table)
        else:
            beautiful.print_section("HPO Configuration")
            print(f"  Method: {method}")
            print(f"  Total Trials: {len(param_combinations)}")
            print(f"  CV Folds: {cv_folds}")
            print(f"  Max Epochs: {max_epochs}")
            print(f"  Patience: {patience}")
            print(f"  Primary Metric: {primary_metric}")
        
        # Initialize results tracking
        all_results = []
        best_score = -np.inf if not minimize_primary else np.inf
        best_params = None
        best_model_state = None
        
        # Setup CSV logging
        csv_path = hpo_dir / "hpo_results.csv"
        csv_fieldnames = ['trial_id', 'params'] + [f'cv_{metric}_{stat}' 
                         for metric in ['r2', 'rmse', 'mae'] 
                         for stat in ['mean', 'std']] + ['trial_time', 'status', 'error']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()
            
            # HPO main loop with beautiful progress
            beautiful.print_section("Running Optimization Trials")
            
            # Common function to handle each trial
            def process_trial(trial_id, params, progress_callback=None):
                nonlocal best_score, best_params, best_model_state, all_results
                
                if progress_callback:
                    progress_callback()
                
                beautiful.print_info(f"\nTrial {trial_id + 1}/{len(param_combinations)}")
                beautiful.print_info(f"Parameters: {params}")
                
                trial_start = time.time()
                
                try:
                    # Run cross-validation with current parameters
                    cv_results = self._run_hpo_trial(
                        dataset=dataset,
                        params=params,
                        cv_folds=cv_folds,
                        max_epochs=max_epochs,
                        patience=patience,
                        trial_id=trial_id,
                        hpo_dir=hpo_dir,
                        save_model=save_all_models
                    )
                    
                    trial_time = time.time() - trial_start
                    current_score = cv_results[f'mean_{primary_metric}']
                    
                    # Check if this is the best trial
                    is_best = (minimize_primary and current_score < best_score) or \
                             (not minimize_primary and current_score > best_score)
                    
                    if is_best:
                        best_score = current_score
                        best_params = params.copy()
                        if hasattr(self, '_current_model_state'):
                            best_model_state = self._current_model_state.copy()
                        beautiful.print_success(f"NEW BEST! {primary_metric}: {current_score:.4f}")
                    
                    # Log results
                    result_row = {
                        'trial_id': trial_id,
                        'params': str(params),
                        'trial_time': f"{trial_time:.2f}s",
                        'status': 'success'
                    }
                    
                    for metric in ['r2', 'rmse', 'mae']:
                        result_row[f'cv_{metric}_mean'] = cv_results[f'mean_{metric}']
                        result_row[f'cv_{metric}_std'] = cv_results[f'std_{metric}']
                    
                    writer.writerow(result_row)
                    csvfile.flush()
                    
                    # Store detailed results
                    trial_result = {
                        'trial_id': trial_id,
                        'params': params,
                        'cv_results': cv_results,
                        'primary_score': current_score,
                        'is_best': is_best,
                        'trial_time': trial_time
                    }
                    all_results.append(trial_result)
                    
                    beautiful.print_info(f"Trial completed in {trial_time:.2f}s")
                    beautiful.print_info(f"R²: {cv_results['mean_r2']:.4f}±{cv_results['std_r2']:.4f}, "
                           f"RMSE: {cv_results['mean_rmse']:.4f}±{cv_results['std_rmse']:.4f}, "
                           f"MAE: {cv_results['mean_mae']:.4f}±{cv_results['std_mae']:.4f}")
                    
                    return True
                    
                except Exception as e:
                    beautiful.print_error(f"Trial {trial_id} failed: {str(e)}")
                    trial_time = time.time() - trial_start
                    
                    # Log failed trial
                    result_row = {
                        'trial_id': trial_id,
                        'params': str(params),
                        'trial_time': f"{trial_time:.2f}s",
                        'status': 'failed',
                        'error': str(e)
                    }
                    
                    # Add empty metric columns for consistency
                    for metric in ['r2', 'rmse', 'mae']:
                        result_row[f'cv_{metric}_mean'] = np.nan
                        result_row[f'cv_{metric}_std'] = np.nan
                    
                    writer.writerow(result_row)
                    csvfile.flush()
                    
                    return False
            
            # Run trials with appropriate progress display
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=beautiful.console
                ) as progress:
                    task = progress.add_task("HPO Progress", total=len(param_combinations))
                    
                    for trial_id, params in enumerate(param_combinations):
                        def update_progress():
                            progress.update(task, 
                                          description=f"Trial {trial_id+1}/{len(param_combinations)}",
                                          advance=1)
                        process_trial(trial_id, params, update_progress)
            else:
                for trial_id, params in enumerate(tqdm(param_combinations, desc="HPO Progress")):
                    process_trial(trial_id, params)

        
        total_time = time.time() - start_time
        
        # Compile final results
        hpo_results = {
            'method': method,
            'total_trials': len(param_combinations),
            'successful_trials': len(all_results),
            'total_time': total_time,
            'best_params': best_params,
            'best_score': best_score,
            'primary_metric': primary_metric,
            'all_results': all_results,
            'hpo_id': hpo_id,
            'timestamp': datetime.now().isoformat(),
            'param_grid': param_grid
        }
        
        # Save comprehensive results
        results_path = hpo_dir / "hpo_summary.json"
        with open(results_path, 'w') as f:
            json.dump(hpo_results, f, indent=2, default=str)
        
        # Save best model state if available
        if best_model_state is not None:
            best_model_path = hpo_dir / "best_model.pth"
            torch.save(best_model_state, best_model_path)
            logger.info(f"Best model saved to {best_model_path}")
        
        # Beautiful completion summary
        beautiful.print_header("🏆 OPTIMIZATION COMPLETED")
        
        if all_results:
            completion_summary = {
                'Total Time (hours)': total_time / 3600,
                'Best Score': best_score,
                'Successful Trials': len(all_results),
                'Total Trials': len(param_combinations),
                'Success Rate (%)': len(all_results) / len(param_combinations) * 100
            }
            
            beautiful.print_metrics_table(completion_summary, "HPO Summary")
            
            if RICH_AVAILABLE:
                # Best parameters table
                best_table = Table(title=f"Best Parameters ({primary_metric}: {best_score:.4f})")
                best_table.add_column("Parameter", style="cyan")
                best_table.add_column("Value", style="green")
                
                for param, value in best_params.items():
                    best_table.add_row(param, str(value))
                
                beautiful.console.print(best_table)
            else:
                beautiful.print_section(f"Best Parameters ({primary_metric}: {best_score:.4f})")
                for param, value in best_params.items():
                    print(f"  {param}: {value}")
        else:
            beautiful.print_error("No successful trials completed!")
        
        beautiful.print_success(f"Results saved to: {hpo_dir}")
        
        logger.info("=" * 60)
        logger.info("HYPERPARAMETER OPTIMIZATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Best {primary_metric}: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Results saved to: {hpo_dir}")
        
        return hpo_results
    
    def _generate_grid_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all possible parameter combinations for grid search."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _generate_random_combinations(self, param_grid: Dict[str, List], 
                                    n_trials: int, random_seed: int) -> List[Dict]:
        """Generate random parameter combinations for random search."""
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        combinations = []
        
        for _ in range(n_trials):
            combo = {}
            for param, values in param_grid.items():
                # Use random.choice for any parameter values (handles nested lists)
                combo[param] = random.choice(values)
            combinations.append(combo)
        
        return combinations
    
    def _run_hpo_trial(self, dataset, params: Dict, cv_folds: int, 
                      max_epochs: int, patience: int, trial_id: int,
                      hpo_dir: Path, save_model: bool = False) -> Dict[str, float]:
        """Run a single HPO trial with cross-validation."""
        
        # Create model with trial parameters
        from ..models.polymer_gcn import PolymerGCN, create_gcn_model_from_config
        
        # Determine node feature dimension from dataset
        try:
            sample = dataset[0] if hasattr(dataset, '__getitem__') else next(iter(dataset))
            node_feature_dim = sample.x.shape[1] if hasattr(sample, 'x') else 157
            logger.info(f"Using node feature dimension: {node_feature_dim}")
        except Exception as e:
            logger.warning(f"Could not determine node feature dimension from dataset: {e}, using fallback 157")
            node_feature_dim = 157
        
        # Create model config
        model_config = {
            'hidden_dims': params.get('hidden_dims', [128, 64, 32]),
            'num_gcn_layers': params.get('num_gcn_layers', 3),
            'dropout_rate': params.get('dropout_rate', 0.2),
            'pooling_method': params.get('pooling_method', 'mean'),
            'activation': params.get('activation', 'relu'),
            'use_molecular_features': params.get('use_molecular_features', True),
            'molecular_feature_dim': params.get('molecular_feature_dim', 13),
            'use_polymer_features': params.get('use_polymer_features', True),
            'polymer_feature_dim': params.get('polymer_feature_dim', 130)
        }
        
        # Create fresh model instance for this trial
        trial_model = create_gcn_model_from_config(model_config, node_feature_dim)
        
        # Create temporary trainer for this trial
        temp_trainer = PolymerGCNTrainer(
            model=trial_model,
            device=self.device,
            results_dir=str(hpo_dir / f"trial_{trial_id}"),
            model_name=f"trial_{trial_id}"
        )
        
        # Run cross-validation
        cv_results = temp_trainer.cross_validate(
            dataset=dataset,
            n_folds=cv_folds,
            batch_size=params.get('batch_size', 32),
            epochs=max_epochs,
            learning_rate=params.get('learning_rate', 0.001),
            weight_decay=params.get('weight_decay', 1e-4),
            patience=patience
        )
        
        # Store model state for potential reuse
        if save_model or cv_results['mean_r2'] > getattr(self, '_best_trial_r2', -np.inf):
            self._current_model_state = trial_model.state_dict()
            self._best_trial_r2 = cv_results['mean_r2']
        
        return cv_results
    
    def retrain_with_best_params(self, 
                                hpo_results: Dict[str, Any],
                                full_dataset,
                                test_dataset=None,
                                max_epochs: int = 100,
                                patience: int = 15,
                                save_final_model: bool = True) -> Dict[str, Any]:
        """
        Retrain model with best hyperparameters on full dataset.
        
        Args:
            hpo_results: Results from hyperparameter optimization
            full_dataset: Full training dataset (train + val)
            test_dataset: Test dataset for final evaluation
            max_epochs: Maximum epochs for final training
            patience: Early stopping patience
            save_final_model: Whether to save the final model
            
        Returns:
            Final training and test results
        """
        beautiful.print_header("🎯 RETRAINING WITH BEST PARAMETERS")
        
        best_params = hpo_results['best_params']
        primary_metric = hpo_results['primary_metric']
        best_cv_score = hpo_results['best_score']
        
        if RICH_AVAILABLE:
            # Best parameters table for final training
            retrain_table = Table(title=f"Final Training Configuration (CV {primary_metric}: {best_cv_score:.4f})")
            retrain_table.add_column("Parameter", style="cyan")
            retrain_table.add_column("Value", style="green")
            
            for param, value in best_params.items():
                retrain_table.add_row(param, str(value))
            
            beautiful.console.print(retrain_table)
        else:
            beautiful.print_section(f"Final Training Configuration (CV {primary_metric}: {best_cv_score:.4f})")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
        
        logger.info("=" * 50)
        logger.info("RETRAINING WITH BEST PARAMETERS")
        logger.info("=" * 50)
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV {primary_metric}: {best_cv_score:.4f}")
        
        # Create model with best parameters
        from ..models.polymer_gcn import create_gcn_model_from_config
        
        sample = full_dataset[0] if hasattr(full_dataset, '__getitem__') else next(iter(full_dataset))
        node_feature_dim = sample.x.shape[1] if hasattr(sample, 'x') else 157
        
        model_config = {
            'hidden_dims': best_params.get('hidden_dims', [128, 64, 32]),
            'num_gcn_layers': best_params.get('num_gcn_layers', 3),
            'dropout_rate': best_params.get('dropout_rate', 0.2),
            'pooling_method': best_params.get('pooling_method', 'mean'),
            'activation': best_params.get('activation', 'relu'),
            'use_molecular_features': best_params.get('use_molecular_features', True),
            'molecular_feature_dim': best_params.get('molecular_feature_dim', 13),
            'use_polymer_features': best_params.get('use_polymer_features', True),
            'polymer_feature_dim': best_params.get('polymer_feature_dim', 130)
        }
        
        final_model = create_gcn_model_from_config(model_config, node_feature_dim)
        
        # Create final trainer
        final_trainer = PolymerGCNTrainer(
            model=final_model,
            device=self.device,
            results_dir=str(self.results_dir / "final_model"),
            model_name="final_optimized_model"
        )
        
        # Split full dataset for final training (80% train, 20% val)
        n_samples = len(full_dataset)
        n_train = int(0.8 * n_samples)
        
        indices = list(range(n_samples))
        np.random.shuffle(indices)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_subset = [full_dataset[i] for i in train_indices]
        val_subset = [full_dataset[i] for i in val_indices]
        
        # Final training
        logger.info(f"Final training: {len(train_subset)} train, {len(val_subset)} validation")
        
        training_results = final_trainer.train(
            train_dataset=train_subset,
            val_dataset=val_subset,
            batch_size=best_params.get('batch_size', 32),
            max_epochs=max_epochs,
            learning_rate=best_params.get('learning_rate', 0.001),
            weight_decay=best_params.get('weight_decay', 1e-4),
            patience=patience
        )
        
        final_results = {
            'hpo_results': hpo_results,
            'final_training': training_results,
            'best_params': best_params
        }
        
        # Test evaluation if test set provided
        if test_dataset is not None:
            logger.info("Evaluating on test set...")
            test_predictions, test_targets = final_trainer.predict(test_dataset)
            
            test_metrics = final_trainer._calculate_metrics(
                test_predictions.tolist(), 
                test_targets.tolist(), 
                0.0  # placeholder loss
            )
            
            final_results['test_metrics'] = test_metrics
            
            logger.info("=" * 50)
            logger.info("FINAL TEST RESULTS")
            logger.info("=" * 50)
            logger.info(f"Test R²: {test_metrics['r2']:.4f}")
            logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
            
            # Check success criteria
            success_criteria = {
                'r2_target': 0.5,
                'rmse_target': 50.0,
                'mae_target': 30.0
            }
            
            r2_success = test_metrics['r2'] >= success_criteria['r2_target']
            rmse_success = test_metrics['rmse'] <= success_criteria['rmse_target']
            mae_success = test_metrics['mae'] <= success_criteria['mae_target']
            
            all_success = r2_success and rmse_success and mae_success
            
            final_results['success_criteria'] = {
                'targets': success_criteria,
                'achieved': {
                    'r2': test_metrics['r2'],
                    'rmse': test_metrics['rmse'],
                    'mae': test_metrics['mae']
                },
                'success': {
                    'r2': r2_success,
                    'rmse': rmse_success,
                    'mae': mae_success,
                    'overall': all_success
                }
            }
            
            if all_success:
                logger.info("🎉 SUCCESS! All criteria met:")
                logger.info(f"   ✅ R² ≥ {success_criteria['r2_target']}: {test_metrics['r2']:.4f}")
                logger.info(f"   ✅ RMSE ≤ {success_criteria['rmse_target']}: {test_metrics['rmse']:.4f}")
                logger.info(f"   ✅ MAE ≤ {success_criteria['mae_target']}: {test_metrics['mae']:.4f}")
            else:
                logger.info("⚠️  Some criteria not met:")
                logger.info(f"   {'✅' if r2_success else '❌'} R² ≥ {success_criteria['r2_target']}: {test_metrics['r2']:.4f}")
                logger.info(f"   {'✅' if rmse_success else '❌'} RMSE ≤ {success_criteria['rmse_target']}: {test_metrics['rmse']:.4f}")
                logger.info(f"   {'✅' if mae_success else '❌'} MAE ≤ {success_criteria['mae_target']}: {test_metrics['mae']:.4f}")
        
        # Save final model if requested
        if save_final_model:
            final_model_path = self.results_dir / "final_optimized_model.pth"
            torch.save({
                'model_state_dict': final_model.state_dict(),
                'best_params': best_params,
                'test_metrics': final_results.get('test_metrics'),
                'model_config': model_config
            }, final_model_path)
            logger.info(f"Final optimized model saved to {final_model_path}")
        
        # Save final results
        final_results_path = self.results_dir / "final_optimization_results.json"
        with open(final_results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        return final_results 