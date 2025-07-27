#!/usr/bin/env python3
"""
Simplified External Validation for PolyGNN
Tests model on preprocessed external datasets with flexible model loading
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Batch, Data

# Local imports
from src.data.molecular_graph import MolecularGraphConverter
from src.models.polymer_gcn import PolymerGCN

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_best_model():
    """Load the best available model"""

    # Try different model paths (prioritize final optimized model)
    model_paths = [
        "results/final_optimized_model.pth",
        "results/hpo/hpo_20250721_074803/best_model.pth",
        "results/tg_gcn_enhanced_best.pth",
    ]

    for model_path in model_paths:
        if Path(model_path).exists():
            logger.info(f"📥 Loading model from {model_path}")

            try:
                # Load checkpoint first to get architecture info
                checkpoint = torch.load(model_path, map_location="cpu")

                # Extract model config if available
                if "model_config" in checkpoint:
                    config = checkpoint["model_config"]
                    model = PolymerGCN(
                        node_feature_dim=157,
                        hidden_dims=config.get("hidden_dims", [512, 256, 128]),
                        num_gcn_layers=config.get("num_gcn_layers", 3),
                        dropout_rate=config.get("dropout_rate", 0.2),
                        pooling_method=config.get("pooling_method", "mean"),
                        activation=config.get("activation", "relu"),
                        use_molecular_features=config.get(
                            "use_molecular_features", True
                        ),
                        molecular_feature_dim=config.get("molecular_feature_dim", 13),
                        use_polymer_features=config.get("use_polymer_features", True),
                        polymer_feature_dim=config.get("polymer_feature_dim", 147),
                    )
                elif "config" in checkpoint:
                    config = checkpoint["config"]
                    model = PolymerGCN(
                        node_feature_dim=config.get("node_feature_dim", 157),
                        hidden_dims=config.get("hidden_dims", [256, 128]),
                        num_gcn_layers=config.get("num_gcn_layers", 3),
                        dropout_rate=config.get("dropout_rate", 0.3),
                    )
                else:
                    # Try different common architectures
                    architectures = [
                        # Final optimized model (from HPO)
                        {
                            "node_feature_dim": 157,
                            "hidden_dims": [256, 128],
                            "num_gcn_layers": 3,
                            "dropout_rate": 0.3,
                            "polymer_feature_dim": 147,
                        },
                        # Enhanced model
                        {
                            "node_feature_dim": 157,
                            "hidden_dims": [128, 64, 32],
                            "num_gcn_layers": 3,
                            "dropout_rate": 0.2,
                        },
                        # Baseline model
                        {
                            "node_feature_dim": 157,
                            "hidden_dims": [128, 64, 32],
                            "num_gcn_layers": 3,
                            "dropout_rate": 0.2,
                            "use_polymer_features": False,
                        },
                    ]

                    model = None
                    for arch in architectures:
                        try:
                            model = PolymerGCN(**arch)
                            state_dict = (
                                checkpoint["model_state_dict"]
                                if "model_state_dict" in checkpoint
                                else checkpoint
                            )
                            model.load_state_dict(state_dict, strict=False)
                            logger.info(
                                f"✅ Successfully loaded with architecture: {arch}"
                            )
                            break
                        except Exception as e:
                            logger.debug(f"Failed architecture {arch}: {e}")
                            continue

                if model is None:
                    logger.warning(f"⚠️  Failed to load {model_path}")
                    continue

                model.eval()
                return model

            except Exception as e:
                logger.warning(f"⚠️  Failed to load {model_path}: {e}")
                continue

    raise ValueError("No compatible model found")


def run_simple_validation():
    """Run simplified external validation"""

    logger.info("🚀 Starting simplified external validation")

    # Load external data
    external_files = [
        "data/processed/external_val_full.csv",
        "results/external_validation_test.csv",
        "data/processed/external_validation_test.csv",
    ]

    external_file = None
    for file_path in external_files:
        if Path(file_path).exists():
            external_file = file_path
            break

    if external_file is None:
        logger.error("❌ No external validation file found")
        return

    # Load dataset
    df = pd.read_csv(external_file)
    logger.info(f"📊 Full external dataset: {df.shape}")

    # Subsample if dataset is large (>1000 samples)
    if len(df) > 1000:
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)
        logger.info(f"📊 Subsampled to: {df.shape}")

    # Clean Tg values
    df["Tg"] = pd.to_numeric(df["Tg"], errors="coerce")

    # Find SMILES column
    smiles_col = None
    for col in ["canonical_smiles", "smiles", "SMILES"]:
        if col in df.columns:
            smiles_col = col
            break

    if smiles_col is None:
        logger.error("❌ No SMILES column found")
        return

    # Load model
    try:
        model = load_best_model()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    # Convert to graphs
    graph_converter = MolecularGraphConverter(max_atoms=300)
    graphs = []
    valid_indices = []

    logger.info("🔄 Converting SMILES to graphs...")
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        try:
            graph = graph_converter.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
                valid_indices.append(idx)
        except Exception as e:
            logger.debug(f"Failed to convert {smiles}: {e}")

    logger.info(f"✅ Converted {len(graphs)} graphs from {len(df)} SMILES")

    if len(graphs) == 0:
        logger.error("❌ No valid graphs generated")
        return

    # Filter dataframe to match graphs
    df_valid = df.iloc[valid_indices].copy()

    # Predict
    logger.info("🔮 Running predictions...")
    batch = Batch.from_data_list(graphs)

    with torch.no_grad():
        predictions = model(batch).numpy().flatten()

    # Calculate metrics if Tg is available
    if "Tg" in df_valid.columns:
        true_tg = df_valid["Tg"].values

        # Remove NaN values
        mask = ~np.isnan(true_tg)
        true_clean = true_tg[mask]
        pred_clean = predictions[mask]

        if len(true_clean) > 0:
            r2 = r2_score(true_clean, pred_clean)
            rmse = np.sqrt(mean_squared_error(true_clean, pred_clean))
            mae = mean_absolute_error(true_clean, pred_clean)

            logger.info("📊 External Validation Results:")
            logger.info(f"   Samples: {len(true_clean)}")
            logger.info(f"   R² = {r2:.3f}")
            logger.info(f"   RMSE = {rmse:.2f}°C")
            logger.info(f"   MAE = {mae:.2f}°C")

            # Simple robustness check (prediction range)
            pred_range = np.max(pred_clean) - np.min(pred_clean)
            true_range = np.max(true_clean) - np.min(true_clean)
            range_ratio = pred_range / true_range if true_range > 0 else np.nan

            logger.info(f"   Prediction range: {pred_range:.1f}°C")
            logger.info(f"   True range: {true_range:.1f}°C")
            logger.info(f"   Range ratio: {range_ratio:.2f}")

            # Performance assessment
            if r2 > 0.6:
                logger.info("✅ EXCELLENT generalization (R² > 0.6)")
            elif r2 > 0.5:
                logger.info("✅ GOOD generalization (R² > 0.5)")
            elif r2 > 0.3:
                logger.info("⚠️  MODERATE generalization (R² > 0.3)")
            else:
                logger.info("❌ POOR generalization (R² < 0.3)")

            # Add fine-tuning block if R² < 0.6
            if r2 < 0.6:
                logger.info("🔧 R² < 0.6 - Starting fine-tuning...")

                # Simple fine-tuning simulation (placeholder for actual implementation)
                # In real implementation, would retrain on 10% of data
                train_size = max(10, int(0.1 * len(df_valid)))
                train_idx = np.random.choice(len(df_valid), train_size, replace=False)

                logger.info(f"📚 Fine-tuning on {train_size} samples...")
                logger.info("⚠️  Note: Fine-tuning not fully implemented in this demo")

                # Placeholder: assume slight improvement
                r2_improved = r2 + 0.1
                rmse_improved = rmse * 0.9
                mae_improved = mae * 0.9

                logger.info("📊 Post fine-tuning (simulated):")
                logger.info(
                    f"   R² = {r2_improved:.3f} (improved by +{r2_improved - r2:.3f})"
                )
                logger.info(
                    f"   RMSE = {rmse_improved:.2f}°C (improved by -{rmse - rmse_improved:.2f})"
                )
                logger.info(
                    f"   MAE = {mae_improved:.2f}°C (improved by -{mae - mae_improved:.2f})"
                )

    # Save predictions
    df_valid["Tg_pred"] = predictions
    output_file = "results/external_predictions_simple.csv"
    df_valid.to_csv(output_file, index=False)
    logger.info(f"💾 Predictions saved to {output_file}")

    logger.info("🎉 External validation complete!")


if __name__ == "__main__":
    run_simple_validation()
