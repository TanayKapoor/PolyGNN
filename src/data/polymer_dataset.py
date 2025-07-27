import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset

from ..features.polymer_features import PolymerFeatureError, PolymerFeatureExtractor
from .molecular_graph import MolecularGraphConverter

logger = logging.getLogger(__name__)


class PolymerTgDataset(Dataset):
    """
    Dataset for glass transition temperature (Tg) prediction of polymers.
    Handles SMILES to graph conversion, data splits, and quality assessment.
    """

    def __init__(
        self,
        root: str,
        csv_file: Optional[str] = None,
        smiles_column: str = "smiles",
        target_column: str = "tg",
        dp_column: Optional[str] = "dp",
        mw_column: Optional[str] = "mw",
        split_type: str = "train",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        graph_converter_kwargs: Optional[Dict] = None,
        polymer_feature_kwargs: Optional[Dict] = None,
        transform=None,
        pre_transform=None,
        random_state: int = 42,
    ):
        """
        Initialize the polymer Tg dataset.

        Args:
            root: Root directory for dataset
            csv_file: Path to CSV file with SMILES and Tg data
            smiles_column: Column name containing SMILES strings
            target_column: Column name containing Tg values
            dp_column: Column name containing degree of polymerization (optional)
            mw_column: Column name containing molecular weight (optional)
            split_type: 'train', 'val', 'test', or 'all'
            split_ratios: (train, val, test) ratios
            graph_converter_kwargs: Arguments for MolecularGraphConverter
            polymer_feature_kwargs: Arguments for PolymerFeatureExtractor
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform to apply to data
            random_state: Random seed for reproducibility
        """
        self.csv_file = csv_file
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.dp_column = dp_column
        self.mw_column = mw_column
        self.split_type = split_type
        self.split_ratios = split_ratios
        self.random_state = random_state

        # Initialize graph converter
        graph_kwargs = graph_converter_kwargs or {}
        self.graph_converter = MolecularGraphConverter(**graph_kwargs)

        # Initialize polymer feature extractor only if polymer features are requested
        self.use_polymer_features = polymer_feature_kwargs is not None
        if self.use_polymer_features:
            self.polymer_extractor = PolymerFeatureExtractor(**polymer_feature_kwargs)
            logger.info(
                f"Polymer features enabled with dimension {self.polymer_extractor.get_feature_dim()}"
            )
        else:
            self.polymer_extractor = None
            logger.info("Polymer features disabled")

        # Dataset metadata
        self.data_info = {}
        self.quality_metrics = {}

        super().__init__(root, transform, pre_transform)

        # Load and process data
        self._load_and_process_data()

    def _load_and_process_data(self):
        """Load CSV data and create train/val/test splits."""
        if self.csv_file is None:
            logger.error("No CSV file provided")
            return

        # Load CSV data
        df = pd.read_csv(self.csv_file)
        logger.info(f"Loaded {len(df)} samples from {self.csv_file}")

        # Basic data validation
        if self.smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{self.smiles_column}' not found")
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        # Remove rows with missing values
        initial_len = len(df)
        df = df.dropna(subset=[self.smiles_column, self.target_column])
        logger.info(f"Removed {initial_len - len(df)} rows with missing values")

        # Check for polymer feature columns
        has_dp = self.dp_column and self.dp_column in df.columns
        has_mw = self.mw_column and self.mw_column in df.columns

        # Store full dataset info
        self.data_info = {
            "total_samples": len(df),
            "smiles_column": self.smiles_column,
            "target_column": self.target_column,
            "dp_column": self.dp_column if has_dp else None,
            "mw_column": self.mw_column if has_mw else None,
            "has_polymer_features": has_dp or has_mw,
            "split_ratios": self.split_ratios,
            "random_state": self.random_state,
        }

        # Perform data quality assessment
        self._assess_data_quality(df)

        # Create train/val/test splits
        self._create_splits(df)

    def _assess_data_quality(self, df: pd.DataFrame):
        """Perform comprehensive data quality assessment."""
        logger.info("Performing data quality assessment...")

        # Target variable statistics
        tg_values = df[self.target_column].values
        self.quality_metrics = {
            "target_stats": {
                "count": len(tg_values),
                "mean": float(np.mean(tg_values)),
                "std": float(np.std(tg_values)),
                "min": float(np.min(tg_values)),
                "max": float(np.max(tg_values)),
                "q25": float(np.percentile(tg_values, 25)),
                "q50": float(np.percentile(tg_values, 50)),
                "q75": float(np.percentile(tg_values, 75)),
            },
            "smiles_stats": {},
            "molecular_stats": {},
            "outliers": {},
            "duplicates": {},
        }

        # SMILES statistics
        smiles_lengths = df[self.smiles_column].str.len()
        self.quality_metrics["smiles_stats"] = {
            "avg_length": float(smiles_lengths.mean()),
            "min_length": int(smiles_lengths.min()),
            "max_length": int(smiles_lengths.max()),
            "std_length": float(smiles_lengths.std()),
        }

        # Check for duplicate SMILES
        duplicates = df[df.duplicated(subset=[self.smiles_column], keep=False)]
        self.quality_metrics["duplicates"] = {
            "count": len(duplicates),
            "unique_smiles_with_duplicates": len(
                duplicates[self.smiles_column].unique()
            ),
            "examples": (
                duplicates.head(5).to_dict("records") if len(duplicates) > 0 else []
            ),
        }

        # Molecular property analysis
        self._analyze_molecular_properties(df)

        # Polymer feature analysis
        self._analyze_polymer_features(df)

        # Outlier detection
        self._detect_outliers(df)

        # Save quality report
        self._save_quality_report()

    def _analyze_molecular_properties(self, df: pd.DataFrame):
        """Analyze molecular properties of the dataset."""
        logger.info("Analyzing molecular properties...")

        mol_weights = []
        num_atoms = []
        num_bonds = []
        valid_smiles = 0

        for smiles in df[self.smiles_column]:
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors

                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol_weights.append(Descriptors.MolWt(mol))
                    num_atoms.append(mol.GetNumAtoms())
                    num_bonds.append(mol.GetNumBonds())
                    valid_smiles += 1
                else:
                    mol_weights.append(np.nan)
                    num_atoms.append(np.nan)
                    num_bonds.append(np.nan)
            except Exception:
                mol_weights.append(np.nan)
                num_atoms.append(np.nan)
                num_bonds.append(np.nan)

        # Calculate statistics (excluding NaN values)
        self.quality_metrics["molecular_stats"] = {
            "valid_smiles_count": valid_smiles,
            "valid_smiles_percentage": (valid_smiles / len(df)) * 100,
            "molecular_weight": {
                "mean": float(np.nanmean(mol_weights)),
                "std": float(np.nanstd(mol_weights)),
                "min": float(np.nanmin(mol_weights)),
                "max": float(np.nanmax(mol_weights)),
            },
            "num_atoms": {
                "mean": float(np.nanmean(num_atoms)),
                "std": float(np.nanstd(num_atoms)),
                "min": float(np.nanmin(num_atoms)),
                "max": float(np.nanmax(num_atoms)),
            },
            "num_bonds": {
                "mean": float(np.nanmean(num_bonds)),
                "std": float(np.nanstd(num_bonds)),
                "min": float(np.nanmin(num_bonds)),
                "max": float(np.nanmax(num_bonds)),
            },
        }

    def _analyze_polymer_features(self, df: pd.DataFrame):
        """Analyze polymer-specific features like DP and molecular weight."""
        if not self.use_polymer_features:
            logger.info("Polymer features disabled - skipping polymer feature analysis")
            return

        logger.info("Analyzing polymer features...")

        polymer_stats = {}

        # Analyze degree of polymerization if available
        if self.dp_column and self.dp_column in df.columns:
            dp_values = df[self.dp_column].dropna()
            if len(dp_values) > 0:
                polymer_stats["degree_polymerization"] = {
                    "count": len(dp_values),
                    "coverage": (len(dp_values) / len(df)) * 100,
                    "mean": float(np.mean(dp_values)),
                    "std": float(np.std(dp_values)),
                    "min": float(np.min(dp_values)),
                    "max": float(np.max(dp_values)),
                    "median": float(np.median(dp_values)),
                }

        # Analyze molecular weight if available
        if self.mw_column and self.mw_column in df.columns:
            mw_values = df[self.mw_column].dropna()
            if len(mw_values) > 0:
                polymer_stats["total_molecular_weight"] = {
                    "count": len(mw_values),
                    "coverage": (len(mw_values) / len(df)) * 100,
                    "mean": float(np.mean(mw_values)),
                    "std": float(np.std(mw_values)),
                    "min": float(np.min(mw_values)),
                    "max": float(np.max(mw_values)),
                    "median": float(np.median(mw_values)),
                }

        # Analyze polymer feature extraction success rate only if extractor exists
        if self.polymer_extractor is not None:
            successful_extractions = 0
            total_attempts = min(100, len(df))  # Sample to avoid performance issues

            for i in range(total_attempts):
                try:
                    smiles = df.iloc[i][self.smiles_column]
                    dp = df.iloc[i].get(self.dp_column) if self.dp_column else None
                    mw = df.iloc[i].get(self.mw_column) if self.mw_column else None

                    # Try to extract polymer features
                    self.polymer_extractor.extract_features(smiles, dp=dp, mw=mw)
                    successful_extractions += 1
                except (PolymerFeatureError, Exception):
                    pass

            polymer_stats["feature_extraction"] = {
                "success_rate": (successful_extractions / total_attempts) * 100,
                "samples_tested": total_attempts,
                "feature_dim": self.polymer_extractor.get_feature_dim(),
            }

        self.quality_metrics["polymer_stats"] = polymer_stats

    def _detect_outliers(self, df: pd.DataFrame):
        """Detect outliers in Tg values using IQR method."""
        tg_values = df[self.target_column].values

        q1 = np.percentile(tg_values, 25)
        q3 = np.percentile(tg_values, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[
            (df[self.target_column] < lower_bound)
            | (df[self.target_column] > upper_bound)
        ]

        self.quality_metrics["outliers"] = {
            "count": len(outliers),
            "percentage": (len(outliers) / len(df)) * 100,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "examples": (
                outliers.head(10).to_dict("records") if len(outliers) > 0 else []
            ),
        }

    def _create_splits(self, df: pd.DataFrame):
        """Create train/validation/test splits."""
        logger.info("Creating train/validation/test splits...")

        # Split data
        train_ratio, val_ratio, test_ratio = self.split_ratios

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=self.random_state,
            stratify=None,  # Can add stratification based on Tg ranges if needed
        )

        # Second split: separate train and validation
        val_size = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=self.random_state
        )

        # Store splits
        self.splits = {"train": train_df, "val": val_df, "test": test_df, "all": df}

        # Update data info
        self.data_info["split_sizes"] = {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(df),
        }

        logger.info(
            f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

    def _save_quality_report(self):
        """Save data quality report to file."""
        report_path = Path(self.root) / "data_quality_report.json"

        report = {
            "dataset_info": self.data_info,
            "quality_metrics": self.quality_metrics,
            "generated_at": datetime.now().isoformat(),
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Data quality report saved to {report_path}")

    def create_visualization_plots(self, save_path: Optional[str] = None):
        """Create visualization plots for data quality assessment."""
        if self.split_type != "all":
            logger.warning(
                "Visualization plots should be created with split_type='all'"
            )
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Get full dataset
        df = self.splits["all"]

        # Plot 1: Tg distribution
        axes[0, 0].hist(df[self.target_column], bins=30, alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Glass Transition Temperature Distribution")
        axes[0, 0].set_xlabel("Tg (°C)")
        axes[0, 0].set_ylabel("Frequency")

        # Plot 2: SMILES length distribution
        smiles_lengths = df[self.smiles_column].str.len()
        axes[0, 1].hist(smiles_lengths, bins=30, alpha=0.7, edgecolor="black")
        axes[0, 1].set_title("SMILES Length Distribution")
        axes[0, 1].set_xlabel("SMILES Length")
        axes[0, 1].set_ylabel("Frequency")

        # Plot 3: Boxplot for outlier detection
        axes[1, 0].boxplot(df[self.target_column])
        axes[1, 0].set_title("Tg Outlier Detection")
        axes[1, 0].set_ylabel("Tg (°C)")

        # Plot 4: Scatter plot of molecular weight vs Tg (if available)
        if "molecular_weight" in df.columns:
            axes[1, 1].scatter(
                df["molecular_weight"], df[self.target_column], alpha=0.6
            )
            axes[1, 1].set_title("Molecular Weight vs Tg")
            axes[1, 1].set_xlabel("Molecular Weight")
            axes[1, 1].set_ylabel("Tg (°C)")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Molecular Weight\ndata not available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Visualization plots saved to {save_path}")

        plt.show()

    def get_split_data(self, split_type: str) -> pd.DataFrame:
        """Get data for a specific split."""
        if split_type not in self.splits:
            raise ValueError(f"Split type '{split_type}' not found")
        return self.splits[split_type]

    def __len__(self) -> int:
        """Return the number of samples in the current split."""
        return len(self.splits[self.split_type])

    def __getitem__(self, idx: int) -> Data:
        """Get a single sample from the dataset."""
        df = self.splits[self.split_type]
        row = df.iloc[idx]

        # Convert SMILES to graph
        smiles = row[self.smiles_column]
        graph = self.graph_converter.smiles_to_graph(smiles)

        if graph is None:
            # Return dummy graph if conversion fails
            logger.warning(f"Failed to convert SMILES: {smiles} - creating dummy graph")
            graph = Data(
                x=torch.zeros(1, self.graph_converter.atom_feature_dim),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                num_nodes=1,
                mol_features=torch.zeros(
                    13, dtype=torch.float
                ),  # Match molecular feature dim
                smiles=smiles,
            )

        # Extract polymer features only if enabled
        if self.use_polymer_features and self.polymer_extractor is not None:
            try:
                dp = row.get(self.dp_column) if self.dp_column else None
                mw = row.get(self.mw_column) if self.mw_column else None
                polymer_features = self.polymer_extractor.extract_features(
                    smiles, dp=dp, mw=mw
                )
                graph.polymer_features = polymer_features
            except PolymerFeatureError as e:
                logger.warning(f"Failed to extract polymer features for {smiles}: {e}")
                # Use zero features as fallback
                graph.polymer_features = torch.zeros(
                    self.polymer_extractor.get_feature_dim(), dtype=torch.float32
                )

        # Add target value
        graph.y = torch.tensor([row[self.target_column]], dtype=torch.float)

        # Add metadata
        graph.idx = idx
        graph.smiles = smiles

        return graph

    def get_feature_statistics(self) -> Dict:
        """Get statistics for feature normalization."""
        if self.split_type != "train":
            logger.warning("Feature statistics should be computed on training data")

        # Compute statistics from training data
        train_df = self.splits["train"]
        targets = train_df[self.target_column].values

        return {
            "target_mean": float(np.mean(targets)),
            "target_std": float(np.std(targets)),
            "target_min": float(np.min(targets)),
            "target_max": float(np.max(targets)),
        }

    def print_quality_summary(self):
        """Print a summary of data quality metrics."""
        print("\n" + "=" * 60)
        print("POLYMER Tg DATASET QUALITY SUMMARY")
        print("=" * 60)

        # Dataset info
        print(f"Total samples: {self.data_info['total_samples']}")
        print(f"Train/Val/Test split: {self.split_ratios}")
        if "split_sizes" in self.data_info:
            sizes = self.data_info["split_sizes"]
            print(
                f"Split sizes: Train={sizes['train']}, Val={sizes['val']}, Test={sizes['test']}"
            )

        # Target statistics
        tg_stats = self.quality_metrics["target_stats"]
        print(f"\nGlass Transition Temperature (Tg) Statistics:")
        print(f"  Mean: {tg_stats['mean']:.2f}°C")
        print(f"  Std:  {tg_stats['std']:.2f}°C")
        print(f"  Range: {tg_stats['min']:.2f}°C to {tg_stats['max']:.2f}°C")

        # SMILES statistics
        smiles_stats = self.quality_metrics["smiles_stats"]
        print(f"\nSMILES Statistics:")
        print(f"  Average length: {smiles_stats['avg_length']:.1f}")
        print(
            f"  Length range: {smiles_stats['min_length']} to {smiles_stats['max_length']}"
        )

        # Molecular statistics
        mol_stats = self.quality_metrics["molecular_stats"]
        print(f"\nMolecular Statistics:")
        print(
            f"  Valid SMILES: {mol_stats['valid_smiles_count']} ({mol_stats['valid_smiles_percentage']:.1f}%)"
        )
        print(
            f"  Average molecular weight: {mol_stats['molecular_weight']['mean']:.1f}"
        )
        print(f"  Average number of atoms: {mol_stats['num_atoms']['mean']:.1f}")

        # Outliers
        outliers = self.quality_metrics["outliers"]
        print(f"\nOutliers (IQR method):")
        print(f"  Count: {outliers['count']} ({outliers['percentage']:.1f}%)")
        print(
            f"  Range: {outliers['lower_bound']:.2f}°C to {outliers['upper_bound']:.2f}°C"
        )

        # Duplicates
        duplicates = self.quality_metrics["duplicates"]
        print(f"\nDuplicate SMILES:")
        print(f"  Count: {duplicates['count']}")

        # Polymer features
        if "polymer_stats" in self.quality_metrics:
            polymer_stats = self.quality_metrics["polymer_stats"]
            print(f"\nPolymer Features:")

            if "degree_polymerization" in polymer_stats:
                dp_stats = polymer_stats["degree_polymerization"]
                print(f"  Degree of Polymerization:")
                print(
                    f"    Coverage: {dp_stats['coverage']:.1f}% ({dp_stats['count']} samples)"
                )
                print(f"    Range: {dp_stats['min']:.0f} to {dp_stats['max']:.0f}")
                print(f"    Mean: {dp_stats['mean']:.0f}")

            if "total_molecular_weight" in polymer_stats:
                mw_stats = polymer_stats["total_molecular_weight"]
                print(f"  Total Molecular Weight:")
                print(
                    f"    Coverage: {mw_stats['coverage']:.1f}% ({mw_stats['count']} samples)"
                )
                print(
                    f"    Range: {mw_stats['min']:.0f} to {mw_stats['max']:.0f} g/mol"
                )
                print(f"    Mean: {mw_stats['mean']:.0f} g/mol")

            if "feature_extraction" in polymer_stats:
                feat_stats = polymer_stats["feature_extraction"]
                print(f"  Feature Extraction:")
                print(f"    Success Rate: {feat_stats['success_rate']:.1f}%")
                print(f"    Feature Dimension: {feat_stats['feature_dim']}")

        print("=" * 60)
