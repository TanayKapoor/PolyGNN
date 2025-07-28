"""
Polymer Fingerprint Baseline Model

Combines molecular fingerprints with polymer-specific chain features
for glass transition temperature (Tg) prediction.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors

logger = logging.getLogger(__name__)


class PolymerFingerprintBaseline(nn.Module):
    """
    Baseline model for polymer Tg prediction using molecular fingerprints
    and polymer-specific chain features.
    """

    def __init__(
        self,
        fingerprint_dim: int = 2048,
        chain_features_dim: int = 10,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.2,
        activation: str = "relu",
    ):
        """
        Initialize the polymer fingerprint baseline model.

        Args:
            fingerprint_dim: Dimension of molecular fingerprint (default: 2048 for Morgan)
            chain_features_dim: Number of polymer chain features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()

        self.fingerprint_dim = fingerprint_dim
        self.chain_features_dim = chain_features_dim
        self.dropout_rate = dropout_rate

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Input projection layers
        input_dim = fingerprint_dim + chain_features_dim
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        self.input_dropout = nn.Dropout(dropout_rate)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.hidden_layers.append(nn.Dropout(dropout_rate))

        # Output layer for Tg prediction
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, fingerprints: torch.Tensor, chain_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            fingerprints: Molecular fingerprints [batch_size, fingerprint_dim]
            chain_features: Polymer chain features [batch_size, chain_features_dim]

        Returns:
            Predicted Tg values [batch_size, 1]
        """
        # Concatenate fingerprints and chain features
        x = torch.cat([fingerprints, chain_features], dim=1)

        # Input projection
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.input_dropout(x)

        # Hidden layers
        for i in range(0, len(self.hidden_layers), 2):
            linear = self.hidden_layers[i]
            dropout = self.hidden_layers[i + 1]
            x = linear(x)
            x = self.activation(x)
            x = dropout(x)

        # Output layer
        tg_pred = self.output_layer(x)

        return tg_pred


class PolymerFeatureExtractor:
    """
    Extract molecular fingerprints and polymer-specific chain features
    from SMILES strings.
    """

    def __init__(
        self,
        fingerprint_type: str = "morgan",
        fingerprint_radius: int = 2,
        fingerprint_dim: int = 2048,
    ):
        """
        Initialize the feature extractor.

        Args:
            fingerprint_type: Type of fingerprint ('morgan', 'maccs', 'topological')
            fingerprint_radius: Radius for Morgan fingerprints
            fingerprint_dim: Dimension of fingerprint
        """
        self.fingerprint_type = fingerprint_type
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_dim = fingerprint_dim

    def extract_fingerprint(self, smiles: str) -> np.ndarray:
        """
        Extract molecular fingerprint from SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Molecular fingerprint as numpy array
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.fingerprint_dim)

            if self.fingerprint_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fingerprint_radius, nBits=self.fingerprint_dim
                )
                return np.array(fp)
            elif self.fingerprint_type == "maccs":
                from rdkit.Chem import MACCSkeys

                fp = MACCSkeys.GenMACCSKeys(mol)
                # Pad or truncate to desired dimension
                fp_array = np.array(fp)
                if len(fp_array) < self.fingerprint_dim:
                    fp_array = np.pad(
                        fp_array, (0, self.fingerprint_dim - len(fp_array))
                    )
                else:
                    fp_array = fp_array[: self.fingerprint_dim]
                return fp_array
            else:
                # Default to Morgan if unknown type
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fingerprint_radius, nBits=self.fingerprint_dim
                )
                return np.array(fp)

        except Exception as e:
            logger.warning(f"Failed to extract fingerprint from SMILES {smiles}: {e}")
            return np.zeros(self.fingerprint_dim)

    def extract_chain_features(self, smiles: str) -> np.ndarray:
        """
        Extract polymer-specific chain features from SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Chain features as numpy array [10 features]
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(10)

            # Calculate molecular descriptors
            features = []

            # Basic molecular properties
            features.append(mol.GetNumAtoms())  # Number of atoms
            features.append(mol.GetNumBonds())  # Number of bonds
            features.append(Descriptors.MolWt(mol))  # Molecular weight
            features.append(Descriptors.MolLogP(mol))  # LogP (hydrophobicity)

            # Structural features
            features.append(Descriptors.NumRotatableBonds(mol))  # Flexibility
            features.append(Descriptors.RingCount(mol))  # Ring count
            features.append(Descriptors.NumAromaticRings(mol))  # Aromatic content
            # Handle different RDKit versions for FractionCsp3
            try:
                features.append(Descriptors.FractionCsp3(mol))  # sp3 carbon fraction
            except AttributeError:
                # Newer RDKit versions might have different naming
                try:
                    features.append(Descriptors.fr_C_sp3(mol))
                except AttributeError:
                    # Fallback if descriptor not available
                    features.append(0.0)

            # Hydrogen bonding
            features.append(Descriptors.NumHDonors(mol))  # H-bond donors
            features.append(Descriptors.NumHAcceptors(mol))  # H-bond acceptors

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.warning(
                f"Failed to extract chain features from SMILES {smiles}: {e}"
            )
            return np.zeros(10)

    def extract_features(self, smiles: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract both fingerprint and chain features from SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Tuple of (fingerprint, chain_features)
        """
        fingerprint = self.extract_fingerprint(smiles)
        chain_features = self.extract_chain_features(smiles)
        return fingerprint, chain_features


class PolymerFingerprintDataset(torch.utils.data.Dataset):
    """
    Dataset for polymer fingerprint baseline training.
    """

    def __init__(
        self,
        smiles_list: List[str],
        targets: List[float],
        feature_extractor: PolymerFeatureExtractor,
    ):
        """
        Initialize the dataset.

        Args:
            smiles_list: List of SMILES strings
            targets: List of target values (e.g., Tg)
            feature_extractor: Feature extractor instance
        """
        self.smiles_list = smiles_list
        self.targets = targets
        self.feature_extractor = feature_extractor

        # Pre-compute features for efficiency
        self.fingerprints = []
        self.chain_features = []

        logger.info(f"Extracting features for {len(smiles_list)} samples...")
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(smiles_list)} samples")

            fp, cf = self.feature_extractor.extract_features(smiles)
            self.fingerprints.append(fp)
            self.chain_features.append(cf)

        self.fingerprints = np.array(self.fingerprints)
        self.chain_features = np.array(self.chain_features)

        logger.info(
            f"Feature extraction complete. Fingerprints: {self.fingerprints.shape}, "
            f"Chain features: {self.chain_features.shape}"
        )

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return {
            "fingerprint": torch.tensor(self.fingerprints[idx], dtype=torch.float32),
            "chain_features": torch.tensor(
                self.chain_features[idx], dtype=torch.float32
            ),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "smiles": self.smiles_list[idx],
        }


def create_baseline_model_and_extractor(
    config: Dict,
) -> Tuple[PolymerFingerprintBaseline, PolymerFeatureExtractor]:
    """
    Create baseline model and feature extractor from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (model, feature_extractor)
    """
    # Create feature extractor
    feature_extractor = PolymerFeatureExtractor(
        fingerprint_type=config.get("fingerprint_type", "morgan"),
        fingerprint_radius=config.get("fingerprint_radius", 2),
        fingerprint_dim=config.get("fingerprint_dim", 2048),
    )

    # Create model
    model = PolymerFingerprintBaseline(
        fingerprint_dim=config.get("fingerprint_dim", 2048),
        chain_features_dim=config.get("chain_features_dim", 10),
        hidden_dims=config.get("hidden_dims", [512, 256, 128]),
        dropout_rate=config.get("dropout_rate", 0.2),
        activation=config.get("activation", "relu"),
    )

    return model, feature_extractor
