import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class MolecularGraphConverter:
    """
    Converts SMILES strings to molecular graphs for GNN processing.
    Handles both small molecules and polymer structures.
    """

    def __init__(
        self,
        max_atoms: int = 200,
        include_hydrogens: bool = False,
        use_chirality: bool = True,
        use_bond_types: bool = True,
    ):
        """
        Initialize the molecular graph converter.

        Args:
            max_atoms: Maximum number of atoms to include in graph
            include_hydrogens: Whether to include hydrogen atoms
            use_chirality: Whether to include chirality information
            use_bond_types: Whether to include bond type information
        """
        self.max_atoms = max_atoms
        self.include_hydrogens = include_hydrogens
        self.use_chirality = use_chirality
        self.use_bond_types = use_bond_types

        # Atom feature dimensions
        self.atom_feature_dims = {
            "atomic_num": 118,  # Elements 1-118
            "degree": 11,  # 0-10 connections
            "formal_charge": 11,  # -5 to +5
            "hybridization": 6,  # sp, sp2, sp3, sp3d, sp3d2, other
            "num_h": 5,  # 0-4 hydrogens
            "aromatic": 2,  # aromatic or not
            "chirality": 4 if use_chirality else 1,  # R, S, unspecified
        }

        # Bond feature dimensions
        self.bond_feature_dims = {
            "bond_type": 4,  # single, double, triple, aromatic
            "conjugated": 2,  # conjugated or not
            "in_ring": 2,  # in ring or not
            "stereo": 6,  # stereo configuration
        }

        # Calculate total feature dimensions
        self.atom_feature_dim = sum(self.atom_feature_dims.values())
        self.bond_feature_dim = (
            sum(self.bond_feature_dims.values()) if use_bond_types else 0
        )

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert SMILES string to PyTorch Geometric Data object.

        Args:
            smiles: SMILES string

        Returns:
            PyTorch Geometric Data object or None if conversion fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                return None

            # Add hydrogens if requested
            if self.include_hydrogens:
                mol = Chem.AddHs(mol)

            # Check atom limit
            if mol.GetNumAtoms() > self.max_atoms:
                logger.warning(
                    f"Molecule too large: {mol.GetNumAtoms()} atoms (max: {self.max_atoms})"
                )
                return None

            # Extract atom features
            atom_features = self._get_atom_features(mol)

            # Extract bond features and connectivity
            edge_index, edge_attr = self._get_bond_features(mol)

            # Create molecular descriptors
            mol_features = self._get_molecular_features(mol)

            # Create PyTorch Geometric Data object
            data_dict = {
                "x": torch.tensor(atom_features, dtype=torch.float),
                "edge_index": torch.tensor(edge_index, dtype=torch.long),
                "num_nodes": mol.GetNumAtoms(),
                "smiles": smiles,
                "mol_features": torch.tensor(mol_features, dtype=torch.float),
            }

            # Only add edge_attr if we have bond features
            if edge_attr is not None:
                data_dict["edge_attr"] = torch.tensor(edge_attr, dtype=torch.float)

            data = Data(**data_dict)

            return data

        except Exception as e:
            logger.error(f"Error converting SMILES {smiles}: {str(e)}")
            return None

    def _get_atom_features(self, mol: Chem.Mol) -> List[List[float]]:
        """Extract atom features from molecule."""
        atom_features = []

        for atom in mol.GetAtoms():
            features = []

            # Atomic number (one-hot encoded)
            atomic_num = atom.GetAtomicNum()
            features.extend(
                self._one_hot_encode(atomic_num, self.atom_feature_dims["atomic_num"])
            )

            # Degree (number of connections)
            degree = atom.GetDegree()
            features.extend(
                self._one_hot_encode(degree, self.atom_feature_dims["degree"])
            )

            # Formal charge
            formal_charge = atom.GetFormalCharge() + 5  # Shift to 0-10 range
            features.extend(
                self._one_hot_encode(
                    formal_charge, self.atom_feature_dims["formal_charge"]
                )
            )

            # Hybridization
            hybridization = atom.GetHybridization()
            hyb_dict = {
                Chem.HybridizationType.SP: 0,
                Chem.HybridizationType.SP2: 1,
                Chem.HybridizationType.SP3: 2,
                Chem.HybridizationType.SP3D: 3,
                Chem.HybridizationType.SP3D2: 4,
            }
            hyb_idx = hyb_dict.get(hybridization, 5)  # 5 for other
            features.extend(
                self._one_hot_encode(hyb_idx, self.atom_feature_dims["hybridization"])
            )

            # Number of hydrogens
            num_h = atom.GetTotalNumHs()
            features.extend(
                self._one_hot_encode(min(num_h, 4), self.atom_feature_dims["num_h"])
            )

            # Aromaticity
            is_aromatic = 1 if atom.GetIsAromatic() else 0
            features.extend(
                self._one_hot_encode(is_aromatic, self.atom_feature_dims["aromatic"])
            )

            # Chirality (if requested)
            if self.use_chirality:
                chirality = atom.GetChiralTag()
                chiral_dict = {
                    Chem.ChiralType.CHI_UNSPECIFIED: 0,
                    Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
                    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
                }
                chiral_idx = chiral_dict.get(chirality, 3)  # 3 for other
                features.extend(
                    self._one_hot_encode(
                        chiral_idx, self.atom_feature_dims["chirality"]
                    )
                )
            else:
                features.extend([1])  # Default value

            atom_features.append(features)

        return atom_features

    def _get_bond_features(
        self, mol: Chem.Mol
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """Extract bond features and connectivity from molecule."""
        edge_index = []
        edge_attr = [] if self.use_bond_types else None

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Add both directions for undirected graph
            edge_index.extend([[i, j], [j, i]])

            if self.use_bond_types:
                bond_features = self._get_single_bond_features(bond)
                edge_attr.extend(
                    [bond_features, bond_features]
                )  # Same features for both directions

        # Transpose edge_index for PyTorch Geometric format
        edge_index = (
            np.array(edge_index).T if edge_index else np.array([[], []], dtype=int)
        )

        return edge_index.tolist(), edge_attr

    def _get_single_bond_features(self, bond: Chem.Bond) -> List[float]:
        """Extract features for a single bond."""
        features = []

        # Bond type
        bond_type = bond.GetBondType()
        bond_dict = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3,
        }
        bond_idx = bond_dict.get(bond_type, 0)
        features.extend(
            self._one_hot_encode(bond_idx, self.bond_feature_dims["bond_type"])
        )

        # Conjugation
        is_conjugated = 1 if bond.GetIsConjugated() else 0
        features.extend(
            self._one_hot_encode(is_conjugated, self.bond_feature_dims["conjugated"])
        )

        # Ring membership
        is_in_ring = 1 if bond.IsInRing() else 0
        features.extend(
            self._one_hot_encode(is_in_ring, self.bond_feature_dims["in_ring"])
        )

        # Stereo configuration
        stereo = bond.GetStereo()
        stereo_dict = {
            Chem.BondStereo.STEREONONE: 0,
            Chem.BondStereo.STEREOANY: 1,
            Chem.BondStereo.STEREOZ: 2,
            Chem.BondStereo.STEREOE: 3,
            Chem.BondStereo.STEREOCIS: 4,
            Chem.BondStereo.STEREOTRANS: 5,
        }
        stereo_idx = stereo_dict.get(stereo, 0)
        features.extend(
            self._one_hot_encode(stereo_idx, self.bond_feature_dims["stereo"])
        )

        return features

    def _get_molecular_features(self, mol: Chem.Mol) -> List[float]:
        """Extract molecular-level features."""
        features = []

        # Basic molecular properties
        features.append(Descriptors.MolWt(mol))  # Molecular weight
        features.append(Descriptors.MolLogP(mol))  # LogP
        features.append(Descriptors.TPSA(mol))  # Topological polar surface area
        features.append(Descriptors.NumHDonors(mol))  # Hydrogen bond donors
        features.append(Descriptors.NumHAcceptors(mol))  # Hydrogen bond acceptors
        features.append(Descriptors.NumRotatableBonds(mol))  # Rotatable bonds
        features.append(Descriptors.RingCount(mol))  # Number of rings
        features.append(Descriptors.NumAromaticRings(mol))  # Aromatic rings
        try:
            features.append(Descriptors.FractionCsp3(mol))  # Fraction of sp3 carbons
        except AttributeError:
            # Fallback for older RDKit versions
            features.append(0.0)
        features.append(Descriptors.NumHeteroatoms(mol))  # Heteroatoms

        # Polymer-relevant features
        features.append(mol.GetNumAtoms())  # Number of atoms
        features.append(mol.GetNumBonds())  # Number of bonds
        features.append(len(Chem.GetSymmSSSR(mol)))  # Smallest set of smallest rings

        return features

    def _one_hot_encode(self, value: int, num_classes: int) -> List[float]:
        """One-hot encode a value."""
        encoding = [0.0] * num_classes
        if 0 <= value < num_classes:
            encoding[value] = 1.0
        return encoding

    def batch_convert(self, smiles_list: List[str]) -> List[Optional[Data]]:
        """Convert multiple SMILES strings to graphs."""
        return [self.smiles_to_graph(smiles) for smiles in smiles_list]

    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for model initialization."""
        return {
            "atom_feature_dim": self.atom_feature_dim,
            "bond_feature_dim": self.bond_feature_dim,
            "mol_feature_dim": 13,  # Number of molecular features
        }
