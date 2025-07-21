"""
Polymer-specific feature extraction for Graph Neural Networks

This module provides comprehensive polymer feature extraction including:
1. Molecular weight of repeating units (handles SMILES with '*' dummies)
2. Degree of polymerization encoding (numerical/log-scaled)
3. Repetition unit structural features (Morgan fingerprint)
4. Chain length descriptors (flexibility, persistence estimates)
5. Repetition unit complexity measures (structural complexity)
6. Polymer-specific molecular descriptors (Tg predictors, crystallinity)

Author: Polymer GNN Team
"""

import rdkit
# Suppress RDKit warnings at import time
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
rkl.logger().setLevel(rkl.ERROR)  # Only show errors, not warnings

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, Crippen
import numpy as np
import torch
import logging
from typing import Optional, Union, Dict, Any, List
import warnings
import pandas as pd
import os
import sys
from contextlib import contextmanager
import math

logger = logging.getLogger(__name__)

# Suppress RDKit warnings globally for this module
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")

@contextmanager
def suppress_rdkit_warnings():
    """Suppress both Python warnings and stderr output from RDKit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Also suppress stderr temporarily to catch C++ warnings
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            yield
        finally:
            sys.stderr.close()
            sys.stderr = stderr

class PolymerFeatureError(Exception):
    """Custom exception for polymer feature extraction errors."""
    pass


def calculate_molecular_weight(smiles: str) -> float:
    """
    Calculate the molecular weight of the polymer repeating unit.
    Handles SMILES with '*' dummy atoms by removing them for accurate calculations.
    
    Args:
        smiles: SMILES string, potentially with '*' connection points
        
    Returns:
        Molecular weight of the repeating unit in g/mol
        
    Raises:
        PolymerFeatureError: If SMILES is invalid or processing fails
        
    Examples:
        >>> calculate_molecular_weight('*CC*')  # Polyethylene unit
        28.031
        >>> calculate_molecular_weight('CCO')   # Ethanol
        46.069
    """
    if not smiles or pd.isna(smiles):
        raise PolymerFeatureError("SMILES string is empty or NaN")
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise PolymerFeatureError(f"Invalid SMILES: {smiles}")
        
        # Ensure molecule is properly processed
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        
        # Remove dummy atoms (*)
        editable_mol = Chem.EditableMol(mol)
        dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
        
        # Remove in reverse order to maintain indices
        for idx in sorted(dummy_indices, reverse=True):
            editable_mol.RemoveAtom(idx)
        
        cleaned_mol = editable_mol.GetMol()
        if cleaned_mol is None:
            raise PolymerFeatureError(f"Failed to process cleaned molecule from: {smiles}")
        
        # Update properties of cleaned molecule
        try:
            Chem.SanitizeMol(cleaned_mol)
            cleaned_mol.UpdatePropertyCache(strict=False)
        except Exception:
            # Continue with unsanitized molecule if sanitization fails
            pass
        
        mw = Descriptors.ExactMolWt(cleaned_mol)
        
        if mw <= 0:
            logger.warning(f"Molecular weight is {mw} for SMILES: {smiles}")
            
        return float(mw)
        
    except Exception as e:
        raise PolymerFeatureError(f"Error calculating molecular weight for {smiles}: {str(e)}")


def encode_degree_polymerization(dp: Union[float, int], 
                                max_dp: int = 10000, 
                                log_scale: bool = True) -> np.ndarray:
    """
    Encode degree of polymerization as a numerical feature.
    Uses log scaling by default to handle high DP values effectively.
    
    Args:
        dp: Degree of polymerization (number of repeat units)
        max_dp: Maximum expected DP for normalization
        log_scale: Whether to use log scaling (recommended for large DP)
        
    Returns:
        1D numpy array with encoded DP feature
        
    Raises:
        PolymerFeatureError: If DP is invalid
        
    Examples:
        >>> encode_degree_polymerization(100, log_scale=True)
        array([4.605])  # ln(100)
        >>> encode_degree_polymerization(100, log_scale=False)
        array([0.01])   # 100/10000
    """
    if pd.isna(dp) or dp is None:
        # Return default value for missing DP
        dp = 1.0
        logger.warning("Missing DP value, using default DP=1.0")
    
    if dp <= 0:
        raise PolymerFeatureError(f"DP must be positive, got: {dp}")
    
    try:
        if log_scale:
            encoded_value = np.log(dp)
        else:
            # Normalize by max_dp
            encoded_value = dp / max_dp
            
        return np.array([encoded_value], dtype=np.float32)
        
    except Exception as e:
        raise PolymerFeatureError(f"Error encoding DP {dp}: {str(e)}")


def extract_repetition_unit_features(smiles: str, 
                                   fingerprint_size: int = 128,
                                   radius: int = 2) -> np.ndarray:
    """
    Extract structural features from the polymer repetition unit using Morgan fingerprint.
    Removes dummy atoms (*) before feature extraction for accurate structural representation.
    
    Args:
        smiles: SMILES string of the repetition unit
        fingerprint_size: Size of the Morgan fingerprint bit vector
        radius: Radius for Morgan fingerprint generation
        
    Returns:
        1D numpy array with Morgan fingerprint features
        
    Raises:
        PolymerFeatureError: If SMILES processing fails
        
    Examples:
        >>> fp = extract_repetition_unit_features('*CC*')
        >>> fp.shape
        (128,)
        >>> fp.dtype
        dtype('float32')
    """
    if not smiles or pd.isna(smiles):
        raise PolymerFeatureError("SMILES string is empty or NaN")
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise PolymerFeatureError(f"Invalid SMILES: {smiles}")
        
        # Remove dummy atoms (*)
        editable_mol = Chem.EditableMol(mol)
        dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
        
        # Remove in reverse order to maintain indices
        for idx in sorted(dummy_indices, reverse=True):
            editable_mol.RemoveAtom(idx)
        
        unit_mol = editable_mol.GetMol()
        if unit_mol is None or unit_mol.GetNumAtoms() == 0:
            logger.warning(f"Empty molecule after removing dummy atoms from: {smiles}")
            # Return zero vector for empty molecules
            return np.zeros(fingerprint_size, dtype=np.float32)
        
        # Ensure molecule is properly initialized before fingerprint generation
        try:
            # Update properties to ensure RingInfo is initialized
            Chem.FastFindRings(unit_mol)
            unit_mol.UpdatePropertyCache(strict=False)
        except Exception:
            # If ring finding fails, continue with basic properties
            pass
        
        # Generate Morgan fingerprint with comprehensive warning suppression
        with suppress_rdkit_warnings():
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    unit_mol, 
                    radius=radius, 
                    nBits=fingerprint_size
                )
            except RuntimeError as e:
                # Handle RingInfo or other runtime errors
                logger.warning(f"Failed to generate fingerprint for {smiles}: {e}")
                return np.zeros(fingerprint_size, dtype=np.float32)
        
        # Convert to numpy array
        fp_array = np.array(fp, dtype=np.float32)
        
        return fp_array
        
    except Exception as e:
        raise PolymerFeatureError(f"Error extracting unit features for {smiles}: {str(e)}")


def calculate_chain_length_descriptors(smiles: str, dp: Union[float, int] = 1) -> np.ndarray:
    """
    Calculate chain length descriptors for polymer chains.
    
    Features calculated:
    1. Chain flexibility (rotatable bonds per repeat unit)
    2. Persistence length estimate (based on rigidity)
    3. End-to-end distance estimate
    4. Radius of gyration estimate
    5. Chain compactness factor
    
    Args:
        smiles: SMILES string of the repetition unit
        dp: Degree of polymerization
        
    Returns:
        1D numpy array with chain length descriptors [5 features]
        
    Raises:
        PolymerFeatureError: If SMILES processing fails
    """
    if not smiles or pd.isna(smiles):
        raise PolymerFeatureError("SMILES string is empty or NaN")
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise PolymerFeatureError(f"Invalid SMILES: {smiles}")
        
        # Remove dummy atoms for analysis
        editable_mol = Chem.EditableMol(mol)
        dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
        for idx in sorted(dummy_indices, reverse=True):
            editable_mol.RemoveAtom(idx)
        
        unit_mol = editable_mol.GetMol()
        if unit_mol is None or unit_mol.GetNumAtoms() == 0:
            return np.zeros(5, dtype=np.float32)
        
        # Ensure proper initialization
        try:
            Chem.FastFindRings(unit_mol)
            unit_mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        
        # 1. Chain flexibility (rotatable bonds per unit)
        try:
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(unit_mol)
        except Exception:
            rotatable_bonds = 0
        
        flexibility = float(rotatable_bonds)
        
        # 2. Persistence length estimate (inverse of flexibility, scaled)
        # Higher rigidity = higher persistence length
        ring_count = rdMolDescriptors.CalcNumRings(unit_mol) if unit_mol.GetNumAtoms() > 0 else 0
        rigidity_factor = 1.0 + ring_count * 2.0  # Rings increase rigidity
        flexibility_factor = 1.0 / (1.0 + rotatable_bonds) if rotatable_bonds > 0 else 1.0
        persistence_length_est = rigidity_factor * flexibility_factor * 10.0  # Scaled estimate
        
        # 3. End-to-end distance estimate (Flory random coil model)
        # R_ee ~ sqrt(N) * l where N is number of bonds, l is bond length
        num_bonds_per_unit = unit_mol.GetNumBonds() if unit_mol.GetNumAtoms() > 1 else 1
        total_bonds = dp * num_bonds_per_unit
        end_to_end_est = math.sqrt(total_bonds) * 1.54  # 1.54 Å average bond length
        
        # 4. Radius of gyration estimate
        # R_g ~ R_ee / sqrt(6) for random coil
        radius_gyration_est = end_to_end_est / math.sqrt(6)
        
        # 5. Chain compactness factor
        # Ratio of actual size to ideal random coil
        # Lower values = more compact
        ideal_volume = (4/3) * math.pi * (radius_gyration_est ** 3)
        unit_volume_est = unit_mol.GetNumAtoms() * 20.0  # Rough atomic volume estimate
        total_volume_est = dp * unit_volume_est
        compactness = total_volume_est / ideal_volume if ideal_volume > 0 else 1.0
        
        chain_descriptors = np.array([
            flexibility,
            persistence_length_est,
            np.log10(end_to_end_est) if end_to_end_est > 0 else 0,  # Log scale for distances
            np.log10(radius_gyration_est) if radius_gyration_est > 0 else 0,
            np.log10(compactness) if compactness > 0 else 0
        ], dtype=np.float32)
        
        return chain_descriptors
        
    except Exception as e:
        raise PolymerFeatureError(f"Error calculating chain descriptors for {smiles}: {str(e)}")


def calculate_repetition_unit_complexity(smiles: str) -> np.ndarray:
    """
    Calculate structural complexity measures for the repetition unit.
    
    Features calculated:
    1. Ring complexity (number and types of rings)
    2. Heteroatom ratio
    3. Bond type diversity
    4. Branching factor
    5. Stereochemical complexity
    6. Aromaticity index
    
    Args:
        smiles: SMILES string of the repetition unit
        
    Returns:
        1D numpy array with complexity descriptors [6 features]
        
    Raises:
        PolymerFeatureError: If SMILES processing fails
    """
    if not smiles or pd.isna(smiles):
        raise PolymerFeatureError("SMILES string is empty or NaN")
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise PolymerFeatureError(f"Invalid SMILES: {smiles}")
        
        # Remove dummy atoms for analysis
        editable_mol = Chem.EditableMol(mol)
        dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
        for idx in sorted(dummy_indices, reverse=True):
            editable_mol.RemoveAtom(idx)
        
        unit_mol = editable_mol.GetMol()
        if unit_mol is None or unit_mol.GetNumAtoms() == 0:
            return np.zeros(6, dtype=np.float32)
        
        # Ensure proper initialization
        try:
            Chem.FastFindRings(unit_mol)
            unit_mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        
        num_atoms = unit_mol.GetNumAtoms()
        num_bonds = unit_mol.GetNumBonds()
        
        if num_atoms == 0:
            return np.zeros(6, dtype=np.float32)
        
        # 1. Ring complexity
        num_rings = rdMolDescriptors.CalcNumRings(unit_mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(unit_mol)
        ring_complexity = float(num_rings + 0.5 * num_aromatic_rings)  # Aromatic rings add complexity
        
        # 2. Heteroatom ratio
        hetero_atoms = sum(1 for atom in unit_mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        heteroatom_ratio = float(hetero_atoms) / num_atoms if num_atoms > 0 else 0.0
        
        # 3. Bond type diversity
        bond_types = set()
        for bond in unit_mol.GetBonds():
            bond_types.add(bond.GetBondType())
        bond_diversity = float(len(bond_types))
        
        # 4. Branching factor
        # Average degree of non-hydrogen atoms
        degrees = [atom.GetDegree() for atom in unit_mol.GetAtoms() if atom.GetAtomicNum() != 1]
        branching_factor = float(np.mean(degrees)) if degrees else 0.0
        
        # 5. Stereochemical complexity
        # Count stereocenters and double bond stereochemistry
        stereo_centers = len(Chem.FindMolChiralCenters(unit_mol, includeUnassigned=True))
        stereo_bonds = sum(1 for bond in unit_mol.GetBonds() 
                          if bond.GetStereo() != Chem.BondStereo.STEREONONE)
        stereo_complexity = float(stereo_centers + stereo_bonds)
        
        # 6. Aromaticity index
        aromatic_atoms = sum(1 for atom in unit_mol.GetAtoms() if atom.GetIsAromatic())
        aromaticity_index = float(aromatic_atoms) / num_atoms if num_atoms > 0 else 0.0
        
        complexity_descriptors = np.array([
            ring_complexity,
            heteroatom_ratio,
            bond_diversity,
            branching_factor,
            stereo_complexity,
            aromaticity_index
        ], dtype=np.float32)
        
        return complexity_descriptors
        
    except Exception as e:
        raise PolymerFeatureError(f"Error calculating unit complexity for {smiles}: {str(e)}")


def calculate_polymer_molecular_descriptors(smiles: str, dp: Union[float, int] = 1) -> np.ndarray:
    """
    Calculate polymer-specific molecular descriptors relevant to properties like Tg.
    
    Features calculated:
    1. Free volume fraction estimate
    2. Chain stiffness parameter
    3. Intermolecular interaction strength
    4. Packing efficiency estimate
    5. Glass transition predictors (flexibility, polarity)
    6. Crystallinity indicators
    
    Args:
        smiles: SMILES string of the repetition unit
        dp: Degree of polymerization
        
    Returns:
        1D numpy array with polymer molecular descriptors [6 features]
        
    Raises:
        PolymerFeatureError: If SMILES processing fails
    """
    if not smiles or pd.isna(smiles):
        raise PolymerFeatureError("SMILES string is empty or NaN")
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise PolymerFeatureError(f"Invalid SMILES: {smiles}")
        
        # Remove dummy atoms for analysis
        editable_mol = Chem.EditableMol(mol)
        dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
        for idx in sorted(dummy_indices, reverse=True):
            editable_mol.RemoveAtom(idx)
        
        unit_mol = editable_mol.GetMol()
        if unit_mol is None or unit_mol.GetNumAtoms() == 0:
            return np.zeros(6, dtype=np.float32)
        
        # Ensure proper initialization
        try:
            Chem.FastFindRings(unit_mol)
            unit_mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        
        num_atoms = unit_mol.GetNumAtoms()
        
        if num_atoms == 0:
            return np.zeros(6, dtype=np.float32)
        
        # 1. Free volume fraction estimate
        # Based on van der Waals volume vs molar volume
        try:
            vdw_volume = rdMolDescriptors.CalcTPSA(unit_mol)  # Use TPSA as proxy
            molecular_weight = Descriptors.ExactMolWt(unit_mol)
            density_est = 1.0  # Assume ~1 g/cm³ density
            molar_volume = molecular_weight / density_est
            free_volume_fraction = max(0, (molar_volume - vdw_volume) / molar_volume) if molar_volume > 0 else 0
        except Exception:
            free_volume_fraction = 0.1  # Default estimate
        
        # 2. Chain stiffness parameter
        # Based on rings and double bonds in backbone
        rings = rdMolDescriptors.CalcNumRings(unit_mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(unit_mol)
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(unit_mol)
        
        stiffness = float(rings * 2.0 + aromatic_rings * 3.0) / max(1, rotatable_bonds + 1)
        
        # 3. Intermolecular interaction strength
        # Based on polar surface area and hydrogen bonding
        try:
            polar_surface_area = rdMolDescriptors.CalcTPSA(unit_mol)
            hbd_count = rdMolDescriptors.CalcNumHBD(unit_mol)  # H-bond donors
            hba_count = rdMolDescriptors.CalcNumHBA(unit_mol)  # H-bond acceptors
            interaction_strength = (polar_surface_area / 100.0) + (hbd_count + hba_count) * 0.5
        except Exception:
            interaction_strength = 0.0
        
        # 4. Packing efficiency estimate
        # Based on molecular shape and branching (using alternative to Kappa1)
        try:
            # Alternative shape descriptor using molecular graph properties
            heavy_atoms = [atom for atom in unit_mol.GetAtoms() if atom.GetAtomicNum() > 1]
            if len(heavy_atoms) > 2:
                # Simple shape measure: ratio of actual bonds to possible bonds
                actual_bonds = unit_mol.GetNumBonds()
                max_possible_bonds = len(heavy_atoms) * (len(heavy_atoms) - 1) // 2
                shape_factor = actual_bonds / max_possible_bonds if max_possible_bonds > 0 else 0.5
            else:
                shape_factor = 0.5
        except Exception:
            shape_factor = 0.5
        
        branching_ratio = sum(1 for atom in unit_mol.GetAtoms() 
                             if atom.GetDegree() > 2 and atom.GetAtomicNum() != 1) / num_atoms
        packing_efficiency = shape_factor / (1.0 + branching_ratio * 2.0)
        
        # 5. Glass transition predictors
        # Combine flexibility and polarity effects
        flexibility_factor = float(rotatable_bonds) / num_atoms if num_atoms > 0 else 0
        try:
            polarity_factor = polar_surface_area / (molecular_weight + 1)  # Normalized polarity
        except Exception:
            polarity_factor = 0.0
        
        tg_predictor = stiffness - flexibility_factor * 0.5 + polarity_factor * 2.0
        
        # 6. Crystallinity indicators
        # Regularity and symmetry measures
        symmetry_factor = float(aromatic_rings + rings) / num_atoms if num_atoms > 0 else 0
        regularity_factor = 1.0 / (1.0 + branching_ratio * 3.0)  # Less branching = more regular
        crystallinity_indicator = (symmetry_factor + regularity_factor) / 2.0
        
        polymer_descriptors = np.array([
            free_volume_fraction,
            np.log10(stiffness + 1),  # Log scale for stiffness
            interaction_strength,
            packing_efficiency,
            tg_predictor,
            crystallinity_indicator
        ], dtype=np.float32)
        
        return polymer_descriptors
        
    except Exception as e:
        raise PolymerFeatureError(f"Error calculating polymer descriptors for {smiles}: {str(e)}")


def extract_polymer_features(smiles: str, 
                           dp: Optional[Union[float, int]] = None, 
                           mw: Optional[float] = None,
                           fingerprint_size: int = 128,
                           log_scale_dp: bool = True,
                           include_chain_descriptors: bool = True,
                           include_complexity: bool = True,
                           include_molecular_descriptors: bool = True) -> torch.Tensor:
    """
    Extract comprehensive polymer features combining all components.
    
    Feature composition:
    - Molecular weight of repeating unit (1 feature)
    - Encoded degree of polymerization (1 feature) 
    - Morgan fingerprint of repeating unit (128 features by default)
    - Chain length descriptors (5 features) [optional]
    - Repetition unit complexity (6 features) [optional]
    - Polymer molecular descriptors (6 features) [optional]
    Total: 147 features by default with all components
    
    Args:
        smiles: SMILES string of the polymer repeating unit
        dp: Degree of polymerization (if None and mw provided, will estimate)
        mw: Total molecular weight (used to estimate DP if dp is None)
        fingerprint_size: Size of Morgan fingerprint
        log_scale_dp: Whether to use log scaling for DP
        include_chain_descriptors: Whether to include chain length descriptors
        include_complexity: Whether to include repetition unit complexity features
        include_molecular_descriptors: Whether to include polymer-specific descriptors
        
    Returns:
        PyTorch tensor with combined polymer features
        
    Raises:
        PolymerFeatureError: If feature extraction fails
        
    Examples:
        >>> features = extract_polymer_features('*CC*', dp=1000)
        >>> features.shape
        torch.Size([147])  # With all features enabled
        >>> features.dtype
        torch.float32
    """
    try:
        # Calculate molecular weight of repeating unit
        unit_mw = calculate_molecular_weight(smiles)
        
        # Determine degree of polymerization
        if dp is None and mw is not None:
            # Estimate DP from total molecular weight
            dp = mw / unit_mw if unit_mw > 0 else 1.0
            logger.debug(f"Estimated DP: {dp} from MW: {mw}, unit_mw: {unit_mw}")
        elif dp is None:
            # Default to DP = 1 for missing information
            dp = 1.0
            # Only log this warning once per session to avoid spam
            if not hasattr(extract_polymer_features, '_dp_warning_logged'):
                logger.info("DP and MW values not provided - using default DP=1.0 for all molecules")
                extract_polymer_features._dp_warning_logged = True
        
        # Extract core features
        dp_encoded = encode_degree_polymerization(dp, log_scale=log_scale_dp)
        unit_features = extract_repetition_unit_features(smiles, fingerprint_size)
        
        # Start with base features
        features_list = [
            [unit_mw],        # Molecular weight (1 feature)
            dp_encoded,       # DP encoding (1 feature)
            unit_features     # Morgan fingerprint (fingerprint_size features)
        ]
        
        # Add optional feature sets
        if include_chain_descriptors:
            chain_descriptors = calculate_chain_length_descriptors(smiles, dp)
            features_list.append(chain_descriptors)
        
        if include_complexity:
            complexity_features = calculate_repetition_unit_complexity(smiles)
            features_list.append(complexity_features)
        
        if include_molecular_descriptors:
            molecular_descriptors = calculate_polymer_molecular_descriptors(smiles, dp)
            features_list.append(molecular_descriptors)
        
        # Combine all features
        features = np.concatenate(features_list)
        
        # Convert to PyTorch tensor
        tensor_features = torch.tensor(features, dtype=torch.float32)
        
        logger.debug(f"Extracted {len(tensor_features)} polymer features for {smiles}")
        
        return tensor_features
        
    except Exception as e:
        raise PolymerFeatureError(f"Error extracting polymer features for {smiles}: {str(e)}")


class PolymerFeatureExtractor:
    """
    Comprehensive polymer feature extractor with configuration options.
    Provides a convenient interface for batch processing and feature caching.
    """
    
    def __init__(self, 
                 fingerprint_size: int = 128,
                 fp_radius: int = 2,
                 log_scale_dp: bool = True,
                 max_dp: int = 10000,
                 cache_features: bool = False,
                 include_chain_descriptors: bool = True,
                 include_complexity: bool = True,
                 include_molecular_descriptors: bool = True):
        """
        Initialize the polymer feature extractor.
        
        Args:
            fingerprint_size: Size of Morgan fingerprint
            fp_radius: Radius for Morgan fingerprint
            log_scale_dp: Whether to use log scaling for DP
            max_dp: Maximum DP for normalization
            cache_features: Whether to cache computed features
            include_chain_descriptors: Include chain length descriptors
            include_complexity: Include repetition unit complexity features
            include_molecular_descriptors: Include polymer-specific descriptors
        """
        self.fingerprint_size = fingerprint_size
        self.fp_radius = fp_radius
        self.log_scale_dp = log_scale_dp
        self.max_dp = max_dp
        self.cache_features = cache_features
        self.include_chain_descriptors = include_chain_descriptors
        self.include_complexity = include_complexity
        self.include_molecular_descriptors = include_molecular_descriptors
        
        self.feature_cache = {} if cache_features else None
        
        # Calculate feature dimensionality
        self.feature_dim = 1 + 1 + fingerprint_size  # mw + dp + fingerprint
        if include_chain_descriptors:
            self.feature_dim += 5  # Chain descriptors
        if include_complexity:
            self.feature_dim += 6  # Complexity features
        if include_molecular_descriptors:
            self.feature_dim += 6  # Polymer descriptors
        
        logger.info(f"Initialized PolymerFeatureExtractor with {self.feature_dim} features")
    
    def extract_features(self, 
                        smiles: str, 
                        dp: Optional[Union[float, int]] = None, 
                        mw: Optional[float] = None) -> torch.Tensor:
        """
        Extract features for a single polymer.
        
        Args:
            smiles: SMILES string
            dp: Degree of polymerization
            mw: Total molecular weight
            
        Returns:
            Feature tensor
        """
        # Check cache if enabled
        cache_key = (f"{smiles}_{dp}_{mw}_{self.include_chain_descriptors}_"
                    f"{self.include_complexity}_{self.include_molecular_descriptors}") if self.cache_features else None
        
        if cache_key and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Extract features
        features = extract_polymer_features(
            smiles, 
            dp=dp, 
            mw=mw,
            fingerprint_size=self.fingerprint_size,
            log_scale_dp=self.log_scale_dp,
            include_chain_descriptors=self.include_chain_descriptors,
            include_complexity=self.include_complexity,
            include_molecular_descriptors=self.include_molecular_descriptors
        )
        
        # Cache if enabled
        if cache_key:
            self.feature_cache[cache_key] = features
        
        return features
    
    def extract_batch_features(self, 
                             smiles_list: list, 
                             dp_list: Optional[list] = None,
                             mw_list: Optional[list] = None) -> torch.Tensor:
        """
        Extract features for a batch of polymers.
        
        Args:
            smiles_list: List of SMILES strings
            dp_list: List of DP values (optional)
            mw_list: List of MW values (optional)
            
        Returns:
            Stacked feature tensor [batch_size, feature_dim]
        """
        if dp_list is None:
            dp_list = [None] * len(smiles_list)
        if mw_list is None:
            mw_list = [None] * len(smiles_list)
        
        features_list = []
        
        for smiles, dp, mw in zip(smiles_list, dp_list, mw_list):
            try:
                features = self.extract_features(smiles, dp, mw)
                features_list.append(features)
            except PolymerFeatureError as e:
                logger.error(f"Failed to extract features for {smiles}: {e}")
                # Use zero features as fallback
                zero_features = torch.zeros(self.feature_dim, dtype=torch.float32)
                features_list.append(zero_features)
        
        return torch.stack(features_list)
    
    def get_feature_names(self) -> List[str]:
        """Get descriptive names for all features."""
        names = ['unit_molecular_weight', 'degree_polymerization']
        names += [f'morgan_fp_{i}' for i in range(self.fingerprint_size)]
        
        if self.include_chain_descriptors:
            names += [
                'chain_flexibility',
                'persistence_length_est',
                'end_to_end_distance_log',
                'radius_gyration_log', 
                'chain_compactness_log'
            ]
        
        if self.include_complexity:
            names += [
                'ring_complexity',
                'heteroatom_ratio',
                'bond_diversity',
                'branching_factor',
                'stereochemical_complexity',
                'aromaticity_index'
            ]
        
        if self.include_molecular_descriptors:
            names += [
                'free_volume_fraction',
                'chain_stiffness_log',
                'interaction_strength',
                'packing_efficiency',
                'tg_predictor',
                'crystallinity_indicator'
            ]
        
        return names
    
    def get_feature_dim(self) -> int:
        """Get total feature dimensionality."""
        return self.feature_dim
    
    def get_feature_groups(self) -> Dict[str, List[int]]:
        """Get feature indices grouped by type for analysis."""
        groups = {}
        idx = 0
        
        # Core features
        groups['molecular_weight'] = [idx]
        idx += 1
        groups['degree_polymerization'] = [idx]
        idx += 1
        groups['morgan_fingerprint'] = list(range(idx, idx + self.fingerprint_size))
        idx += self.fingerprint_size
        
        # Optional feature groups
        if self.include_chain_descriptors:
            groups['chain_descriptors'] = list(range(idx, idx + 5))
            idx += 5
        
        if self.include_complexity:
            groups['complexity'] = list(range(idx, idx + 6))
            idx += 6
        
        if self.include_molecular_descriptors:
            groups['molecular_descriptors'] = list(range(idx, idx + 6))
            idx += 6
        
        return groups


# Import pandas for NaN checking
import pandas as pd




 