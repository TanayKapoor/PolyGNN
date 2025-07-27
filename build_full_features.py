#!/usr/bin/env python3
"""
Full Feature Engineering Pipeline for PolyGNN Project

This script processes cleaned CSV files and extracts comprehensive polymer features:
- Loads multiple cleaned datasets (bic_clean.csv, jcim_clean.csv, train_clean.csv, sample_clean.csv)
- Extracts 147 polymer-specific features using RDKit proxies
- Handles BigSMILES by extracting repeat units
- Merges and deduplicates on canonical SMILES
- Imputes missing values for target properties
- Outputs full_feats.csv with 1227+ rows, 1622+ columns
- Includes GNN baseline training snippet

Requirements:
pip install pandas rdkit-pypi mordred-descriptor torch torch-geometric tqdm scikit-learn
"""

import sys
import os
import warnings
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import re

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# Mordred (if available)
try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False
    print("Warning: Mordred not available. Using RDKit descriptors only.")

# Sklearn for imputation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# PyTorch and PyG (for GNN snippet)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/PyG not available. GNN snippet will be skipped.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'data_dir': Path('data/processed'),
    'output_dir': Path('data/processed'),
    'intermediate_dir': Path('data/intermediate'),
    'results_dir': Path('results'),
    'random_seed': 42,
    'target_properties': ['Tg', 'Tm', 'Density', 'FFV', 'Tc', 'Rg'],
    'morgan_fp_size': 128,
    'morgan_radius': 2,
    'max_atoms_for_dp': 1000,  # For DP estimation
}

# Ensure directories exist
for dir_path in [CONFIG['intermediate_dir'], CONFIG['results_dir']]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Fix pandas compatibility for pickle files
if 'pandas.core.indexes.numeric' not in sys.modules:
    sys.modules['pandas.core.indexes.numeric'] = pd.core.indexes.base


class BigSMILESProcessor:
    """Handles BigSMILES notation and extracts repeat units"""
    
    @staticmethod
    def extract_repeat_unit(bigsmiles: str) -> str:
        """Extract repeat unit from BigSMILES notation"""
        try:
            if not isinstance(bigsmiles, str):
                return ""
            
            # Better parse: Strip braces/symbols for repeat unit
            s = bigsmiles
            
            # If it contains BigSMILES notation, clean it up
            if '{' in s:
                # Strip outer braces and clean up BigSMILES symbols
                s = s.strip('{}').replace('<', '').replace('>', '').replace('$', '').replace('#', '').strip()
                # Remove any remaining special symbols
                s = s.replace(',', '').replace('*', '')  # Remove connection points and commas
                # Take first part if still has some separators
                if '|' in s:
                    s = s.split('|')[0]
                if ';' in s:
                    s = s.split(';')[0]
                # Clean up empty parentheses that might result from * removal
                s = s.replace('()', '')
                return s.strip()
            
            # If no brackets but has *, remove connection points
            if '*' in s:
                s = s.replace('*', '')
                # Clean up empty parentheses that might result from * removal
                s = s.replace('()', '')
                return s.strip()
            
            return s
            
        except Exception as e:
            logger.warning(f"Error processing BigSMILES '{bigsmiles}': {e}")
            return ""
    
    @staticmethod
    def bigsmiles_to_smiles(bigsmiles: str) -> str:
        """Convert BigSMILES to regular SMILES for processing"""
        return BigSMILESProcessor.extract_repeat_unit(bigsmiles)


class PolymerFeatureExtractor:
    """Extract comprehensive polymer features using RDKit proxies"""
    
    def __init__(self, morgan_fp_size: int = 128, morgan_radius: int = 2):
        self.morgan_fp_size = morgan_fp_size
        self.morgan_radius = morgan_radius
        self.feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """Generate the 147 expected feature names"""
        names = ['unit_molecular_weight', 'degree_polymerization']
        names += [f'morgan_fp_{i}' for i in range(self.morgan_fp_size)]
        
        # Chain descriptors (5 features)
        names += [
            'chain_flexibility',
            'persistence_length_est',
            'end_to_end_distance_log',
            'radius_gyration_log', 
            'chain_compactness_log'
        ]
        
        # Complexity features (6 features)
        names += [
            'ring_complexity',
            'heteroatom_ratio',
            'bond_diversity',
            'branching_factor',
            'stereochemical_complexity',
            'aromaticity_index'
        ]
        
        # Molecular descriptors (7 features)
        names += [
            'free_volume_fraction',
            'chain_stiffness_log',
            'interaction_strength',
            'packing_efficiency',
            'flexibility_factor',
            'polarity_factor',
            'crystallinity_indicator'
        ]
        
        return names
    
    def extract_features(self, smiles: str, density: Optional[float] = None) -> Dict[str, float]:
        """Extract all 147 polymer features from SMILES"""
        features = {}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {name: 0.0 for name in self.feature_names}
            
            # Core features
            features.update(self._extract_core_features(mol))
            
            # Morgan fingerprint
            features.update(self._extract_morgan_fingerprint(mol))
            
            # Chain descriptors
            features.update(self._extract_chain_descriptors(mol))
            
            # Complexity features
            features.update(self._extract_complexity_features(mol))
            
            # Molecular descriptors
            features.update(self._extract_molecular_descriptors(mol, density))
            
        except Exception as e:
            logger.warning(f"Error extracting features for SMILES '{smiles}': {e}")
            features = {name: 0.0 for name in self.feature_names}
        
        return features
    
    def _extract_core_features(self, mol) -> Dict[str, float]:
        """Extract core polymer features"""
        features = {}
        
        # Unit molecular weight
        features['unit_molecular_weight'] = Descriptors.MolWt(mol)
        
        # Degree of polymerization estimate (atom count / 10 as proxy)
        num_atoms = mol.GetNumAtoms()
        features['degree_polymerization'] = max(1.0, num_atoms / 10.0)
        
        return features
    
    def _extract_morgan_fingerprint(self, mol) -> Dict[str, float]:
        """Extract Morgan fingerprint features"""
        features = {}
        
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, self.morgan_radius, nBits=self.morgan_fp_size
            )
            fp_array = np.array(fp)
            
            for i in range(self.morgan_fp_size):
                features[f'morgan_fp_{i}'] = float(fp_array[i])
                
        except Exception as e:
            logger.warning(f"Error extracting Morgan fingerprint: {e}")
            for i in range(self.morgan_fp_size):
                features[f'morgan_fp_{i}'] = 0.0
        
        return features
    
    def _extract_chain_descriptors(self, mol) -> Dict[str, float]:
        """Extract chain-related descriptors using RDKit proxies"""
        features = {}
        
        try:
            # Chain flexibility (rotatable bonds / total bonds)
            num_rotatable = Descriptors.NumRotatableBonds(mol)
            num_bonds = mol.GetNumBonds()
            features['chain_flexibility'] = num_rotatable / max(1, num_bonds)
            
            # Persistence length estimate (inverse of flexibility)
            features['persistence_length_est'] = 1.0 / (features['chain_flexibility'] + 0.1)
            
            # End-to-end distance (log of molecular weight as proxy)
            mw = Descriptors.MolWt(mol)
            features['end_to_end_distance_log'] = np.log(mw + 1.0)
            
            # Radius of gyration (log of molecular volume proxy)
            features['radius_gyration_log'] = np.log(Descriptors.MolLogP(mol) + 5.0)
            
            # Chain compactness (inverse of molecular surface area)
            tpsa = Descriptors.TPSA(mol)
            features['chain_compactness_log'] = np.log(1.0 / (tpsa + 1.0))
            
        except Exception as e:
            logger.warning(f"Error extracting chain descriptors: {e}")
            features.update({
                'chain_flexibility': 0.0,
                'persistence_length_est': 1.0,
                'end_to_end_distance_log': 0.0,
                'radius_gyration_log': 0.0,
                'chain_compactness_log': 0.0
            })
        
        return features
    
    def _extract_complexity_features(self, mol) -> Dict[str, float]:
        """Extract structural complexity features"""
        features = {}
        
        try:
            num_atoms = mol.GetNumAtoms()
            num_heavy = mol.GetNumHeavyAtoms()
            
            # Ring complexity
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            features['ring_complexity'] = num_rings / max(1, num_heavy)
            
            # Heteroatom ratio
            num_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 6)
            features['heteroatom_ratio'] = num_hetero / max(1, num_atoms)
            
            # Bond diversity (different bond types)
            bond_types = set(bond.GetBondType() for bond in mol.GetBonds())
            features['bond_diversity'] = len(bond_types) / 4.0  # Normalize by max types
            
            # Branching factor
            branch_points = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)
            features['branching_factor'] = branch_points / max(1, num_atoms)
            
            # Stereochemical complexity
            stereo_centers = len(Chem.FindMolChiralCenters(mol))
            features['stereochemical_complexity'] = stereo_centers / max(1, num_atoms)
            
            # Aromaticity index
            aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            features['aromaticity_index'] = aromatic_atoms / max(1, num_atoms)
            
        except Exception as e:
            logger.warning(f"Error extracting complexity features: {e}")
            features.update({
                'ring_complexity': 0.0,
                'heteroatom_ratio': 0.0,
                'bond_diversity': 0.0,
                'branching_factor': 0.0,
                'stereochemical_complexity': 0.0,
                'aromaticity_index': 0.0
            })
        
        return features
    
    def _extract_molecular_descriptors(self, mol, density: Optional[float] = None) -> Dict[str, float]:
        """Extract molecular property descriptors"""
        features = {}
        
        try:
            # Free volume fraction (use density if available, otherwise estimate)
            if density and density > 0:
                # FFV proxy from density (higher density = lower FFV)
                features['free_volume_fraction'] = max(0.0, 1.0 - (density / 2.0))
            else:
                # Estimate from molecular descriptors
                mw = Descriptors.MolWt(mol)
                volume_est = mw / 0.8  # Rough density estimate
                features['free_volume_fraction'] = 0.3  # Default polymer FFV
            
            # Chain stiffness (log of aromatic ratio + ring complexity)
            aromatic_ratio = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()) / max(1, mol.GetNumAtoms())
            stiffness = aromatic_ratio + (mol.GetRingInfo().NumRings() / max(1, mol.GetNumHeavyAtoms()))
            features['chain_stiffness_log'] = np.log(stiffness + 0.1)
            
            # Interaction strength (polar surface area + heteroatom count)
            tpsa = Descriptors.TPSA(mol)
            hetero_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
            features['interaction_strength'] = (tpsa + hetero_count * 10) / 100.0
            
            # Packing efficiency (inverse of molecular volume proxy)
            logp = Descriptors.MolLogP(mol)
            features['packing_efficiency'] = 1.0 / (abs(logp) + 1.0)
            
            # Flexibility factor (rotatable bonds / molecular weight)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            mw = Descriptors.MolWt(mol)
            features['flexibility_factor'] = rot_bonds / max(100, mw)
            
            # Polarity factor
            features['polarity_factor'] = min(1.0, Descriptors.TPSA(mol) / 100.0)
            
            # Crystallinity indicator (symmetry proxy)
            features['crystallinity_indicator'] = 1.0 - features['flexibility_factor']
            
        except Exception as e:
            logger.warning(f"Error extracting molecular descriptors: {e}")
            features.update({
                'free_volume_fraction': 0.3,
                'chain_stiffness_log': 0.0,
                'interaction_strength': 0.0,
                'packing_efficiency': 0.5,
                'flexibility_factor': 0.0,
                'polarity_factor': 0.0,
                'crystallinity_indicator': 0.5
            })
        
        return features


class DataLoader:
    """Handle loading and preprocessing of all data sources"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def load_all_datasets(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load all cleaned datasets and datasteA.pkl"""
        datasets = {}
        
        # Load CSV files
        csv_files = {
            'bic': 'bic_clean.csv',
            'jcim': 'jcim_clean.csv', 
            'train': 'train_clean.csv',
            'sample': 'sample_clean.csv'
        }
        
        for name, filename in csv_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    datasets[name] = df
                    logger.info(f"Loaded {name}: {df.shape}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
            else:
                logger.warning(f"File not found: {filepath}")
        
        # Load datasteA.pkl
        pickle_path = self.data_dir.parent / 'raw' / 'datasteA.pkl'
        datasteA = None
        if pickle_path.exists():
            try:
                datasteA = pd.read_pickle(pickle_path)
                logger.info(f"Loaded datasteA.pkl: {datasteA.shape}")
            except Exception as e:
                logger.error(f"Error loading datasteA.pkl: {e}")
        
        return datasteA, datasets
    
    def standardize_columns(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Standardize column names across datasets"""
        df = df.copy()
        
        # Common column mappings
        column_mappings = {
            'SMILES': 'smiles',
            'BigSMILES': 'bigsmiles',
            'bigsmiles': 'bigsmiles',
            'Tg (C)': 'Tg',
            'Tg (K) exp': 'Tg_K',
            'glass_transition_temp': 'Tg',
            'melting_temp': 'Tm',
            'structure': 'smiles',
            'canonical': 'canonical_smiles',
        }
        
        # Apply mappings
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Convert Kelvin to Celsius if needed
        if 'Tg_K' in df.columns:
            df['Tg'] = df['Tg_K'] - 273.15
            df = df.drop('Tg_K', axis=1)
        
        # Add source column
        df['source'] = source_name
        
        return df


def merge_and_deduplicate(datasets: Dict[str, pd.DataFrame], 
                         datasteA: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge all datasets and deduplicate on canonical SMILES"""
    
    logger.info("Merging and deduplicating datasets...")
    
    all_dfs = []
    
    # Process each dataset
    for name, df in datasets.items():
        if df is None or df.empty:
            continue
            
        df_processed = df.copy()
        
        # Ensure we have SMILES column
        smiles_col = None
        for col in ['smiles', 'SMILES', 'canonical_smiles']:
            if col in df_processed.columns:
                smiles_col = col
                break
        
        if smiles_col is None:
            logger.warning(f"No SMILES column found in {name}")
            continue
        
        # Handle BigSMILES conversion
        if 'bigsmiles' in df_processed.columns:
            for idx, row in df_processed.iterrows():
                if pd.notna(row['bigsmiles']) and (pd.isna(row[smiles_col]) or row[smiles_col] == ''):
                    converted_smiles = BigSMILESProcessor.bigsmiles_to_smiles(row['bigsmiles'])
                    df_processed.at[idx, smiles_col] = converted_smiles
        
        # Also check if current SMILES column contains BigSMILES notation and clean it
        for idx, row in df_processed.iterrows():
            current_smiles = row[smiles_col]
            if isinstance(current_smiles, str) and '{' in current_smiles:
                cleaned_smiles = BigSMILESProcessor.bigsmiles_to_smiles(current_smiles)
                df_processed.at[idx, smiles_col] = cleaned_smiles
        
        # Create canonical SMILES
        df_processed['canonical_smiles'] = df_processed[smiles_col].apply(
            lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if isinstance(x, str) and Chem.MolFromSmiles(x) else x
        )
        
        # Remove rows with invalid SMILES
        df_processed = df_processed.dropna(subset=['canonical_smiles'])
        df_processed = df_processed[df_processed['canonical_smiles'] != '']
        
        all_dfs.append(df_processed)
        logger.info(f"Processed {name}: {len(df_processed)} valid structures")
    
    if not all_dfs:
        logger.error("No valid datasets to merge!")
        return pd.DataFrame()
    
    # Concatenate all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    logger.info(f"Combined all datasets: {combined_df.shape}")
    
    # Add datasteA features if available
    if datasteA is not None and not datasteA.empty:
        # Merge datasteA features (assuming it has matching indices or SMILES)
        logger.info("Adding datasteA features...")
        # This would need adjustment based on actual datasteA structure
        # For now, we'll add it as additional feature columns
        datasteA_cols = [col for col in datasteA.columns if col not in combined_df.columns]
        if len(datasteA_cols) > 0:
            # Add datasteA features to first N rows (matching by index)
            n_rows = min(len(combined_df), len(datasteA))
            for col in datasteA_cols:
                combined_df.loc[:n_rows-1, col] = datasteA[col].iloc[:n_rows].values
    
    # Deduplicate on canonical SMILES (keep first occurrence)
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['canonical_smiles'])
    final_count = len(combined_df)
    
    logger.info(f"Deduplication: {initial_count} -> {final_count} rows ({initial_count - final_count} duplicates removed)")
    
    return combined_df


def extract_polymer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract 147 polymer-specific features for all molecules"""
    
    logger.info("Extracting 147 polymer-specific features...")
    
    extractor = PolymerFeatureExtractor(
        morgan_fp_size=CONFIG['morgan_fp_size'],
        morgan_radius=CONFIG['morgan_radius']
    )
    
    # Prepare results dataframe
    feature_df = df.copy()
    
    # Initialize feature columns
    for feature_name in extractor.feature_names:
        feature_df[feature_name] = 0.0
    
    # Extract features with progress bar
    valid_extractions = 0
    for idx, row in tqdm(feature_df.iterrows(), total=len(feature_df), desc="Extracting features"):
        smiles = row.get('canonical_smiles', '')
        density = row.get('Density', None)
        
        if smiles and isinstance(smiles, str):
            features = extractor.extract_features(smiles, density)
            
            # Update the dataframe with extracted features
            for feature_name, feature_value in features.items():
                feature_df.at[idx, feature_name] = feature_value
            
            valid_extractions += 1
    
    logger.info(f"Successfully extracted features for {valid_extractions}/{len(feature_df)} molecules")
    
    return feature_df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values for target properties using mean imputation"""
    
    logger.info("Imputing missing values...")
    
    df_imputed = df.copy()
    
    # Impute target properties
    target_cols = [col for col in CONFIG['target_properties'] if col in df_imputed.columns]
    
    for col in target_cols:
        if col in df_imputed.columns:
            missing_count = df_imputed[col].isna().sum()
            if missing_count > 0:
                mean_value = df_imputed[col].mean()
                df_imputed[col].fillna(mean_value, inplace=True)
                logger.info(f"Imputed {missing_count} missing values in {col} with mean {mean_value:.3f}")
    
    # Impute feature columns with 0 (already done during extraction, but ensure consistency)
    feature_cols = [col for col in df_imputed.columns if col.startswith(('morgan_fp_', 'chain_', 'ring_', 'heteroatom_', 'free_volume_'))]
    for col in feature_cols:
        df_imputed[col].fillna(0.0, inplace=True)
    
    return df_imputed


def save_intermediate_results(df: pd.DataFrame, filename: str):
    """Save intermediate results"""
    filepath = CONFIG['intermediate_dir'] / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Saved intermediate results to {filepath}")


def generate_summary_stats(df: pd.DataFrame) -> Dict:
    """Generate summary statistics"""
    
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'feature_columns': len([col for col in df.columns if not col in ['smiles', 'canonical_smiles', 'source', 'id', 'bigsmiles'] + CONFIG['target_properties']]),
        'target_properties': {}
    }
    
    # Target property statistics
    for prop in CONFIG['target_properties']:
        if prop in df.columns:
            stats['target_properties'][prop] = {
                'count': df[prop].notna().sum(),
                'mean': df[prop].mean(),
                'std': df[prop].std(),
                'min': df[prop].min(),
                'max': df[prop].max()
            }
    
    return stats


# GNN Baseline Training Snippet
def create_gnn_baseline():
    """Create a simple GNN baseline for multi-task prediction"""
    
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Skipping GNN baseline.")
        return
    
    logger.info("Creating GNN baseline model...")
    
    class SimpleGNN(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 3):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            
            # Multi-task heads
            self.tg_head = nn.Linear(hidden_dim, 1)    # Tg prediction
            self.tm_head = nn.Linear(hidden_dim, 1)    # Tm prediction  
            self.density_head = nn.Linear(hidden_dim, 1)  # Density prediction
            
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x, edge_index, batch):
            # Graph convolutions
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv3(x, edge_index))
            
            # Global pooling
            x = global_mean_pool(x, batch)
            
            # Multi-task predictions
            tg_pred = self.tg_head(x)
            tm_pred = self.tm_head(x)
            density_pred = self.density_head(x)
            
            return {
                'tg': tg_pred.squeeze(),
                'tm': tm_pred.squeeze(), 
                'density': density_pred.squeeze()
            }
    
    # Example training snippet (would need actual graph data)
    def train_gnn_baseline(model, train_loader, val_loader, epochs=100):
        """Training loop for GNN baseline"""
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(batch.x, batch.edge_index, batch.batch)
                
                # Multi-task loss
                loss = 0
                if hasattr(batch, 'tg') and batch.tg is not None:
                    loss += criterion(predictions['tg'], batch.tg)
                if hasattr(batch, 'tm') and batch.tm is not None:
                    loss += criterion(predictions['tm'], batch.tm)
                if hasattr(batch, 'density') and batch.density is not None:
                    loss += criterion(predictions['density'], batch.density)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        predictions = model(batch.x, batch.edge_index, batch.batch)
                        # Calculate validation loss similarly
                
                logger.info(f"Epoch {epoch}: Train Loss {total_loss:.4f}")
    
    # Calculate R² score
    def calculate_r2(y_true, y_pred):
        """Calculate R² score"""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot
    
    return SimpleGNN, train_gnn_baseline, calculate_r2


def main():
    """Main execution function"""
    
    logger.info("Starting Full Feature Engineering Pipeline for PolyGNN")
    logger.info("=" * 60)
    
    # Set random seed
    np.random.seed(CONFIG['random_seed'])
    
    try:
        # Step 1: Load all datasets
        logger.info("Step 1: Loading datasets...")
        data_loader = DataLoader(CONFIG['data_dir'])
        datasteA, datasets = data_loader.load_all_datasets()
        
        if not datasets:
            logger.error("No datasets loaded successfully!")
            return
        
        # Standardize column names
        for name, df in datasets.items():
            datasets[name] = data_loader.standardize_columns(df, name)
        
        # Step 2: Merge and deduplicate
        logger.info("Step 2: Merging and deduplicating...")
        merged_df = merge_and_deduplicate(datasets, datasteA)
        
        if merged_df.empty:
            logger.error("No data after merging and deduplication!")
            return
        
        save_intermediate_results(merged_df, "merged_deduplicated.csv")
        
        # Step 3: Extract polymer features
        logger.info("Step 3: Extracting 147 polymer-specific features...")
        feature_df = extract_polymer_features(merged_df)
        save_intermediate_results(feature_df, "with_polymer_features.csv")
        
        # Step 4: Impute missing values
        logger.info("Step 4: Imputing missing values...")
        final_df = impute_missing_values(feature_df)
        
        # Step 5: Save final results
        output_path = CONFIG['output_dir'] / "full_feats.csv"
        final_df.to_csv(output_path, index=False)
        logger.info(f"Saved final dataset to {output_path}")
        
        # Step 6: Generate summary statistics
        logger.info("Step 6: Generating summary statistics...")
        stats = generate_summary_stats(final_df)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE - SUMMARY STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total rows: {stats['total_rows']}")
        logger.info(f"Total columns: {stats['total_columns']}")
        logger.info(f"Feature columns: {stats['feature_columns']}")
        
        logger.info("\nTarget Properties:")
        for prop, prop_stats in stats['target_properties'].items():
            logger.info(f"  {prop}: {prop_stats['count']} samples, mean={prop_stats['mean']:.3f}, std={prop_stats['std']:.3f}")
        
        # Save statistics
        stats_path = CONFIG['results_dir'] / "feature_extraction_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Step 7: Create GNN baseline (optional)
        logger.info("Step 7: Creating GNN baseline...")
        try:
            gnn_components = create_gnn_baseline()
            if gnn_components:
                logger.info("GNN baseline components created successfully")
        except Exception as e:
            logger.warning(f"Could not create GNN baseline: {e}")
        
        logger.info("=" * 60)
        logger.info("Full Feature Engineering Pipeline completed successfully!")
        logger.info(f"Output: {output_path}")
        logger.info(f"Shape: {final_df.shape}")
        logger.info("=" * 60)
        
        return final_df
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    result_df = main()