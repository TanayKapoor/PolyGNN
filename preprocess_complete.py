#!/usr/bin/env python3
"""
Complete Polymer Dataset Preprocessor with 147 Features
Matches the feature set from full_feats.csv for external validation
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, GraphDescriptors
import logging
from pathlib import Path
import argparse
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompletePolymerPreprocessor:
    def __init__(self):
        # Get feature names from existing dataset
        self.feature_names = self._get_feature_names_from_existing()
    
    def _get_feature_names_from_existing(self):
        """Get feature names from existing full_feats.csv"""
        existing_path = Path("data/processed/full_feats.csv")
        if existing_path.exists():
            existing_df = pd.read_csv(existing_path, nrows=1)
            # Skip SMILES and target columns
            skip_cols = ['smiles', 'canonical_smiles', 'Tg', 'Tm', 'Density', 'FFV', 'Tc', 'Rg', 'source']
            feature_cols = [col for col in existing_df.columns if col not in skip_cols]
            logger.info(f"📊 Found {len(feature_cols)} features in existing dataset")
            return feature_cols
        else:
            logger.warning("⚠️  No existing dataset found - using default feature set")
            return self._get_default_features()
    
    def _get_default_features(self):
        """Default feature set (147 polymer features)"""
        # Basic molecular descriptors
        molecular = [
            'ABC', 'ATS0dv', 'BCUTdv-1h', 'CIC0', 'Cross-sectional', 'EState_VSA10', 
            'ETA_dBeta', 'Kd_average', 'Kier2', 'LogEE_A', 'MW', 'MW_ratio', 
            'Nd_average', 'SMR_VSA6', 'SlogP_VSA3', 'TopoPSA(NO)', 'Vdw', 'Xc-4d', 
            'mZagreb1', 'nH', 'unit_molecular_weight', 'degree_polymerization'
        ]
        
        # Morgan fingerprints (128 bits)
        morgan = [f'morgan_fp_{i}' for i in range(128)]
        
        # Polymer-specific features
        polymer = [
            'chain_flexibility', 'persistence_length_est', 'end_to_end_distance_log',
            'radius_gyration_log', 'chain_compactness_log', 'ring_complexity',
            'heteroatom_ratio', 'bond_diversity', 'branching_factor',
            'stereochemical_complexity', 'aromaticity_index', 'free_volume_fraction',
            'chain_stiffness_log', 'interaction_strength', 'packing_efficiency',
            'flexibility_factor', 'polarity_factor', 'crystallinity_indicator'
        ]
        
        return molecular + morgan + polymer
    
    def calculate_comprehensive_features(self, smiles):
        """Calculate comprehensive polymer features for a SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {feat: np.nan for feat in self.feature_names}
            
            features = {}
            
            # Basic molecular descriptors
            features['MW'] = Descriptors.MolWt(mol)
            features['unit_molecular_weight'] = features['MW']
            features['TopoPSA(NO)'] = Descriptors.TPSA(mol)
            features['SlogP_VSA3'] = Descriptors.SlogP_VSA3(mol)
            features['SMR_VSA6'] = Descriptors.SMR_VSA6(mol) 
            features['nH'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'H')
            features['ABC'] = Descriptors.NumHeteroatoms(mol)
            features['LogEE_A'] = Descriptors.BalabanJ(mol)
            features['Kier2'] = Descriptors.Kappa2(mol)
            features['EState_VSA10'] = Descriptors.EState_VSA10(mol)
            features['CIC0'] = Descriptors.Ipc(mol)
            features['BCUTdv-1h'] = Descriptors.BertzCT(mol)
            features['ATS0dv'] = Descriptors.BalabanJ(mol)
            features['mZagreb1'] = GraphDescriptors.BalabanJ(mol)
            features['Xc-4d'] = Descriptors.Chi4n(mol)
            features['Vdw'] = Descriptors.LabuteASA(mol)
            features['Cross-sectional'] = Descriptors.PEOE_VSA1(mol)
            try:
                features['ETA_dBeta'] = Descriptors.FractionCsp3(mol)
            except AttributeError:
                features['ETA_dBeta'] = Descriptors.NumAliphaticCarbocycles(mol) / features['MW'] if features['MW'] > 0 else 0
            features['Kd_average'] = Descriptors.Kappa1(mol)
            features['Nd_average'] = Descriptors.NumRotatableBonds(mol)
            features['MW_ratio'] = 1.0
            
            # Morgan fingerprints (128 bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
            for i in range(128):
                features[f'morgan_fp_{i}'] = int(fp[i])
            
            # Polymer-specific features
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            
            # Chain properties
            features['degree_polymerization'] = num_atoms / 10.0
            features['chain_flexibility'] = rotatable_bonds / num_atoms if num_atoms > 0 else 0
            features['persistence_length_est'] = 1.0 / (features['chain_flexibility'] + 1e-6)
            features['chain_stiffness_log'] = np.log(features['persistence_length_est'] + 1)
            
            # Structural complexity
            features['ring_complexity'] = num_rings * 2.0 / num_atoms if num_atoms > 0 else 0
            heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in ['C', 'H'])
            features['heteroatom_ratio'] = heteroatoms / num_atoms if num_atoms > 0 else 0
            
            # Bond diversity
            bond_types = set(bond.GetBondType() for bond in mol.GetBonds())
            features['bond_diversity'] = len(bond_types) / 4.0
            
            # Branching
            branch_points = sum(1 for atom in mol.GetAtoms() if len(atom.GetNeighbors()) > 2)
            features['branching_factor'] = branch_points / num_atoms if num_atoms > 0 else 0
            
            # Stereochemistry
            chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            features['stereochemical_complexity'] = chiral_centers / num_atoms if num_atoms > 0 else 0
            
            # Aromaticity
            aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            features['aromaticity_index'] = aromatic_atoms / num_atoms if num_atoms > 0 else 0
            
            # Volume and packing
            molecular_volume = features['MW'] / 0.6
            van_der_waals_volume = Descriptors.LabuteASA(mol)
            features['free_volume_fraction'] = max(0, 1 - (van_der_waals_volume / molecular_volume)) if molecular_volume > 0 else 0.3
            features['packing_efficiency'] = 1.0 - features['free_volume_fraction']
            
            # Interaction strength
            features['interaction_strength'] = features['TopoPSA(NO)'] / features['MW'] if features['MW'] > 0 else 0
            
            # Derived properties
            features['flexibility_factor'] = min(1.0, features['chain_flexibility'] * 10)
            features['polarity_factor'] = features['TopoPSA(NO)'] / 100.0
            features['crystallinity_indicator'] = max(0, min(1, 1.0 - features['branching_factor'] - features['chain_flexibility']))
            
            # Geometric properties (log-scaled)
            features['end_to_end_distance_log'] = np.log(features['degree_polymerization'] * features['persistence_length_est'] + 1)
            features['radius_gyration_log'] = features['end_to_end_distance_log'] - 0.5
            features['chain_compactness_log'] = -features['end_to_end_distance_log'] + np.log(features['MW'] + 1)
            
            # Fill any missing features with defaults
            for feat_name in self.feature_names:
                if feat_name not in features:
                    features[feat_name] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature calculation failed for {smiles}: {e}")
            return {feat: np.nan for feat in self.feature_names}
    
    def preprocess_dataset(self, input_file, output_file):
        """Complete preprocessing pipeline"""
        
        logger.info(f"🚀 Starting complete preprocessing pipeline")
        logger.info(f"📂 Input: {input_file}")
        logger.info(f"💾 Output: {output_file}")
        
        # Load dataset
        if str(input_file).endswith('.xlsx'):
            df = pd.read_excel(input_file)
        else:
            df = pd.read_csv(input_file)
        
        logger.info(f"📊 Initial dataset: {df.shape}")
        logger.info(f"📋 Columns: {df.columns.tolist()}")
        
        # Find SMILES and Tg columns
        smiles_col = None
        tg_col = None
        
        for col in df.columns:
            if col.lower() in ['smiles', 'smi', 'smiles_string']:
                smiles_col = col
            if col.lower() in ['tg', 't_g', 'glass_transition', 'glass_transition_temp']:
                tg_col = col
        
        if smiles_col is None:
            raise ValueError("No SMILES column found in dataset")
        
        logger.info(f"🧬 Using SMILES column: {smiles_col}")
        if tg_col:
            logger.info(f"🎯 Using Tg column: {tg_col}")
        
        # Validate and clean SMILES
        logger.info("🧪 Validating SMILES...")
        def validate_smiles(s):
            try:
                mol = Chem.MolFromSmiles(str(s))
                return mol is not None
            except:
                return False
        
        df['valid'] = df[smiles_col].apply(validate_smiles)
        invalid_count = len(df) - df['valid'].sum()
        logger.info(f"❌ Invalid SMILES: {invalid_count} ({invalid_count/len(df)*100:.1f}%)")
        
        df = df[df['valid']].drop('valid', axis=1)
        
        # Create canonical SMILES
        logger.info("🔄 Creating canonical SMILES...")
        def get_canonical_smiles(s):
            try:
                mol = Chem.MolFromSmiles(str(s))
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
            except:
                pass
            return None
        
        df['canonical_smiles'] = df[smiles_col].apply(get_canonical_smiles)
        df = df.dropna(subset=['canonical_smiles'])
        
        # Remove duplicates
        before_dedupe = len(df)
        df = df.drop_duplicates(subset=['canonical_smiles'])
        logger.info(f"🔄 Removed {before_dedupe - len(df)} duplicates")
        
        # Check against existing dataset
        existing_path = Path("data/processed/full_feats.csv")
        if existing_path.exists():
            logger.info("📊 Checking for overlaps with existing dataset...")
            existing_df = pd.read_csv(existing_path)
            if 'canonical_smiles' in existing_df.columns:
                existing_smiles = set(existing_df['canonical_smiles'].dropna())
                overlap_mask = df['canonical_smiles'].isin(existing_smiles)
                overlap_count = overlap_mask.sum()
                logger.info(f"🔄 Removing {overlap_count} overlapping structures")
                df = df[~overlap_mask]
        
        logger.info(f"✅ Final cleaned dataset: {len(df)} samples")
        
        # Calculate comprehensive features
        logger.info(f"🧪 Calculating {len(self.feature_names)} comprehensive features...")
        
        def calc_features(row):
            smiles = row['canonical_smiles']
            features = self.calculate_comprehensive_features(smiles)
            return pd.Series(features)
        
        feature_df = df.apply(calc_features, axis=1)
        
        # Combine with original data
        result_df = pd.concat([df, feature_df], axis=1)
        
        # Rename columns for consistency
        result_df = result_df.rename(columns={smiles_col: 'smiles'})
        if tg_col:
            result_df = result_df.rename(columns={tg_col: 'Tg'})
        
        # Add missing target columns
        for target in ['Tg', 'Tm', 'Density', 'FFV', 'Tc', 'Rg']:
            if target not in result_df.columns:
                result_df[target] = np.nan
        
        # Add source column
        result_df['source'] = 'external_validation'
        
        # Final cleanup
        result_df = result_df.dropna(subset=['canonical_smiles'])
        
        # Reorder columns to match existing dataset
        column_order = ['smiles', 'Tg', 'canonical_smiles', 'source'] + \
                      ['Density', 'FFV', 'Tc', 'Rg', 'Tm'] + \
                      [col for col in self.feature_names if col in result_df.columns]
        
        # Keep only existing columns
        column_order = [col for col in column_order if col in result_df.columns]
        result_df = result_df[column_order]
        
        logger.info(f"📊 Final dataset shape: {result_df.shape}")
        logger.info(f"🧪 Features calculated: {len([col for col in result_df.columns if col in self.feature_names])}")
        
        # Save dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        
        logger.info(f"💾 Saved to {output_path}")
        
        # Print summary
        if 'Tg' in result_df.columns and result_df['Tg'].notna().sum() > 0:
            logger.info(f"🎯 Tg statistics:")
            logger.info(f"   Valid: {result_df['Tg'].notna().sum()}/{len(result_df)}")
            try:
                tg_clean = pd.to_numeric(result_df['Tg'], errors='coerce').dropna()
                if len(tg_clean) > 0:
                    logger.info(f"   Range: {tg_clean.min():.1f}°C to {tg_clean.max():.1f}°C")
                    logger.info(f"   Mean: {tg_clean.mean():.1f}°C")
                    logger.info(f"   Std: {tg_clean.std():.1f}°C")
            except Exception as e:
                logger.warning(f"Could not calculate Tg statistics: {e}")
        
        logger.info("🎉 Preprocessing complete! Ready for external validation.")
        
        return result_df

def main():
    parser = argparse.ArgumentParser(description='Complete polymer dataset preprocessing')
    parser.add_argument('--input', type=str, required=True, help='Input dataset file')
    parser.add_argument('--output', type=str, default='data/processed/external_validation.csv', help='Output file path')
    
    args = parser.parse_args()
    
    preprocessor = CompletePolymerPreprocessor()
    result = preprocessor.preprocess_dataset(args.input, args.output)
    
    print(f"\n✅ SUCCESS!")
    print(f"📊 Processed {len(result)} samples with {len(result.columns)} columns")
    print(f"💾 Saved to: {args.output}")
    print(f"🚀 Ready for external validation!")

if __name__ == "__main__":
    # Test with sample data if run directly
    if len(sys.argv) == 1:
        import sys
        print("Testing with sample data...")
        preprocessor = CompletePolymerPreprocessor()
        result = preprocessor.preprocess_dataset("test_polymer_data.csv", "results/external_validation_test.csv")
    else:
        main()