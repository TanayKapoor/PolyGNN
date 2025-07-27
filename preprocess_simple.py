#!/usr/bin/env python3
"""
Simple Polymer Dataset Preprocessor
Creates external validation dataset with polymer features
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_polymer_dataset(input_file, output_file):
    """Simple preprocessing pipeline"""
    
    logger.info(f"🔬 Loading dataset from {input_file}")
    
    # Load dataset
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    
    logger.info(f"📊 Initial shape: {df.shape}")
    logger.info(f"📋 Columns: {df.columns.tolist()}")
    
    # Find SMILES and Tg columns
    smiles_col = None
    tg_col = None
    
    for col in df.columns:
        if col.lower() in ['smiles', 'smi']:
            smiles_col = col
        if col.lower() in ['tg', 't_g']:
            tg_col = col
    
    if smiles_col is None:
        raise ValueError("No SMILES column found")
    
    logger.info(f"🧬 Using SMILES column: {smiles_col}")
    if tg_col:
        logger.info(f"🎯 Using Tg column: {tg_col}")
    
    # Validate SMILES
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
    def get_canonical(s):
        try:
            mol = Chem.MolFromSmiles(str(s))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return None
    
    df['canonical_smiles'] = df[smiles_col].apply(get_canonical)
    df = df.dropna(subset=['canonical_smiles'])
    df = df.drop_duplicates(subset=['canonical_smiles'])
    
    logger.info(f"✅ After cleaning: {len(df)} samples")
    
    # Calculate basic polymer features
    def calc_features(row):
        smiles = row['canonical_smiles']
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return pd.Series({})
            
            features = {}
            
            # Basic descriptors
            features['MW'] = Descriptors.MolWt(mol)
            features['TPSA'] = Descriptors.TPSA(mol)
            features['LogP'] = Descriptors.MolLogP(mol)
            features['NumAtoms'] = mol.GetNumAtoms()
            features['NumBonds'] = mol.GetNumBonds()
            features['NumRotatableBonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
            features['NumRings'] = rdMolDescriptors.CalcNumRings(mol)
            
            # Polymer-specific
            features['chain_flexibility'] = features['NumRotatableBonds'] / features['NumAtoms'] if features['NumAtoms'] > 0 else 0
            features['degree_polymerization'] = features['NumAtoms'] / 10.0
            features['ring_complexity'] = features['NumRings'] / features['NumAtoms'] if features['NumAtoms'] > 0 else 0
            
            # Morgan fingerprints (first 20 bits for demo)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=20)
            for i in range(20):
                features[f'morgan_fp_{i}'] = int(fp[i])
            
            return pd.Series(features)
            
        except Exception as e:
            logger.warning(f"Feature calculation failed for {smiles}: {e}")
            return pd.Series({})
    
    logger.info("🧪 Calculating features...")
    feature_df = df.apply(calc_features, axis=1)
    
    # Combine with original data
    result_df = pd.concat([df, feature_df], axis=1)
    
    # Rename columns for consistency
    result_df = result_df.rename(columns={smiles_col: 'SMILES'})
    if tg_col:
        result_df = result_df.rename(columns={tg_col: 'Tg'})
    
    # Add missing target columns
    for target in ['Tg', 'Tm', 'Density', 'FFV']:
        if target not in result_df.columns:
            result_df[target] = np.nan
    
    # Clean up final dataset
    result_df = result_df.dropna(subset=['canonical_smiles'])
    
    logger.info(f"📊 Final shape: {result_df.shape}")
    
    # Save
    result_df.to_csv(output_file, index=False)
    logger.info(f"💾 Saved to {output_file}")
    
    return result_df

if __name__ == "__main__":
    # Test with sample data
    df = preprocess_polymer_dataset("test_polymer_data.csv", "results/test_preprocessing/processed.csv")
    print("\n✅ Preprocessing complete!")
    print(f"📊 Final dataset: {len(df)} samples, {len(df.columns)} columns")
    print(f"🧪 Sample features: {[col for col in df.columns if 'morgan_fp' in col][:5]}")