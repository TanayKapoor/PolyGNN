#!/usr/bin/env python3
"""
Data Standardization and Cleaning Pipeline

Load & Standardize all polymer datasets:
- Convert Tg units to °C
- Rename columns for merge consistency  
- Validate structures with RDKit
- Create canonical SMILES for dedupe
- Clean missing values and outliers
- Save processed datasets
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from pathlib import Path
import os

def validate_canonical(df):
    """
    Validate structures with RDKit and create canonical SMILES
    """
    print(f"  Validating {len(df)} structures...")
    
    def is_bigsmiles(s):
        """Check if a structure string is BigSMILES notation"""
        if pd.isna(s):
            return False
        return '{' in s and (('<' in s and '>' in s) or ('[' in s and ']' in s))
    
    def validate_structure(s):
        """Validate structure - handle both SMILES and BigSMILES"""
        if pd.isna(s):
            return False
        
        # If it's BigSMILES, just check it's not empty and has proper braces
        if is_bigsmiles(s):
            return len(s.strip()) > 0 and s.count('{') == s.count('}')
        
        # For regular SMILES, use RDKit
        try:
            return Chem.MolFromSmiles(s) is not None
        except:
            return False
    
    def create_canonical(s):
        """Create canonical representation"""
        if pd.isna(s):
            return None
        
        # For BigSMILES, just clean and normalize whitespace
        if is_bigsmiles(s):
            return s.strip()
        
        # For regular SMILES, use RDKit canonicalization
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            return None
        except:
            return None
    
    # Check structure validity
    df['valid'] = df['structure'].apply(validate_structure)
    initial_count = len(df)
    df = df[df['valid']].drop('valid', axis=1)
    invalid_count = initial_count - len(df)
    
    if invalid_count > 0:
        print(f"  Dropped {invalid_count} invalid structures ({invalid_count/initial_count*100:.1f}%)")
    
    # Create canonical representation
    df['canonical'] = df['structure'].apply(create_canonical)
    
    # Count BigSMILES vs regular SMILES
    bigsmiles_count = df['structure'].apply(is_bigsmiles).sum()
    regular_smiles_count = len(df) - bigsmiles_count
    
    if bigsmiles_count > 0:
        print(f"  Found {bigsmiles_count} BigSMILES and {regular_smiles_count} regular SMILES")
    
    return df

def clean_missing_outliers(df, df_name):
    """
    Clean missing values and outliers
    """
    print(f"  Cleaning {df_name}...")
    initial_count = len(df)
    
    if initial_count == 0:
        print(f"    Dataset is empty - skipping cleaning")
        return df
    
    # Drop rows missing Tg (core property)
    df = df.dropna(subset=['Tg'])
    tg_dropped = initial_count - len(df)
    if tg_dropped > 0:
        print(f"    Dropped {tg_dropped} rows missing Tg")
    
    if len(df) == 0:
        print(f"    No rows remaining after dropping missing Tg values")
        return df
    
    # Fill missing values for other properties with means
    numeric_cols = ['Density', 'FFV', 'Tc', 'Rg', 'Tm']
    for col in numeric_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            mean_val = df[col].mean()
            missing_count = df[col].isna().sum()
            df[col].fillna(mean_val, inplace=True)
            print(f"    Filled {missing_count} missing {col} values with mean: {mean_val:.2f}")
    
    # Remove Tg outliers (reasonable range: -200 to 500°C)
    outlier_mask = (df['Tg'] < -200) | (df['Tg'] > 500)
    outlier_count = outlier_mask.sum()
    if outlier_count > 0:
        print(f"    Removing {outlier_count} Tg outliers (< -200°C or > 500°C)")
        df = df[~outlier_mask]
    
    final_count = len(df)
    if initial_count > 0:
        retention_pct = final_count/initial_count*100
        print(f"    Final count: {final_count} (retained {retention_pct:.1f}%)")
    else:
        print(f"    Final count: {final_count}")
    
    return df

def main():
    """
    Main standardization and cleaning pipeline
    """
    print("POLYMER DATA STANDARDIZATION & CLEANING PIPELINE")
    print("=" * 60)
    
    # Paths
    raw_dir = Path(__file__).parent.parent / 'data' / 'raw'
    processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    print(f"Raw data directory: {raw_dir}")
    print(f"Processed data directory: {processed_dir}")
    
    # =================
    # 1. LOAD & STANDARDIZE
    # =================
    print("\n" + "="*40)
    print("1. LOADING & STANDARDIZING DATASETS")
    print("="*40)
    
    # Bicerano dataset
    print("\nProcessing Bicerano dataset...")
    bic = pd.read_csv(raw_dir / 'Bicerano_bigsmiles.csv', encoding='latin-1')
    print(f"  Original shape: {bic.shape}")
    print(f"  Original columns: {list(bic.columns)}")
    
    # Convert K to C and select/rename columns
    bic['Tg'] = bic['Tg (K) exp'] - 273.15
    bic = bic[['BigSMILES', 'Tg']].rename(columns={'BigSMILES': 'structure'})
    print(f"  After standardization: {bic.shape}")
    
    # JCIM dataset
    print("\nProcessing JCIM dataset...")
    jcim = pd.read_csv(raw_dir / 'JCIM_sup_bigsmiles.csv')
    print(f"  Original shape: {jcim.shape}")
    print(f"  Original columns: {list(jcim.columns)}")
    
    jcim = jcim[['BigSMILES', 'Tg (C)']].rename(columns={'BigSMILES': 'structure', 'Tg (C)': 'Tg'})
    print(f"  After standardization: {jcim.shape}")
    
    # Training dataset
    print("\nProcessing train dataset...")
    train = pd.read_csv(raw_dir / 'train.csv')
    print(f"  Original shape: {train.shape}")
    print(f"  Original columns: {list(train.columns)}")
    
    available_cols = ['SMILES', 'Tg']
    for col in ['Density', 'FFV', 'Tc', 'Rg']:
        if col in train.columns:
            available_cols.append(col)
    
    train = train[available_cols].rename(columns={'SMILES': 'structure'})
    print(f"  After standardization: {train.shape}")
    
    # Sample dataset
    print("\nProcessing sample dataset...")
    sample = pd.read_csv(raw_dir / 'sample_polymer_data.csv')
    print(f"  Original shape: {sample.shape}")
    print(f"  Original columns: {list(sample.columns)}")
    
    if 'glass_transition_temp' in sample.columns:
        sample['Tg'] = sample['glass_transition_temp'].astype(float)
        
    available_cols = ['bigsmiles', 'Tg']
    rename_dict = {'bigsmiles': 'structure'}
    
    if 'density' in sample.columns:
        available_cols.append('density')
        rename_dict['density'] = 'Density'
    if 'melting_temp' in sample.columns:
        available_cols.append('melting_temp')
        rename_dict['melting_temp'] = 'Tm'
        
    sample = sample[available_cols].rename(columns=rename_dict)
    print(f"  After standardization: {sample.shape}")
    
    # =================
    # 2. VALIDATE STRUCTURES
    # =================
    print("\n" + "="*40)
    print("2. VALIDATING STRUCTURES & CREATING CANONICAL SMILES")
    print("="*40)
    
    print("\nBicerano validation:")
    bic = validate_canonical(bic)
    
    print("\nJCIM validation:")
    jcim = validate_canonical(jcim)
    
    print("\nTrain validation:")
    train = validate_canonical(train)
    
    print("\nSample validation:")
    sample = validate_canonical(sample)
    
    print(f"\nPost-validation shapes: bic={bic.shape}, jcim={jcim.shape}, train={train.shape}, sample={sample.shape}")
    
    # =================
    # 3. CLEAN MISSING VALUES & OUTLIERS
    # =================
    print("\n" + "="*40)
    print("3. CLEANING MISSING VALUES & OUTLIERS")
    print("="*40)
    
    bic = clean_missing_outliers(bic, "Bicerano")
    jcim = clean_missing_outliers(jcim, "JCIM")
    train = clean_missing_outliers(train, "Train")
    sample = clean_missing_outliers(sample, "Sample")
    
    # =================
    # 4. SAVE CLEAN DATASETS
    # =================
    print("\n" + "="*40)
    print("4. SAVING CLEAN DATASETS")
    print("="*40)
    
    datasets = {
        'bic_clean.csv': bic,
        'jcim_clean.csv': jcim,
        'train_clean.csv': train,
        'sample_clean.csv': sample
    }
    
    for filename, df in datasets.items():
        filepath = processed_dir / filename
        df.to_csv(filepath, index=False)
        print(f"  Saved {filename}: {df.shape}")
    
    # =================
    # 5. QUICK STATS CHECK
    # =================
    print("\n" + "="*40)
    print("5. QUICK STATS CHECK")
    print("="*40)
    
    for name, df in [('Bicerano', bic), ('JCIM', jcim), ('Train', train), ('Sample', sample)]:
        print(f"\n{name} Dataset Stats:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        if 'Tg' in df.columns:
            tg_stats = df['Tg'].describe()
            print(f"  Tg stats: mean={tg_stats['mean']:.1f}°C, std={tg_stats['std']:.1f}°C")
            print(f"  Tg range: {tg_stats['min']:.1f}°C to {tg_stats['max']:.1f}°C")
        
        # Check for other properties
        numeric_props = ['Density', 'FFV', 'Tc', 'Rg', 'Tm']
        available_props = [col for col in numeric_props if col in df.columns]
        if available_props:
            print(f"  Other properties: {available_props}")
            for prop in available_props:
                prop_stats = df[prop].describe()
                print(f"    {prop}: mean={prop_stats['mean']:.2f}, range={prop_stats['min']:.2f}-{prop_stats['max']:.2f}")
    
    print("\n" + "="*60)
    print("STANDARDIZATION & CLEANING COMPLETE!")
    print("="*60)
    print(f"Clean datasets saved to: {processed_dir}")
    
    # Load and check datasteA.pkl
    print("\n" + "="*40)
    print("6. CHECKING DATASTEA.PKL")
    print("="*40)
    
    try:
        datastea = pd.read_pickle(raw_dir / 'datasteA.pkl')
        print(f"DatasteA shape: {datastea.shape}")
        print(f"DatasteA columns: {list(datastea.columns) if hasattr(datastea, 'columns') else 'Not a DataFrame'}")
        if hasattr(datastea, 'head'):
            print("DatasteA sample:")
            print(datastea.head())
    except Exception as e:
        print(f"Error loading datasteA.pkl: {e}")

if __name__ == "__main__":
    main() 