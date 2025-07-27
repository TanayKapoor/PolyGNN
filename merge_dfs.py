#!/usr/bin/env python3
"""
Merge and Deduplicate Polymer Datasets

This script merges the cleaned polymer datasets, handles deduplication based on 
canonical structures, and adds molecular descriptors where possible.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def is_bigsmiles(structure):
    """Check if structure is BigSMILES notation"""
    if pd.isna(structure):
        return False
    return '{' in str(structure) and (('<' in str(structure) and '>' in str(structure)) or 
                                     ('[' in str(structure) and ']' in str(structure)))

def smart_deduplicate(group):
    """
    Smart deduplication: prefer rows with more non-null values,
    then prefer rows from datasets with more properties
    """
    # Count non-null values per row
    non_null_counts = group.isnull().sum(axis=1)
    
    # Get row with minimum null values (most complete)
    best_idx = non_null_counts.idxmin()
    
    return group.loc[best_idx]

def calculate_molecular_descriptors(structure):
    """
    Calculate molecular descriptors for regular SMILES
    Returns None for BigSMILES structures
    """
    if pd.isna(structure) or is_bigsmiles(structure):
        return pd.Series(dtype=float)
    
    try:
        mol = Chem.MolFromSmiles(structure)
        if mol is None:
            return pd.Series(dtype=float)
        
        # Calculate descriptors
        calc = Calculator(descriptors, ignore_3D=True)
        desc_dict = calc(mol).fill_missing().asdict()
        
        return pd.Series(desc_dict)
    
    except Exception as e:
        print(f"Error calculating descriptors for {structure}: {e}")
        return pd.Series(dtype=float)

def main():
    """Main merge and deduplication pipeline"""
    print("POLYMER DATASET MERGE & DEDUPLICATION")
    print("=" * 50)
    
    # Load cleaned datasets
    processed_dir = Path('data/processed')
    
    print("Loading cleaned datasets...")
    bic = pd.read_csv(processed_dir / 'bic_clean.csv')
    jcim = pd.read_csv(processed_dir / 'jcim_clean.csv')
    train = pd.read_csv(processed_dir / 'train_clean.csv')
    sample = pd.read_csv(processed_dir / 'sample_clean.csv')
    
    # Add source column to track origin
    bic['source'] = 'Bicerano'
    jcim['source'] = 'JCIM'
    train['source'] = 'Train'
    sample['source'] = 'Sample'
    
    print(f"Initial counts: Bicerano={len(bic)}, JCIM={len(jcim)}, Train={len(train)}, Sample={len(sample)}")
    
    # Concatenate all datasets
    print("\nConcatenating datasets...")
    merged = pd.concat([bic, jcim, train, sample], ignore_index=True)
    print(f"Total rows before deduplication: {len(merged)}")
    
    # Check for duplicates
    duplicates = merged['canonical'].duplicated()
    print(f"Found {duplicates.sum()} duplicate canonical structures")
    
    # Deduplicate based on canonical structure
    print("\nDeduplicating by canonical structure...")
    deduplicated = merged.groupby('canonical').apply(smart_deduplicate).reset_index(drop=True)
    print(f"Rows after deduplication: {len(deduplicated)}")
    
    # Report deduplication stats
    unique_originals = len(merged['canonical'].unique())
    print(f"Unique canonical structures: {unique_originals}")
    
    # Source distribution after deduplication
    print("\nSource distribution after deduplication:")
    print(deduplicated['source'].value_counts())
    
    # Property completeness analysis
    print("\nProperty completeness:")
    property_cols = ['Tg', 'Density', 'FFV', 'Tc', 'Rg', 'Tm']
    for col in property_cols:
        if col in deduplicated.columns:
            missing = deduplicated[col].isna().sum()
            total = len(deduplicated)
            print(f"  {col}: {total-missing}/{total} ({100*(total-missing)/total:.1f}%) complete")
    
    # Impute missing values for properties
    print("\nImputing missing property values...")
    for col in property_cols:
        if col in deduplicated.columns:
            missing_before = deduplicated[col].isna().sum()
            if missing_before > 0:
                mean_val = deduplicated[col].mean()
                deduplicated[col] = deduplicated[col].fillna(mean_val)
                print(f"  Filled {missing_before} missing {col} values with mean: {mean_val:.2f}")
    
    # Structure type analysis
    bigsmiles_count = deduplicated['structure'].apply(is_bigsmiles).sum()
    regular_smiles_count = len(deduplicated) - bigsmiles_count
    print(f"\nStructure types: {regular_smiles_count} regular SMILES, {bigsmiles_count} BigSMILES")
    
    # Calculate molecular descriptors for regular SMILES only
    if regular_smiles_count > 0:
        print(f"\nCalculating molecular descriptors for {regular_smiles_count} regular SMILES structures...")
        
        # Only process regular SMILES
        regular_mask = ~deduplicated['structure'].apply(is_bigsmiles)
        regular_structures = deduplicated[regular_mask]
        
        if len(regular_structures) > 0:
            print("  Computing descriptors (this may take a few minutes)...")
            descriptors_df = regular_structures['structure'].apply(calculate_molecular_descriptors)
            
            # Add descriptors to the main dataframe
            if not descriptors_df.empty and len(descriptors_df.columns) > 0:
                # Align descriptors with original dataframe
                full_descriptors = pd.DataFrame(index=deduplicated.index, columns=descriptors_df.columns)
                full_descriptors.loc[regular_mask] = descriptors_df.values
                
                # Concatenate with main dataframe
                final_merged = pd.concat([deduplicated, full_descriptors], axis=1)
                
                print(f"  Added {len(descriptors_df.columns)} molecular descriptors")
                print(f"  Descriptors calculated for {(~full_descriptors.isnull().all(axis=1)).sum()} structures")
            else:
                print("  Warning: No descriptors calculated")
                final_merged = deduplicated
        else:
            print("  No regular SMILES structures found for descriptor calculation")
            final_merged = deduplicated
    else:
        print("\nSkipping molecular descriptor calculation (no regular SMILES structures)")
        final_merged = deduplicated
    
    # Final statistics
    print(f"\nFinal merged dataset:")
    print(f"  Rows: {len(final_merged)}")
    print(f"  Columns: {len(final_merged.columns)}")
    print(f"  Memory usage: {final_merged.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Save merged dataset
    output_path = processed_dir / 'merged_polymer.csv'
    final_merged.to_csv(output_path, index=False)
    print(f"\nSaved merged dataset to: {output_path}")
    
    # Final summary statistics
    print(f"\nTg statistics (°C):")
    tg_stats = final_merged['Tg'].describe()
    print(f"  Mean: {tg_stats['mean']:.1f}°C")
    print(f"  Std: {tg_stats['std']:.1f}°C") 
    print(f"  Range: {tg_stats['min']:.1f}°C to {tg_stats['max']:.1f}°C")
    
    # Check for other properties
    other_props = ['Density', 'FFV', 'Tc', 'Rg', 'Tm']
    available_props = [col for col in other_props if col in final_merged.columns and final_merged[col].notna().sum() > 0]
    if available_props:
        print(f"\nOther available properties: {', '.join(available_props)}")
    
    print(f"\n{'='*50}")
    print("MERGE COMPLETE!")
    print(f"{'='*50}")
    
    return final_merged

if __name__ == "__main__":
    merged_data = main()