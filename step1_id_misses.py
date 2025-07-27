#!/usr/bin/env python3
"""
Step 1: ID Misses - Identify missing features in merged_polymer.csv

This script loads the merged polymer dataset and identifies which of the expected 
147 features are missing from the current columns.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path to import polymer features
sys.path.append(str(Path(__file__).parent / 'src'))

from features.polymer_features import PolymerFeatureExtractor


def load_merged_data():
    """Load the merged polymer dataset"""
    # Fix pandas compatibility for any pickle files that might be loaded
    if 'pandas.core.indexes.numeric' not in sys.modules:
        sys.modules['pandas.core.indexes.numeric'] = pd.core.indexes.base
    
    data_path = Path(__file__).parent / 'data' / 'processed' / 'merged_polymer.csv'
    
    if not data_path.exists():
        print(f"Error: {data_path} does not exist!")
        return None
    
    print(f"Loading merged data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset shape: {df.shape}")
    
    return df


def get_expected_147_features():
    """Get the expected 147 feature names from PolymerFeatureExtractor"""
    # Create extractor with all features enabled (default config gives 147 features)
    extractor = PolymerFeatureExtractor(
        fingerprint_size=128,  # Default Morgan fingerprint size
        include_chain_descriptors=True,
        include_complexity=True,
        include_molecular_descriptors=True
    )
    
    feature_names = extractor.get_feature_names()
    feature_dim = extractor.get_feature_dim()
    
    print(f"Expected feature dimension: {feature_dim}")
    print(f"Number of feature names: {len(feature_names)}")
    
    return feature_names


def identify_missing_features(df, expected_features):
    """Identify which expected features are missing from the dataframe"""
    current_cols = df.columns.tolist()
    
    print(f"\nCurrent dataset columns: {len(current_cols)}")
    print(f"Expected features: {len(expected_features)}")
    
    # Find missing features
    misses = [col for col in expected_features if col not in current_cols]
    
    # Find extra features (in dataset but not expected)
    extras = [col for col in current_cols if col not in expected_features and 
              col not in ['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg', 'source']]
    
    return misses, extras


def main():
    """Main function to identify missing features"""
    print("=" * 60)
    print("STEP 1: ID MISSES - Identify Missing Features")
    print("=" * 60)
    
    # Load merged data
    df = load_merged_data()
    if df is None:
        return
    
    # Get expected 147 features
    print("\nGetting expected 147 features...")
    expected_features = get_expected_147_features()
    
    # Identify misses
    print("\nAnalyzing feature gaps...")
    misses, extras = identify_missing_features(df, expected_features)
    
    # Report results
    print(f"\n{'='*60}")
    print("MISSING FEATURES ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\n✅ Features present: {len(expected_features) - len(misses)}")
    print(f"❌ Features missing: {len(misses)}")
    print(f"➕ Extra features: {len(extras)}")
    
    if misses:
        print(f"\n🚨 MISSING FEATURES ({len(misses)}):")
        for i, miss in enumerate(misses, 1):
            print(f"  {i:2d}. {miss}")
    else:
        print("\n🎉 All expected features are present!")
    
    if extras:
        print(f"\n💡 EXTRA FEATURES ({len(extras)}):")
        for i, extra in enumerate(extras, 1):
            print(f"  {i:2d}. {extra}")
    
    # Show sample of current columns
    print(f"\n📋 SAMPLE OF CURRENT COLUMNS (first 20):")
    feature_cols = [col for col in df.columns if col not in ['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg', 'source']]
    for i, col in enumerate(feature_cols[:20], 1):
        print(f"  {i:2d}. {col}")
    if len(feature_cols) > 20:
        print(f"  ... and {len(feature_cols) - 20} more")
    
    print(f"\n{'='*60}")
    print("STEP 1 COMPLETE")
    print(f"{'='*60}")
    
    # Return for potential use in next steps
    return {
        'dataframe': df,
        'expected_features': expected_features,
        'missing_features': misses,
        'extra_features': extras,
        'current_feature_count': len(feature_cols)
    }


if __name__ == "__main__":
    result = main()