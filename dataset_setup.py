#!/usr/bin/env python3
"""
Polymer GNN Dataset Setup Script - Real Dataset Version

This script processes the real polymer dataset from train.csv for property prediction.
Handles polymer repeat units with connection points (*) and multiple target properties.

Features:
- Multi-property polymer dataset processing
- Polymer SMILES preprocessing (connection points)
- Data quality assessment and filtering
- Train/validation/test splits
- Support for multiple target properties (Tg, FFV, Tc, Density, Rg)

Usage:
    python dataset_setup.py --target_property Tg
    python dataset_setup.py --target_property FFV
    python dataset_setup.py --multi_target  # Use all available properties
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data import MolecularGraphConverter, PolymerTgDataset
from torch_geometric.data import DataLoader


def setup_logging(log_level='INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dataset_setup.log')
        ]
    )


def analyze_dataset(csv_file: str):
    """Analyze the polymer dataset structure and quality."""
    print("🔍 ANALYZING POLYMER DATASET")
    print("="*60)
    
    df = pd.read_csv(csv_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Check missing values
    print("Missing values per column:")
    property_coverage = {}
    for col in df.columns:
        missing = df[col].isnull().sum()
        missing_pct = (missing / len(df)) * 100
        available = len(df) - missing
        property_coverage[col] = available
        print(f"  {col}: {missing:,} missing ({missing_pct:.1f}%) - {available:,} available")
    
    # Focus on target properties
    target_properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    available_targets = {prop: property_coverage.get(prop, 0) for prop in target_properties}
    
    print(f"\nTarget Properties Summary:")
    for prop, count in available_targets.items():
        pct = (count / len(df)) * 100
        print(f"  {prop}: {count:,} samples ({pct:.1f}%)")
    
    # Analyze SMILES
    print(f"\nSMILES Analysis:")
    smiles_lengths = df['SMILES'].str.len()
    print(f"  Average length: {smiles_lengths.mean():.1f}")
    print(f"  Length range: {smiles_lengths.min()} to {smiles_lengths.max()}")
    
    # Check for polymer-specific patterns
    with_star = df['SMILES'].str.contains(r'\*', na=False).sum()
    print(f"  With * (connection points): {with_star:,} ({with_star/len(df)*100:.1f}%)")
    
    return df, available_targets


def preprocess_polymer_smiles(smiles: str) -> str:
    """
    Preprocess polymer SMILES strings.
    Handle connection points (*) and other polymer-specific notation.
    """
    if pd.isna(smiles):
        return smiles
    
    # For now, remove connection points (*)
    # In a more sophisticated approach, we might want to cap them with hydrogens
    # or handle them as special connection features
    processed = smiles.replace('*', '')
    
    # Remove any empty parentheses that might result
    processed = processed.replace('()', '')
    
    return processed


def create_filtered_dataset(df: pd.DataFrame, target_property: str, min_samples: int = 50):
    """Create a filtered dataset with valid samples for the target property."""
    print(f"\n📊 CREATING FILTERED DATASET FOR {target_property}")
    print("-" * 50)
    
    # Filter for samples with the target property
    valid_mask = df[target_property].notna()
    filtered_df = df[valid_mask].copy()
    
    print(f"Samples with {target_property}: {len(filtered_df):,}")
    
    if len(filtered_df) < min_samples:
        print(f"⚠️  Warning: Only {len(filtered_df)} samples available for {target_property}")
        print(f"   Minimum recommended: {min_samples}")
    
    # Preprocess SMILES
    print("Processing polymer SMILES...")
    filtered_df['processed_smiles'] = filtered_df['SMILES'].apply(preprocess_polymer_smiles)
    
    # Check for any invalid SMILES after preprocessing
    invalid_smiles = filtered_df['processed_smiles'].str.len() < 2
    if invalid_smiles.any():
        print(f"⚠️  Removing {invalid_smiles.sum()} samples with invalid processed SMILES")
        filtered_df = filtered_df[~invalid_smiles]
    
    # Save filtered dataset
    output_file = f"data/processed/filtered_{target_property.lower()}_dataset.csv"
    filtered_df.to_csv(output_file, index=False)
    print(f"✅ Filtered dataset saved to: {output_file}")
    
    # Print statistics
    target_values = filtered_df[target_property]
    print(f"\n{target_property} Statistics:")
    print(f"  Count: {len(target_values):,}")
    print(f"  Mean: {target_values.mean():.2f}")
    print(f"  Std: {target_values.std():.2f}")
    print(f"  Min: {target_values.min():.2f}")
    print(f"  Max: {target_values.max():.2f}")
    
    return filtered_df, output_file


def test_molecular_graph_converter(df: pd.DataFrame, max_test_samples: int = 5):
    """Test the molecular graph converter with polymer SMILES."""
    print("\n🧪 TESTING MOLECULAR GRAPH CONVERTER")
    print("="*60)
    
    converter = MolecularGraphConverter(
        max_atoms=300,  # Increased for larger polymer structures
        include_hydrogens=False,
        use_chirality=True,
        use_bond_types=True
    )
    
    # Test with a few sample SMILES
    test_smiles = df['processed_smiles'].dropna().head(max_test_samples)
    
    successful_conversions = 0
    
    for i, smiles in enumerate(test_smiles):
        print(f"\n--- Testing SMILES {i+1}: {smiles[:50]}{'...' if len(smiles) > 50 else ''} ---")
        
        graph = converter.smiles_to_graph(smiles)
        
        if graph is not None:
            successful_conversions += 1
            print(f"✅ Conversion successful")
            print(f"  Nodes: {graph.num_nodes}")
            print(f"  Edges: {graph.edge_index.shape[1]}")
            print(f"  Node feature dim: {graph.x.shape[1]}")
            print(f"  Molecular features: {graph.mol_features.shape[0]}")
            if graph.edge_attr is not None:
                print(f"  Edge feature dim: {graph.edge_attr.shape[1]}")
        else:
            print(f"❌ Conversion failed")
    
    success_rate = successful_conversions / len(test_smiles)
    print(f"\n📊 Conversion Success Rate: {success_rate:.1%} ({successful_conversions}/{len(test_smiles)})")
    
    return success_rate > 0.8  # Return True if most conversions succeed


def test_polymer_dataset(csv_file: str, target_property: str = 'Tg'):
    """Test the polymer dataset with the real data."""
    print(f"\n🧬 TESTING POLYMER DATASET - {target_property}")
    print("="*60)
    
    # Create dataset
    dataset = PolymerTgDataset(
        root='./data/processed',
        csv_file=csv_file,
        smiles_column='processed_smiles',
        target_column=target_property,
        split_type='all',
        split_ratios=(0.7, 0.15, 0.15),
        random_state=42
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    if len(dataset) > 0:
        # Print quality summary
        dataset.print_quality_summary()
        
        # Test data loading
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  SMILES: {sample.smiles}")
        print(f"  {target_property}: {sample.y.item():.2f}")
        print(f"  Nodes: {sample.num_nodes}")
        print(f"  Edges: {sample.edge_index.shape[1]}")
        
        # Test DataLoader
        print("\n🔄 Testing DataLoader...")
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print(f"  Batch created successfully")
        print(f"  Batch size: {batch.batch.max().item() + 1}")
        print(f"  Total nodes: {batch.x.shape[0]}")
        print(f"  Total edges: {batch.edge_index.shape[1]}")
        
        return True
    else:
        print("❌ No samples in dataset")
        return False


def main():
    """Main function to run polymer dataset setup."""
    parser = argparse.ArgumentParser(description='Real Polymer Dataset Setup')
    parser.add_argument('--csv_file', type=str, default='data/raw/train.csv',
                       help='Path to polymer dataset CSV file')
    parser.add_argument('--target_property', type=str, default='Tg',
                       choices=['Tg', 'FFV', 'Tc', 'Density', 'Rg'],
                       help='Target property for prediction')
    parser.add_argument('--multi_target', action='store_true',
                       help='Setup for multi-target learning')
    parser.add_argument('--min_samples', type=int, default=50,
                       help='Minimum samples required for a target property')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("🚀 REAL POLYMER DATASET SETUP")
    print("="*60)
    
    # Create directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Check if CSV file exists
    if not Path(args.csv_file).exists():
        logger.error(f"CSV file not found: {args.csv_file}")
        return
    
    try:
        # Analyze dataset
        df, available_targets = analyze_dataset(args.csv_file)
        
        # Determine target properties to process
        if args.multi_target:
            target_properties = [prop for prop, count in available_targets.items() 
                               if count >= args.min_samples and prop in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']]
            print(f"\n🎯 Multi-target mode: {target_properties}")
        else:
            target_properties = [args.target_property]
            print(f"\n🎯 Single target mode: {args.target_property}")
        
        # Process each target property
        for target_prop in target_properties:
            if available_targets.get(target_prop, 0) < args.min_samples:
                print(f"⚠️  Skipping {target_prop}: only {available_targets.get(target_prop, 0)} samples available")
                continue
            
            # Create filtered dataset
            filtered_df, filtered_file = create_filtered_dataset(df, target_prop, args.min_samples)
            
            # Test molecular graph converter
            if test_molecular_graph_converter(filtered_df):
                # Test polymer dataset
                if test_polymer_dataset(filtered_file, target_prop):
                    print(f"✅ Successfully set up dataset for {target_prop}")
                else:
                    print(f"❌ Failed to set up dataset for {target_prop}")
            else:
                print(f"❌ Graph conversion failed for {target_prop} - check SMILES preprocessing")
        
        print("\n" + "="*60)
        print("✅ POLYMER DATASET SETUP COMPLETED!")
        print("="*60)
        
        print(f"\nProcessed {len(target_properties)} target properties")
        print("Next steps:")
        print("1. Train GNN models on the processed datasets")
        print("2. Experiment with different polymer SMILES representations")
        print("3. Consider multi-target learning approaches")
        print("4. Validate on additional polymer datasets")
        
    except Exception as e:
        logger.error(f"Error during dataset setup: {e}")
        raise


if __name__ == "__main__":
    main() 