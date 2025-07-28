"""
Data processing utilities for PolyGNN Showcase application.
Handles SMILES validation, CSV processing, and data formatting.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.rdDepictor import Compute2DCoords
import streamlit as st
from PIL import Image
import io

def validate_smiles(smiles):
    """
    Validate a SMILES string using RDKit.
    
    Args:
        smiles (str): SMILES notation to validate
        
    Returns:
        tuple: (is_valid, message)
    """
    try:
        if not smiles or smiles.strip() == "":
            return False, "Empty SMILES string"
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES notation - cannot parse molecular structure"
        
        # Additional validation for polymer-like structures
        if '*' not in smiles:
            return True, "Valid SMILES (Note: No repeat units '*' detected - may not be a polymer)"
        
        return True, "Valid polymer SMILES with repeat units"
        
    except Exception as e:
        return False, f"Error validating SMILES: {str(e)}"

def process_csv_upload(df):
    """
    Process uploaded CSV file and validate content.
    
    Args:
        df (pd.DataFrame): Uploaded CSV data
        
    Returns:
        tuple: (is_valid, message, processed_df)
    """
    try:
        # Check required columns
        if 'SMILES' not in df.columns:
            return False, "CSV must contain a 'SMILES' column", None
        
        # Check row limit
        if len(df) > 100:
            return False, "CSV contains more than 100 rows. Please limit to 100 rows or fewer.", None
        
        # Remove empty rows
        df = df.dropna(subset=['SMILES'])
        
        if len(df) == 0:
            return False, "No valid SMILES found in the CSV file", None
        
        # Validate all SMILES
        invalid_smiles = []
        valid_indices = []
        
        for idx, smiles in enumerate(df['SMILES']):
            is_valid, _ = validate_smiles(str(smiles))
            if is_valid:
                valid_indices.append(idx)
            else:
                invalid_smiles.append((idx, smiles))
        
        if len(valid_indices) == 0:
            return False, "No valid SMILES found in the CSV file", None
        
        # Filter to valid SMILES only
        processed_df = df.iloc[valid_indices].copy().reset_index(drop=True)
        
        # Validate optional true value columns
        optional_cols = ['Tg_true', 'Tm_true', 'Density_true']
        for col in optional_cols:
            if col in processed_df.columns:
                # Convert to numeric, coerce errors to NaN
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Prepare message
        message = f"Successfully loaded {len(processed_df)} valid polymers"
        if invalid_smiles:
            message += f" (Skipped {len(invalid_smiles)} invalid SMILES)"
        
        return True, message, processed_df
        
    except Exception as e:
        return False, f"Error processing CSV: {str(e)}", None

def calculate_molecular_descriptors(smiles_list):
    """
    Calculate molecular descriptors for a list of SMILES.
    
    Args:
        smiles_list (list): List of SMILES strings
        
    Returns:
        pd.DataFrame: DataFrame with molecular descriptors
    """
    descriptors_data = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                desc = {
                    'SMILES': smiles,
                    'MolWeight': Descriptors.ExactMolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                    'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                    'TPSA': Descriptors.TPSA(mol)
                }
                descriptors_data.append(desc)
            else:
                # Add empty row for invalid SMILES
                desc = {
                    'SMILES': smiles,
                    'MolWeight': np.nan,
                    'LogP': np.nan,
                    'NumRotatableBonds': np.nan,
                    'NumAromaticRings': np.nan,
                    'TPSA': np.nan
                }
                descriptors_data.append(desc)
        except Exception:
            # Add empty row for error cases
            desc = {
                'SMILES': smiles,
                'MolWeight': np.nan,
                'LogP': np.nan,
                'NumRotatableBonds': np.nan,
                'NumAromaticRings': np.nan,
                'TPSA': np.nan
            }
            descriptors_data.append(desc)
    
    return pd.DataFrame(descriptors_data)

def format_prediction_results(predictions, input_data):
    """
    Format prediction results for display.
    
    Args:
        predictions (dict): Dictionary with prediction arrays
        input_data (pd.DataFrame): Input data with SMILES
        
    Returns:
        pd.DataFrame: Formatted results dataframe
    """
    results_df = input_data.copy()
    
    # Add predictions
    results_df['Tg_pred'] = predictions['Tg'].round(2)
    results_df['Tm_pred'] = predictions['Tm'].round(2)
    results_df['Density_pred'] = predictions['Density'].round(3)
    results_df['Uncertainty'] = predictions['unc'].round(3)
    
    # Add confidence intervals
    results_df['Tg_lower'] = (predictions['Tg'] - predictions['unc']).round(2)
    results_df['Tg_upper'] = (predictions['Tg'] + predictions['unc']).round(2)
    
    return results_df

@st.cache_data
def render_smiles_structure(smiles, size=(400, 300), repeats=3):
    """
    Render 2D molecular structure from SMILES notation.
    For polymers, extend the chain by repeating units.
    
    Args:
        smiles (str): SMILES notation
        size (tuple): Image size (width, height)
        repeats (int): Number of repeats for polymer visualization
        
    Returns:
        PIL.Image: Rendered molecular structure or None if invalid
    """
    try:
        # Handle polymer SMILES with repeat units
        if '*' in smiles:
            # Create extended polymer chain for visualization
            # Remove asterisks and repeat the unit
            repeat_unit = smiles.replace('*', '')
            extended_smiles = repeat_unit * repeats
        else:
            extended_smiles = smiles
        
        # Create molecule object
        mol = Chem.MolFromSmiles(extended_smiles)
        if mol is None:
            return None
        
        # Generate 2D coordinates
        Compute2DCoords(mol)
        
        # Render to image
        img = Draw.MolToImage(mol, size=size, kekulize=True)
        
        return img
        
    except Exception as e:
        st.error(f"Error rendering SMILES structure: {str(e)}")
        return None

def validate_polymer_structure(smiles):
    """
    Additional validation specifically for polymer structures.
    
    Args:
        smiles (str): SMILES notation
        
    Returns:
        dict: Validation results with polymer-specific checks
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'is_valid': False,
                'is_polymer': False,
                'has_repeat_units': False,
                'message': 'Invalid SMILES'
            }
        
        # Check for repeat unit indicators
        has_repeat_units = '*' in smiles
        
        # Check for polymer-like patterns
        is_polymer_like = has_repeat_units or any(pattern in smiles for pattern in ['CC', 'OO', 'NN'])
        
        return {
            'is_valid': True,
            'is_polymer': is_polymer_like,
            'has_repeat_units': has_repeat_units,
            'molecular_weight': Descriptors.ExactMolWt(mol),
            'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
            'message': 'Valid structure'
        }
        
    except Exception as e:
        return {
            'is_valid': False,
            'is_polymer': False,
            'has_repeat_units': False,
            'message': f'Error: {str(e)}'
        }
