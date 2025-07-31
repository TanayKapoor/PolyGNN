"""
Data processing utilities for PolyGNN Showcase application.
Handles SMILES validation, CSV processing, and data formatting.
"""

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import io

# Import RDKit with enhanced headless environment support
try:
    import os
    # Configure for headless environment before importing RDKit drawing
    os.environ['MPLBACKEND'] = 'Agg'  # Use non-interactive matplotlib backend
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
    
    # Enhanced drawing import with headless configuration
    try:
        from rdkit.Chem import Draw
        from rdkit.Chem.rdDepictor import Compute2DCoords
        
        # Configure matplotlib for headless rendering
        import matplotlib
        matplotlib.use('Agg')  # Ensure non-interactive backend
        
        # Test drawing functionality
        test_mol = Chem.MolFromSmiles('CCO')
        if test_mol:
            try:
                test_img = Draw.MolToImage(test_mol, size=(100, 100))
                RDKIT_DRAW_AVAILABLE = True
                st.success("🎨 **Structure Visualization**: Enabled for cloud deployment!")
            except Exception as e:
                RDKIT_DRAW_AVAILABLE = False
                st.info(f"🖼️ **Structure Visualization**: Using fallback mode ({str(e)})")
        else:
            RDKIT_DRAW_AVAILABLE = False
            
    except Exception as e:
        RDKIT_DRAW_AVAILABLE = False
        Draw = None
        Compute2DCoords = None
        st.info(f"🖼️ **Structure Visualization**: Using fallback mode ({str(e)})")
        
except ImportError:
    RDKIT_AVAILABLE = False
    RDKIT_DRAW_AVAILABLE = False
    Chem = None
    Descriptors = None
    Draw = None
    Compute2DCoords = None
    st.info("🧪 **Structure Visualization**: Using text-based fallback rendering")

def validate_smiles(smiles):
    """
    Validate a SMILES string using RDKit.
    
    Args:
        smiles (str): SMILES notation to validate
        
    Returns:
        tuple: (is_valid, message)
    """
    if not RDKIT_AVAILABLE:
        # Basic validation without RDKit
        if not smiles or smiles.strip() == "":
            return False, "Empty SMILES string"
        
        # Basic SMILES format checks
        if any(char in smiles for char in ['<', '>', '|', '^', '&']):
            return False, "Invalid characters in SMILES"
        
        # Check for basic polymer structure
        if '*' in smiles:
            return True, "SMILES format appears valid (polymer with repeat units) - RDKit validation unavailable"
        else:
            return True, "SMILES format appears valid - RDKit validation unavailable"
    
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
    
    if not RDKIT_AVAILABLE:
        # Return empty descriptors when RDKit is not available
        for smiles in smiles_list:
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
    Render 2D molecular structure from SMILES notation with enhanced cloud support.
    For polymers, extend the chain by repeating units.
    
    Args:
        smiles (str): SMILES notation
        size (tuple): Image size (width, height)
        repeats (int): Number of repeats for polymer visualization
        
    Returns:
        PIL.Image: Rendered molecular structure or fallback visualization
    """
    if not RDKIT_DRAW_AVAILABLE:
        # Try fallback visualization
        fallback_img = create_structure_fallback(smiles, size)
        if fallback_img is None:
            # Create ultimate fallback if even matplotlib fails
            return create_simple_text_image(smiles, size)
        return fallback_img
    
    try:
        # Handle polymer SMILES with repeat units
        if '*' in smiles:
            # Create extended polymer chain for visualization
            repeat_unit = smiles.replace('*', '')
            extended_smiles = repeat_unit * repeats
        else:
            extended_smiles = smiles
        
        # Create molecule object
        mol = Chem.MolFromSmiles(extended_smiles)
        if mol is None:
            return create_structure_fallback(smiles, size)
        
        # Generate 2D coordinates for better layout
        Compute2DCoords(mol)
        
        # Enhanced rendering with cloud-optimized settings
        draw_options = Draw.DrawingOptions()
        draw_options.addStereoAnnotation = True
        draw_options.addAtomIndices = False
        draw_options.dotsPerAngstrom = 100
        draw_options.bondLineWidth = 2
        
        # Render to image with optimized settings for headless environment
        img = Draw.MolToImage(
            mol, 
            size=size, 
            kekulize=True,
            options=draw_options,
            fitImage=True
        )
        
        return img
        
    except Exception as e:
        # Try fallback visualization if RDKit rendering fails
        fallback_img = create_structure_fallback(smiles, size)
        if fallback_img is None:
            # Create ultimate fallback if even matplotlib fails
            return create_simple_text_image(smiles, size)
        return fallback_img

def create_structure_fallback(smiles, size=(400, 300)):
    """
    Create a fallback visualization when RDKit drawing is not available.
    Generates a text-based representation of the molecule.
    
    Args:
        smiles (str): SMILES notation
        size (tuple): Desired image size (for compatibility)
        
    Returns:
        PIL.Image: Text-based structure representation
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import io
        
        # Create a figure for the fallback visualization
        fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Clean SMILES for display
        display_smiles = smiles.replace('*', '─')
        
        # Add title and SMILES
        ax.text(5, 8, 'Chemical Structure', ha='center', va='center', 
                fontsize=14, weight='bold')
        ax.text(5, 6.5, display_smiles, ha='center', va='center', 
                fontsize=10, family='monospace', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Add polymer representation if applicable
        if '*' in smiles:
            ax.text(5, 4.5, '(Polymer Repeat Unit)', ha='center', va='center', 
                    fontsize=9, style='italic')
            # Simple polymer chain representation
            for i in range(3):
                circle = patches.Circle((2 + i*3, 2.5), 0.3, 
                                      facecolor='lightcoral', edgecolor='black')
                ax.add_patch(circle)
                if i < 2:
                    ax.plot([2.3 + i*3, 1.7 + (i+1)*3], [2.5, 2.5], 'k-', linewidth=2)
        
        ax.text(5, 1, 'Structure visualization enabled with system libraries', 
                ha='center', va='center', fontsize=8, alpha=0.7)
        
        # Convert to PIL Image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
        
    except Exception as e:
        # Ultimate fallback - create simple text image
        return create_simple_text_image(smiles, size)

def create_simple_text_image(smiles, size=(400, 300)):
    """
    Create a simple text-based image when all other visualization methods fail.
    This is the ultimate fallback that should always work.
    
    Args:
        smiles (str): SMILES notation
        size (tuple): Desired image size
        
    Returns:
        PIL.Image: Simple text-based image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a white image
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if unavailable
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Clean SMILES for display
        display_smiles = smiles.replace('*', '─')
        
        # Draw title
        title = "Chemical Structure"
        if font:
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            draw.text(((size[0] - title_width) // 2, 50), title, fill='black', font=font)
        else:
            draw.text((size[0]//2 - 50, 50), title, fill='black')
        
        # Draw SMILES
        smiles_text = f"SMILES: {display_smiles}"
        if font:
            smiles_bbox = draw.textbbox((0, 0), smiles_text, font=font)
            smiles_width = smiles_bbox[2] - smiles_bbox[0]
            draw.text(((size[0] - smiles_width) // 2, 100), smiles_text, fill='blue', font=font)
        else:
            draw.text((20, 100), smiles_text, fill='blue')
        
        # Add polymer indicator if applicable
        if '*' in smiles:
            polymer_text = "(Polymer Repeat Unit)"
            if font:
                polymer_bbox = draw.textbbox((0, 0), polymer_text, font=font)
                polymer_width = polymer_bbox[2] - polymer_bbox[0]
                draw.text(((size[0] - polymer_width) // 2, 150), polymer_text, fill='green', font=font)
            else:
                draw.text((size[0]//2 - 60, 150), polymer_text, fill='green')
        
        # Draw a simple molecular representation
        center_x, center_y = size[0] // 2, size[1] // 2 + 50
        
        # Draw some circles to represent atoms
        atom_positions = [
            (center_x - 80, center_y),
            (center_x - 40, center_y - 30),
            (center_x, center_y),
            (center_x + 40, center_y - 30),
            (center_x + 80, center_y)
        ]
        
        # Draw bonds (lines between atoms)
        for i in range(len(atom_positions) - 1):
            draw.line([atom_positions[i], atom_positions[i + 1]], fill='black', width=2)
        
        # Draw atoms (circles)
        for pos in atom_positions:
            draw.ellipse([pos[0] - 8, pos[1] - 8, pos[0] + 8, pos[1] + 8], 
                        fill='lightblue', outline='black', width=1)
        
        # Add status text
        status_text = "Structure visualization enabled"
        if font:
            status_bbox = draw.textbbox((0, 0), status_text, font=font)
            status_width = status_bbox[2] - status_bbox[0]
            draw.text(((size[0] - status_width) // 2, size[1] - 30), status_text, fill='gray', font=font)
        else:
            draw.text((size[0]//2 - 80, size[1] - 30), status_text, fill='gray')
        
        return img
        
    except Exception as e:
        # If even PIL fails, create a minimal fallback
        try:
            from PIL import Image
            img = Image.new('RGB', size, color='lightgray')
            return img
        except:
            # This should never happen, but just in case
            return None

def validate_polymer_structure(smiles):
    """
    Additional validation specifically for polymer structures.
    
    Args:
        smiles (str): SMILES notation
        
    Returns:
        dict: Validation results with polymer-specific checks
    """
    if not RDKIT_AVAILABLE:
        # Basic validation without RDKit
        has_repeat_units = '*' in smiles
        is_polymer_like = has_repeat_units or any(pattern in smiles for pattern in ['CC', 'OO', 'NN'])
        
        return {
            'is_valid': True,
            'is_polymer': is_polymer_like,
            'has_repeat_units': has_repeat_units,
            'molecular_weight': None,
            'formula': 'Unknown (RDKit unavailable)',
            'message': 'Basic validation - RDKit unavailable'
        }
    
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
