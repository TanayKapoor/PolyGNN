"""
Test suite for data processing modules.

Tests for BigSMILES parser, molecular graph converter, and polymer dataset functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data.bigsmiles_parser import BigSMILESParser
from src.data.molecular_graph import MolecularGraphConverter
from src.data.polymer_dataset import PolymerTgDataset


class TestBigSMILESParser:
    """Test suite for BigSMILES parser functionality."""
    
    def test_bigsmiles_parser(self):
        """Test the BigSMILES parser with example polymers"""
        parser = BigSMILESParser()
        
        # Test cases
        test_cases = [
            # Polyethylene
            "CC{[CH2][CH2]}CC",
            # Polystyrene
            "CC{[CH2][CH]c1ccccc1}CC",
            # PET (Polyethylene terephthalate)
            "{[#OCH2CH2OC(=O)c1ccc(cc1)C(=O)#]}",
            # Random copolymer
            "CC{[CH2][CH2],[CH2][CH]C}CC"
        ]
        
        for i, bigsmiles in enumerate(test_cases):
            result = parser.parse_bigsmiles(bigsmiles)
            
            # Basic assertions
            assert isinstance(result, dict)
            assert 'repeat_units' in result
            assert 'avg_repeat_mw' in result
            assert 'complexity_score' in result
            assert len(result['repeat_units']) > 0
            assert result['avg_repeat_mw'] > 0
            assert result['complexity_score'] >= 0


class TestMolecularGraphConverter:
    """Test suite for molecular graph converter."""
    
    def test_molecular_graph_converter(self):
        """Test molecular graph conversion functionality"""
        converter = MolecularGraphConverter()
        
        # Test SMILES
        test_smiles = [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(=O)O",  # Acetic acid
            "*CC*",  # Ethylene (with dummy atoms)
            "C=C",  # Ethylene
            "CCN",  # Ethylamine
            "CC(C)O",  # Isopropanol
        ]
        
        successful_conversions = 0
        for i, smiles in enumerate(test_smiles):
            try:
                result = converter.smiles_to_data(smiles)
                if result is not None:
                    successful_conversions += 1
                    # Basic assertions
                    assert hasattr(result, 'x')  # Node features
                    assert hasattr(result, 'edge_index')  # Edge connectivity
                    assert result.x.size(0) > 0  # Has nodes
            except Exception as e:
                print(f"Failed to convert {smiles}: {e}")
        
        # Expect at least 80% success rate
        success_rate = successful_conversions / len(test_smiles)
        assert success_rate >= 0.8, f"Conversion success rate too low: {success_rate:.1%}"


class TestPolymerTgDataset:
    """Test suite for polymer Tg dataset functionality."""
    
    def test_polymer_tg_dataset_creation(self):
        """Test polymer dataset creation with minimal data"""
        
        # Create minimal test data
        example_data = [
            {'smiles': 'CCO', 'tg': 150.0},
            {'smiles': 'c1ccccc1', 'tg': 200.0},
            {'smiles': 'CC(=O)O', 'tg': 175.0},
        ]
        
        import pandas as pd
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(example_data).to_csv(f.name, index=False)
            test_csv = f.name
        
        try:
            # Test dataset creation
            dataset = PolymerTgDataset(
                root='./test_data',
                csv_file=test_csv,
                smiles_col='smiles',
                target_col='tg',
                split_ratios=(0.7, 0.15, 0.15),
                use_polymer_features=False
            )
            
            # Basic assertions
            assert len(dataset) > 0
            assert hasattr(dataset, 'data')
            assert hasattr(dataset, 'slices')
            
            # Test data access
            sample = dataset[0]
            assert hasattr(sample, 'x')  # Node features
            assert hasattr(sample, 'y')  # Target value
            assert hasattr(sample, 'edge_index')  # Edge connectivity
            
        finally:
            # Clean up
            import os
            if os.path.exists(test_csv):
                os.unlink(test_csv)


if __name__ == "__main__":
    pytest.main([__file__]) 