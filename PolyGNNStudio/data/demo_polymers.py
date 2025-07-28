"""
Demo polymer data for PolyGNN Showcase application.
Contains pre-loaded polymer samples for demonstration purposes.
"""

def get_demo_polymers():
    """
    Get list of demo polymers with SMILES and experimental properties.
    
    Returns:
        list: List of dictionaries containing polymer information
    """
    demo_polymers = [
        {
            'Name': 'Polyethylene',
            'SMILES': '*CC*',
            'Description': 'High-density polyethylene (HDPE)',
            'Tg_experimental': -120,
            'Tm_experimental': 135,
            'Density_experimental': 0.95,
            'Applications': 'Bottles, containers, pipes'
        },
        {
            'Name': 'Polystyrene',
            'SMILES': '*CC(c1ccccc1)*',
            'Description': 'General purpose polystyrene',
            'Tg_experimental': 100,
            'Tm_experimental': None,  # Amorphous polymer
            'Density_experimental': 1.04,
            'Applications': 'Packaging, disposable cups, insulation'
        },
        {
            'Name': 'Polyvinyl Chloride',
            'SMILES': '*CC(Cl)*',
            'Description': 'Rigid PVC',
            'Tg_experimental': 80,
            'Tm_experimental': 200,
            'Density_experimental': 1.38,
            'Applications': 'Pipes, window frames, flooring'
        },
        {
            'Name': 'Poly(methyl methacrylate)',
            'SMILES': '*CC(C)(C(=O)OC)*',
            'Description': 'PMMA (Acrylic glass)',
            'Tg_experimental': 105,
            'Tm_experimental': None,  # Amorphous polymer
            'Density_experimental': 1.18,
            'Applications': 'Optical lenses, displays, signs'
        },
        {
            'Name': 'Polypropylene',
            'SMILES': '*CC(C)*',
            'Description': 'Isotactic polypropylene',
            'Tg_experimental': -10,
            'Tm_experimental': 165,
            'Density_experimental': 0.90,
            'Applications': 'Automotive parts, textiles, packaging'
        }
    ]
    
    return demo_polymers

def get_extended_demo_polymers():
    """
    Get extended list of 10 demo polymers for comprehensive testing.
    Includes diverse polymer structures from common commercial polymers.
    
    Returns:
        list: Extended list of polymer dictionaries
    """
    extended_polymers = get_demo_polymers() + [
        {
            'Name': 'Polyethylene Terephthalate',
            'SMILES': '*OC(=O)c1ccc(cc1)C(=O)OCC*',
            'Description': 'PET plastic',
            'Tg_experimental': 75,
            'Tm_experimental': 255,
            'Density_experimental': 1.38,
            'Applications': 'Bottles, clothing fibers, films'
        },
        {
            'Name': 'Nylon 6,6',
            'SMILES': '*NC(=O)CCCCC(=O)NCCCCCCN*',
            'Description': 'Polyamide 6,6',
            'Tg_experimental': 50,
            'Tm_experimental': 265,
            'Density_experimental': 1.14,
            'Applications': 'Textiles, carpets, rope'
        },
        {
            'Name': 'Polycarbonate',
            'SMILES': '*OC(=O)Oc1ccc(cc1)C(C)(C)c2ccc(cc2)O*',
            'Description': 'Bisphenol A polycarbonate',
            'Tg_experimental': 145,
            'Tm_experimental': None,  # Amorphous polymer
            'Density_experimental': 1.20,
            'Applications': 'Optical discs, safety glasses, electronics'
        },
        {
            'Name': 'Polyvinyl Acetate',
            'SMILES': '*CC(OC(=O)C)*',
            'Description': 'PVAc polymer',
            'Tg_experimental': 30,
            'Tm_experimental': None,  # Amorphous polymer
            'Density_experimental': 1.19,
            'Applications': 'Adhesives, paints, coatings'
        },
        {
            'Name': 'Polyurethane',
            'SMILES': '*NC(=O)OCCCCCCOC(=O)NCCCCCCN*',
            'Description': 'Flexible polyurethane',
            'Tg_experimental': -40,
            'Tm_experimental': 180,
            'Density_experimental': 1.25,
            'Applications': 'Foams, elastomers, coatings'
        }
    ]
    
    return extended_polymers

def get_polymer_by_name(name):
    """
    Get specific polymer data by name.
    
    Args:
        name (str): Polymer name
        
    Returns:
        dict or None: Polymer data dictionary or None if not found
    """
    polymers = get_extended_demo_polymers()
    
    for polymer in polymers:
        if polymer['Name'].lower() == name.lower():
            return polymer
    
    return None

def get_polymer_categories():
    """
    Get polymers organized by categories.
    
    Returns:
        dict: Dictionary with polymer categories
    """
    categories = {
        'Thermoplastics': [
            'Polyethylene',
            'Polystyrene', 
            'Polyvinyl Chloride',
            'Polypropylene',
            'Polyethylene Terephthalate'
        ],
        'Engineering Plastics': [
            'Poly(methyl methacrylate)',
            'Polycarbonate',
            'Nylon 6,6'
        ],
        'Specialty Polymers': [
            'Polyurethane',
            'Polyvinyl Acetate'
        ]
    }
    
    return categories

def validate_demo_polymer_smiles():
    """
    Validate all demo polymer SMILES strings.
    
    Returns:
        dict: Validation results for each polymer
    """
    from rdkit import Chem
    
    polymers = get_extended_demo_polymers()
    validation_results = {}
    
    for polymer in polymers:
        name = polymer['Name']
        smiles = polymer['SMILES']
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            is_valid = mol is not None
            
            validation_results[name] = {
                'smiles': smiles,
                'is_valid': is_valid,
                'has_repeat_units': '*' in smiles,
                'message': 'Valid' if is_valid else 'Invalid SMILES'
            }
            
        except Exception as e:
            validation_results[name] = {
                'smiles': smiles,
                'is_valid': False,
                'has_repeat_units': False,
                'message': f'Error: {str(e)}'
            }
    
    return validation_results

def get_polymer_property_ranges():
    """
    Get typical property ranges for polymers.
    
    Returns:
        dict: Property ranges for reference
    """
    property_ranges = {
        'Tg': {
            'min': -150,  # Very flexible polymers
            'max': 300,   # Rigid engineering plastics
            'unit': '°C',
            'description': 'Glass Transition Temperature'
        },
        'Tm': {
            'min': -50,   # Low melting point polymers
            'max': 400,   # High performance polymers
            'unit': '°C',
            'description': 'Melting Temperature'
        },
        'Density': {
            'min': 0.8,   # Polyolefins
            'max': 2.5,   # Filled or high-density polymers
            'unit': 'g/cm³',
            'description': 'Density'
        }
    }
    
    return property_ranges

def get_sample_csv_data():
    """
    Generate sample CSV data for testing upload functionality.
    
    Returns:
        str: CSV string with sample polymer data
    """
    polymers = get_demo_polymers()
    
    csv_data = "SMILES,Name,Tg_true,Tm_true,Density_true\n"
    
    for polymer in polymers:
        csv_data += f"{polymer['SMILES']},{polymer['Name']},"
        csv_data += f"{polymer['Tg_experimental']},"
        csv_data += f"{polymer.get('Tm_experimental', '')},"
        csv_data += f"{polymer['Density_experimental']}\n"
    
    return csv_data
