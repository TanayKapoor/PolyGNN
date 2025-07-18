import requests
import pandas as pd
from pathlib import Path


def setup_dataset_directory():
    """Create directory structure for datasets"""
    data_dir = Path("data")
    subdirs = ["raw", "processed", "external", "interim"]
    
    for subdir in subdirs:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("✅ Dataset directory structure created")
    return data_dir


def download_neurips_dataset():
    """Download NeurIPS 2025 polymer dataset"""
    # Note: Replace with actual dataset URL when available
    dataset_url = "https://example.com/neurips2025_polymer_dataset.csv"
    data_dir = Path("data/raw")
    
    try:
        # Download dataset
        response = requests.get(dataset_url)
        response.raise_for_status()
        
        with open(data_dir / "neurips2025_polymer_dataset.csv", "wb") as f:
            f.write(response.content)
        
        print("✅ NeurIPS 2025 dataset downloaded successfully")
        
        # Verify dataset structure
        df = pd.read_csv(data_dir / "neurips2025_polymer_dataset.csv")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")
        print("Please manually download the dataset from the competition page")


def setup_external_datasets():
    """Setup additional polymer property databases"""
    external_sources = {
        "polymer_database": ("https://polymerdatabase.com/export/"
                             "properties.csv"),
        "literature_data": ("https://github.com/polymer-ml/data/"
                            "polymer_properties.json")
    }
    
    data_dir = Path("data/external")
    
    for name, url in external_sources.items():
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            file_ext = url.split('.')[-1]
            filename = f"{name}.{file_ext}"
            
            with open(data_dir / filename, "wb") as f:
                f.write(response.content)
            
            print(f"✅ {name} downloaded successfully")
            
        except Exception as e:
            print(f"⚠️ {name} download failed: {e}")


def create_sample_data():
    """Create sample polymer data for testing"""
    sample_data = {
        'bigsmiles': [
            "CC{[CH2][CH2]}CC",
            "CC{[CH2][CH]c1ccccc1}CC",
            "{[#OCH2CH2OC(=O)c1ccc(cc1)C(=O)#]}",
            "CC{[CH2][CH2],[CH2][CH]C}CC"
        ],
        'polymer_name': [
            'Polyethylene',
            'Polystyrene',
            'PET',
            'Random Copolymer'
        ],
        'glass_transition_temp': [
            -120,
            100,
            80,
            -50
        ],
        'melting_temp': [
            130,
            240,
            260,
            160
        ],
        'density': [
            0.92,
            1.05,
            1.38,
            0.96
        ]
    }
    
    df = pd.DataFrame(sample_data)
    data_dir = Path("data/raw")
    df.to_csv(data_dir / "sample_polymer_data.csv", index=False)
    print("✅ Sample polymer data created")
    
    return df


if __name__ == "__main__":
    setup_dataset_directory()
    create_sample_data()
    download_neurips_dataset()
    setup_external_datasets() 