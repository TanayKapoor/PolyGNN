#!/usr/bin/env python
"""
Quick test script to verify polymer GNN environment setup
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_bigsmiles_parser():
    """Test BigSMILES parser functionality"""
    print("Testing BigSMILES parser...")
    
    try:
        from data.bigsmiles_parser import BigSMILESParser
        
        parser = BigSMILESParser()
        
        # Test with polyethylene
        result = parser.parse_bigsmiles("CC{[CH2][CH2]}CC")
        
        print(f"✅ BigSMILES parser working")
        print(f"   Repeat units: {len(result['repeat_units'])}")
        print(f"   Average MW: {result['avg_repeat_mw']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ BigSMILES parser failed: {e}")
        return False

def test_imports():
    """Test essential imports"""
    print("Testing essential imports...")
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('pathlib', 'pathlib'),
        ('requests', 'requests'),
    ]
    
    all_imported = True
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}: OK")
        except ImportError as e:
            print(f"❌ {package_name}: Failed - {e}")
            all_imported = False
    
    return all_imported

def test_directory_structure():
    """Test that all required directories exist"""
    print("Testing directory structure...")
    
    required_dirs = [
        'src/data',
        'src/models',
        'src/training',
        'src/evaluation',
        'data/raw',
        'data/processed',
        'data/external',
        'data/interim',
        'notebooks',
        'tests',
        'models',
        'results',
        'docs'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}: OK")
        else:
            print(f"❌ {dir_path}: Missing")
            all_exist = False
    
    return all_exist

def test_files():
    """Test that required files exist"""
    print("Testing required files...")
    
    required_files = [
        'environment.yml',
        'requirements.txt',
        'setup_conda_env.sh',
        'verify_conda_setup.py',
        'dataset_setup.py',
        'src/data/bigsmiles_parser.py',
        'README.md'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}: OK")
        else:
            print(f"❌ {file_path}: Missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧪 Polymer GNN Environment Setup Test\n")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Required Files", test_files),
        ("Essential Imports", test_imports),
        ("BigSMILES Parser", test_bigsmiles_parser),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n=== {test_name} ===")
        success = test_func()
        results.append(success)
        print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
    
    print(f"\n{'='*50}")
    print(f"Overall Result: {'✅ ALL TESTS PASSED' if all(results) else '❌ SOME TESTS FAILED'}")
    
    if all(results):
        print("\n🎉 Environment setup complete!")
        print("Next steps:")
        print("1. Run: chmod +x setup_conda_env.sh && ./setup_conda_env.sh")
        print("2. Run: conda activate polymer-gnn")
        print("3. Run: python verify_conda_setup.py")
        print("4. Run: python dataset_setup.py")
        print("5. Start developing: jupyter notebook")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 