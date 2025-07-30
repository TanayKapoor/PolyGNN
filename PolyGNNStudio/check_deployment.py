#!/usr/bin/env python3
"""
Deployment readiness check for PolyGNN Showcase
"""

import os
import sys
import importlib.util

def check_file_exists(filepath, required=True):
    """Check if a file exists and report status."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = " (required)" if required else " (optional)"
    print(f"{status} {filepath}{req_text}")
    return exists

def check_directory_exists(dirpath, required=True):
    """Check if a directory exists and report status."""
    exists = os.path.isdir(dirpath)
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = " (required)" if required else " (optional)"
    print(f"{status} {dirpath}/{req_text}")
    return exists

def check_import(module_name, required=True):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} import successful")
        return True
    except ImportError as e:
        status = "❌" if required else "⚠️"
        req_text = " (required)" if required else " (optional)"
        print(f"{status} {module_name} import failed{req_text}: {e}")
        return False

def check_model_files():
    """Check for required model files."""
    print("\n📁 Model Files:")
    model_paths = [
        "../results/final_optimized_model.pth",
        "../results/hpo/hpo_20250721_074803/best_model.pth"
    ]
    
    found_model = False
    for path in model_paths:
        exists = check_file_exists(path, required=False)
        if exists:
            found_model = True
            # Check file size
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  └── Size: {size_mb:.1f} MB")
    
    if not found_model:
        print("❌ No model files found - app will use fallback mode")
    
    return found_model

def main():
    """Run deployment readiness checks."""
    print("🔍 PolyGNN Showcase Deployment Readiness Check")
    print("=" * 50)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    all_checks_passed = True
    
    # Core application files
    print("\n📄 Core Application Files:")
    core_files = [
        "app.py",
        "requirements.txt",
        "polygnn_integration.py",
        ".streamlit/config.toml"
    ]
    
    for file in core_files:
        if not check_file_exists(file):
            all_checks_passed = False
    
    # Utility modules
    print("\n📦 Utility Modules:")
    util_files = [
        "utils/model_utils.py",
        "utils/data_processing.py",
        "utils/visualization.py",
        "data/demo_polymers.py"
    ]
    
    for file in util_files:
        if not check_file_exists(file):
            all_checks_passed = False
    
    # Deployment files
    print("\n🚀 Deployment Files:")
    deployment_files = [
        ("Dockerfile", True),
        ("Procfile", False),
        ("setup.sh", False),
        ("package.json", False),
        ("DEPLOYMENT.md", False)
    ]
    
    for file, required in deployment_files:
        check_file_exists(file, required)
    
    # Python dependencies
    print("\n🐍 Python Dependencies:")
    critical_deps = [
        "streamlit",
        "torch",
        "pandas",
        "numpy",
        "plotly"
    ]
    
    for dep in critical_deps:
        if not check_import(dep):
            all_checks_passed = False
    
    # Optional dependencies
    optional_deps = [
        "torch_geometric",
        "rdkit",
        "shap"
    ]
    
    for dep in optional_deps:
        check_import(dep, required=False)
    
    # Model files
    model_found = check_model_files()
    
    # Source code availability
    print("\n📚 Source Code:")
    check_directory_exists("../src", required=False)
    
    # Final assessment
    print("\n" + "=" * 50)
    if all_checks_passed:
        if model_found:
            print("🎉 DEPLOYMENT READY - All checks passed with real models!")
            print("   The app will run with full PolyGNN functionality.")
        else:
            print("✅ DEPLOYMENT READY - Core functionality available")
            print("   The app will run in demonstration mode (no real models).")
        print("\nRecommended deployment platforms:")
        print("  • Streamlit Cloud (easiest)")
        print("  • Heroku (free tier)")
        print("  • Railway (modern platform)")
        print("  • Docker (self-hosted)")
        return 0
    else:
        print("❌ DEPLOYMENT NOT READY - Missing critical files/dependencies")
        print("   Please resolve the issues marked with ❌ above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())