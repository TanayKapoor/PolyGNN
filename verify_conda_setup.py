import os
import subprocess
import importlib


def check_conda_environment():
    """Verify conda environment is properly configured"""
    print("=== Conda Environment Verification ===\n")
    
    # Check if we're in the right environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"Current conda environment: {conda_env}")
    
    if conda_env != 'polymer-gnn':
        print("⚠️  Not in polymer-gnn environment. "
              "Run: conda activate polymer-gnn")
        return False
    
    return True


def check_conda_packages():
    """Check conda-specific packages"""
    print("\n=== Conda Package Verification ===")
    
    # Check conda-installed packages
    conda_packages = [
        'pytorch',
        'torchvision',
        'torchaudio',
        'rdkit',
        'openbabel',
        'pyg',
        'pandas',
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'jupyter'
    ]
    
    try:
        result = subprocess.run(['conda', 'list'], capture_output=True, 
                                text=True)
        installed_packages = result.stdout.lower()
        all_installed = True
        
        for package in conda_packages:
            if package in installed_packages:
                print(f"✅ {package}: Installed via conda")
            else:
                print(f"❌ {package}: Not found in conda list")
                all_installed = False
        
        return all_installed
    except Exception as e:
        print(f"❌ Could not check conda packages: {e}")
        return False


def check_package_imports():
    """Check if packages can be imported"""
    print("\n=== Package Import Verification ===")
    
    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('torch-geometric', 'torch_geometric'),
        ('rdkit', 'rdkit'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('jupyter', 'jupyter'),
        ('openbabel', 'openbabel')
    ]
    
    all_working = True
    
    for package_name, import_name in packages:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package_name}: {version}")
        except ImportError:
            print(f"❌ {package_name}: Import failed")
            all_working = False
    
    return all_working


def check_gpu_setup():
    """Check GPU configuration"""
    print("\n=== GPU Configuration ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test GPU computation
            device = torch.device('cuda')
            x = torch.randn(100, 100, device=device)
            y = torch.mm(x, x)
            print("✅ GPU computation test passed")
        else:
            print("⚠️  CUDA not available - will use CPU")
            
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False
    
    return True


def check_torch_geometric():
    """Specific check for PyTorch Geometric"""
    print("\n=== PyTorch Geometric Verification ===")
    
    try:
        import torch
        import torch_geometric
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
        
        # Test basic functionality
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create simple graph
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long,
                                  device=device)
        x = torch.randn(3, 16, device=device)
        
        # Create data object
        data = Data(x=x, edge_index=edge_index)
        
        # Test GCN layer
        conv = GCNConv(16, 32).to(device)
        out = conv(data.x, data.edge_index)
        
        print(f"✅ PyTorch Geometric test passed on {device}")
        print(f"   Input shape: {data.x.shape}")
        print(f"   Output shape: {out.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch Geometric test failed: {e}")
        return False


def check_cuda_installation():
    """Check CUDA installation specifically"""
    print("\n=== CUDA Installation Test ===")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU Name: {torch.cuda.get_device_name()}")
            
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
            
            # Test tensor operations
            device = torch.device('cuda')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            
            print(f"Matrix multiplication test passed on: {device}")
            
        return True
        
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False


def main():
    """Main verification function"""
    print("🧪 Polymer GNN Conda Environment Verification\n")
    
    checks = [
        check_conda_environment(),
        check_conda_packages(),
        check_package_imports(),
        check_gpu_setup(),
        check_torch_geometric(),
        check_cuda_installation()
    ]
    
    if all(checks):
        print("\n🎉 All checks passed! Your conda environment is ready for polymer GNN development!")
        print("\n📋 Quick start commands:")
        print("   conda activate polymer-gnn")
        print("   jupyter notebook")
        print("   python -c 'import torch; print(f\"PyTorch ready: {torch.cuda.is_available()}\")'")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("💡 Common fixes:")
        print("   - Ensure you're in the polymer-gnn environment: conda activate polymer-gnn")
        print("   - Reinstall failed packages: conda install <package_name>")
        print("   - Check CUDA drivers: nvidia-smi")


if __name__ == "__main__":
    main() 