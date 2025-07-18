#!/bin/bash
echo "🚀 Setting up Polymer GNN Conda Environment..."

# Create environment
echo "Creating conda environment..."
conda create -n polymer-gnn python=3.9 -y
echo "Activating environment..."
conda activate polymer-gnn

# Core ML libraries
echo "📦 Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Chemistry libraries
echo "🧪 Installing chemistry libraries..."
conda install -c conda-forge rdkit openbabel -y

# Graph neural networks
echo "🕸️ Installing PyTorch Geometric..."
conda install pyg -c pyg -y
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv --find-links https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Data science stack
echo "📊 Installing data science libraries..."
conda install pandas numpy scipy scikit-learn matplotlib seaborn plotly jupyter notebook ipywidgets tqdm -y

# Additional tools
echo "🔧 Installing development tools..."
conda install pytest pytest-cov -y
pip install wandb tensorboard

# Custom packages
echo "🔬 Installing polymer-specific packages..."
pip install mordred  # Molecular descriptors
pip install bigsmiles || echo "⚠️ BigSMILES not available - will use custom implementation"

echo "✅ Environment setup complete!"
echo "📋 To activate: conda activate polymer-gnn"
echo "📋 To verify: python verify_conda_setup.py" 