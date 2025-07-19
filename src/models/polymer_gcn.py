"""
Simple GCN Model for Polymer Property Prediction

Graph Convolutional Network implementation using PyTorch Geometric
for glass transition temperature (Tg) prediction from molecular graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class PolymerGCN(nn.Module):
    """
    Simple Graph Convolutional Network for polymer property prediction.
    
    Uses GCN layers to process molecular graphs and predict Tg values.
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 num_gcn_layers: int = 3,
                 dropout_rate: float = 0.2,
                 pooling_method: str = 'mean',
                 use_molecular_features: bool = True,
                 molecular_feature_dim: int = 13,
                 activation: str = 'relu'):
        """
        Initialize the GCN model.
        
        Args:
            node_feature_dim: Dimension of node features
            hidden_dims: List of hidden layer dimensions for final MLP
            num_gcn_layers: Number of GCN layers
            dropout_rate: Dropout rate for regularization
            pooling_method: Graph pooling method ('mean', 'max', 'sum')
            use_molecular_features: Whether to use molecular-level features
            molecular_feature_dim: Dimension of molecular features
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()
        
        self.num_gcn_layers = num_gcn_layers
        self.dropout_rate = dropout_rate
        self.pooling_method = pooling_method
        self.use_molecular_features = use_molecular_features
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        gcn_dims = [node_feature_dim] + [hidden_dims[0]] * num_gcn_layers
        
        for i in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(gcn_dims[i], gcn_dims[i + 1]))
        
        # Graph pooling
        if pooling_method == 'mean':
            self.pool = global_mean_pool
        elif pooling_method == 'max':
            self.pool = global_max_pool
        elif pooling_method == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pooling_method}")
        
        # Calculate input dimension for final MLP
        graph_feature_dim = gcn_dims[-1]  # Output from GCN layers
        mlp_input_dim = graph_feature_dim
        
        if use_molecular_features:
            mlp_input_dim += molecular_feature_dim
        
        # Final MLP layers
        self.mlp_layers = nn.ModuleList()
        mlp_dims = [mlp_input_dim] + hidden_dims[1:] + [1]  # +1 for final output
        
        for i in range(len(mlp_dims) - 1):
            self.mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:  # No dropout after final layer
                self.mlp_layers.append(nn.Dropout(dropout_rate))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass through the GCN model.
        
        Args:
            data: Batch of graph data from PyTorch Geometric
            
        Returns:
            Predicted property values [batch_size, 1]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GCN layers
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            x = self.activation(x)
            
            # Apply dropout except after last GCN layer
            if i < len(self.gcn_layers) - 1:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Graph-level pooling
        graph_features = self.pool(x, batch)
        
        # Concatenate with molecular features if available
        if self.use_molecular_features and hasattr(data, 'mol_features'):
            mol_features = data.mol_features
            batch_size = graph_features.shape[0]
            
            # Handle PyTorch Geometric's batching of molecular features
            if mol_features.dim() == 1:
                # PyTorch Geometric concatenates features, so reshape back to [batch_size, feature_dim]
                expected_feature_dim = mol_features.shape[0] // batch_size
                mol_features = mol_features.view(batch_size, expected_feature_dim)
            
            graph_features = torch.cat([graph_features, mol_features], dim=1)
        
        # Final MLP
        x = graph_features
        for i in range(0, len(self.mlp_layers), 2):
            linear = self.mlp_layers[i]
            x = linear(x)
            
            # Apply activation and dropout (except for final layer)
            if i < len(self.mlp_layers) - 2:
                x = self.activation(x)
                if i + 1 < len(self.mlp_layers):
                    dropout = self.mlp_layers[i + 1]
                    x = dropout(x)
        
        return x
    
    def get_graph_embeddings(self, data: Batch) -> torch.Tensor:
        """
        Get graph-level embeddings without final prediction layer.
        
        Args:
            data: Batch of graph data
            
        Returns:
            Graph embeddings [batch_size, embedding_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Graph-level pooling
        graph_features = self.pool(x, batch)
        
        # Concatenate with molecular features if available
        if self.use_molecular_features and hasattr(data, 'mol_features'):
            mol_features = data.mol_features
            batch_size = graph_features.shape[0]
            
            # Handle PyTorch Geometric's batching of molecular features
            if mol_features.dim() == 1:
                # PyTorch Geometric concatenates features, so reshape back to [batch_size, feature_dim]
                expected_feature_dim = mol_features.shape[0] // batch_size
                mol_features = mol_features.view(batch_size, expected_feature_dim)
            
            graph_features = torch.cat([graph_features, mol_features], dim=1)
        
        return graph_features


class PolymerGCNDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that works with the existing PolymerTgDataset
    for GCN training.
    """
    
    def __init__(self, polymer_dataset):
        """
        Initialize with a PolymerTgDataset.
        
        Args:
            polymer_dataset: Instance of PolymerTgDataset
        """
        self.polymer_dataset = polymer_dataset
    
    def __len__(self):
        return len(self.polymer_dataset)
    
    def __getitem__(self, idx):
        return self.polymer_dataset[idx]


def create_gcn_model_from_config(config: dict, node_feature_dim: int) -> PolymerGCN:
    """
    Create GCN model from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        node_feature_dim: Dimension of node features
        
    Returns:
        Configured PolymerGCN model
    """
    model = PolymerGCN(
        node_feature_dim=node_feature_dim,
        hidden_dims=config.get('hidden_dims', [128, 64, 32]),
        num_gcn_layers=config.get('num_gcn_layers', 3),
        dropout_rate=config.get('dropout_rate', 0.2),
        pooling_method=config.get('pooling_method', 'mean'),
        use_molecular_features=config.get('use_molecular_features', True),
        molecular_feature_dim=config.get('molecular_feature_dim', 13),
        activation=config.get('activation', 'relu')
    )
    
    return model


# Example usage and testing
def test_polymer_gcn():
    """Test the GCN model with dummy data."""
    print("Testing Polymer GCN Model")
    print("=" * 40)
    
    # Create dummy graph data
    batch_size = 4
    num_nodes_per_graph = 20
    node_feature_dim = 157  # Based on MolecularGraphConverter
    mol_feature_dim = 13
    
    # Create dummy batch
    graphs = []
    for i in range(batch_size):
        # Random node features
        x = torch.randn(num_nodes_per_graph, node_feature_dim)
        
        # Random edge connectivity (small graph)
        num_edges = 30
        edge_index = torch.randint(0, num_nodes_per_graph, (2, num_edges))
        
        # Random molecular features
        mol_features = torch.randn(mol_feature_dim)
        
        # Target value
        y = torch.randn(1)
        
        graph = Data(x=x, edge_index=edge_index, mol_features=mol_features, y=y)
        graphs.append(graph)
    
    # Create batch
    from torch_geometric.loader import DataLoader
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    
    # Test model
    model = PolymerGCN(
        node_feature_dim=node_feature_dim,
        hidden_dims=[128, 64, 32],
        num_gcn_layers=3,
        use_molecular_features=True,
        molecular_feature_dim=mol_feature_dim
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch)
        embeddings = model.get_graph_embeddings(batch)
    
    print(f"Input batch size: {batch_size}")
    print(f"Output shape: {output.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample predictions: {output.squeeze().numpy()}")
    
    # Test training mode
    model.train()
    output_train = model(batch)
    loss = F.mse_loss(output_train.squeeze(), batch.y)
    
    print(f"Training loss: {loss.item():.4f}")
    print("✅ GCN model test passed!")


if __name__ == "__main__":
    test_polymer_gcn() 