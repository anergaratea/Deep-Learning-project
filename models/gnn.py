import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, n_layers=3):
        """
        Molecular Graph Neural Network
        
        Args:
            node_dim (int): Number of atom features
            edge_dim (int): Number of bond features
            hidden_dim (int): Hidden dimension size
            output_dim (int): Output size (e.g., 1 for regression, N for classification)
            n_layers (int): Number of message passing layers
        """
        super(MolecularGNN, self).__init__()
        
        self.n_layers = n_layers
        
        # Initial embeddings
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        
        # Message passing layers
        self.conv_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(n_layers)
        ])
        
        # Readout (Global pooling layer) and final MLP
        self.pool = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x (Tensor): Node features. Shape: (num_nodes, node_dim)
            edge_index (Tensor): Edge indices. Shape: (2, num_edges)
            edge_attr (Tensor): Edge features. Shape: (num_edges, edge_dim)
            batch (Tensor): Batch indices for nodes.
        """
        # Embed node and edge features
        x = F.relu(self.node_embed(x))
        edge_attr = F.relu(self.edge_embed(edge_attr))
        
        # Message Passing steps
        for i in range(self.n_layers):
            row, col = edge_index
            
            # Message construction (concatenate node features with edge features)
            messages = torch.cat([x[row], edge_attr], dim=1)
            messages = F.relu(self.conv_layers[i](messages))
            
            # Aggregate messages (sum pooling)
            aggr_out = torch.zeros(x.size(0), x.size(1), device=x.device)
            aggr_out.index_add_(0, col, messages)
            
            # Update node representations
            x = x + aggr_out

        # Global Readout: Sum pooling over all nodes in the molecule
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        out.index_add_(0, batch, x)
        
        # Final prediction
        out = self.fc(out)
        return out
