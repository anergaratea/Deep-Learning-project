import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    A simple Graph Attention Layer (GAT) for molecular graphs.
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: Node features (N, in_features)
        adj: Adjacency matrix (N, N)
        """
        num_nodes = h.size(0)
        
        # Linear Transformation
        Wh = self.W(h)  # (N, out_features)

        # Self-attention mechanism
        # To compute attention between node i and j, we concatenate Wh_i and Wh_j
        Wh_repeated_in = Wh.repeat_interleave(num_nodes, dim=0) # (N*N, out_features)
        Wh_repeated_out = Wh.repeat(num_nodes, 1)              # (N*N, out_features)
        
        # (N*N, 2*out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in, Wh_repeated_out], dim=1)
        
        # (N, N, 1) -> (N, N)
        e = self.leakyrelu(self.a(all_combinations_matrix).view(num_nodes, num_nodes))

        # Mask attention coefficients
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Normalize with Softmax
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to node features
        h_prime = torch.matmul(attention, Wh) # (N, out_features)
        
        return F.elu(h_prime)

class MolecularGAT(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        """
        Molecular Graph Attention Network
        
        Args:
            node_dim (int): Number of atom features
            hidden_dim (int): Hidden dimension size
            output_dim (int): Output size (e.g., 1 for regression)
            n_layers (int): Number of GAT layers
            dropout (float): Dropout probability
        """
        super(MolecularGAT, self).__init__()
        
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(n_layers)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, adj_matrix, batch):
        """
        Args:
            x (Tensor): Node features. Shape: (num_nodes, node_dim)
            adj_matrix (Tensor): Block diagonal dense adjacency matrix. Shape: (num_nodes, num_nodes)
            batch (Tensor): Batch indices for nodes.
        """
        x = F.relu(self.node_embed(x))
        
        for layer in self.gat_layers:
            x = layer(x, adj_matrix)
        
        # Global Readout: Mean pooling over all nodes in each molecule
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        
        # Count nodes per graph to compute the mean
        count = torch.zeros(num_graphs, 1, device=x.device)
        out.index_add_(0, batch, x)
        count.index_add_(0, batch, torch.ones(x.size(0), 1, device=x.device))
        
        out = out / count.clamp(min=1)
        
        # Final prediction
        out = self.fc(out)
        return out
