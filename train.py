import torch
import torch.nn as nn
import torch.optim as optim
from models.gnn import MolecularGNN
from models.gat import MolecularGAT
from data.preprocessing import create_dataset
from data.dataset import load_real_dataset
from utils.visualization import plot_training_metrics, plot_true_vs_predicted, visualize_molecule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np

def collate_fn(batch):
    """
    Collate function to batch multiple graph dictionaries into a single large graph.
    """
    batch_atom_features = []
    batch_edges = []
    batch_edge_features = []
    batch_labels = []
    batch_indices = []

    node_offset = 0
    for i, graph in enumerate(batch):
        num_atoms = graph['num_atoms']
        
        batch_atom_features.append(graph['atom_features'])
        
        # Shift edge indices by the number of nodes already in the batch
        edges = graph['edges'] + node_offset
        batch_edges.append(edges)
        
        batch_edge_features.append(graph['edge_features'])
        batch_labels.append(graph['label'])
        
        # Batch index for readout pooling
        batch_indices.append(torch.full((num_atoms,), i, dtype=torch.long))

        node_offset += num_atoms

    return {
        'atom_features': torch.cat(batch_atom_features, dim=0),
        'edges': torch.cat(batch_edges, dim=1),
        'edge_features': torch.cat(batch_edge_features, dim=0),
        'labels': torch.cat(batch_labels, dim=0),
        'batch_idx': torch.cat(batch_indices, dim=0)
    }

def train():
    print("Loading dataset...")
    smiles_list, labels = load_real_dataset()
    
    # Optional: visualize the first molecule
    visualize_molecule(smiles_list[0], title="molecule_0", save_path="molecule.png")

    print(f"Processing {len(smiles_list)} molecules. This might take a few seconds...")
    dataset = create_dataset(smiles_list, labels)
    print(f"Valid molecules converted to graphs: {len(dataset)}")
    
    # Split
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Model settings
    node_dim = 4  # Matches size of features from preprocessing
    edge_dim = 6  # Matches number of edge features
    hidden_dim = 64
    output_dim = 1
    
    use_gat = False # Toggle this to switch to GAT Model

    if use_gat:
        # GAT doesn't use edge variables directly by default in standard design, just adjacency
        model = MolecularGAT(node_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2)
    else:
        model = MolecularGNN(node_dim, edge_dim, hidden_dim, output_dim, n_layers=3)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 100
    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            x = batch['atom_features']
            edge_index = batch['edges']
            edge_attr = batch['edge_features']
            batch_idx = batch['batch_idx']
            targets = batch['labels']

            if use_gat:
                # Convert edge_index to dense adjacency matrix for GAT
                num_nodes = x.size(0)
                adj = torch.zeros((num_nodes, num_nodes), device=x.device)
                adj[edge_index[0], edge_index[1]] = 1.0
                preds = model(x, adj, batch_idx).view(-1)
            else:
                preds = model(x, edge_index, edge_attr, batch_idx).view(-1)
                
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_true = []
        all_preds = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['atom_features']
                edge_index = batch['edges']
                edge_attr = batch['edge_features']
                batch_idx = batch['batch_idx']
                targets = batch['labels']

                if use_gat:
                    num_nodes = x.size(0)
                    adj = torch.zeros((num_nodes, num_nodes), device=x.device)
                    adj[edge_index[0], edge_index[1]] = 1.0
                    preds = model(x, adj, batch_idx).view(-1)
                else:
                    preds = model(x, edge_index, edge_attr, batch_idx).view(-1)

                loss = criterion(preds, targets)
                val_loss += loss.item()
                
                all_true.extend(targets.numpy())
                all_preds.extend(preds.numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Visualizations after training
    print("Saving visualizations...")
    plot_training_metrics(train_losses, val_losses, save_path="training_metrics.png")
    plot_true_vs_predicted(all_true, all_preds, save_path="true_vs_predicted.png")

if __name__ == '__main__':
    train()
