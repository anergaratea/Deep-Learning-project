"""
Tutorial: Cómo entrenar una Red Neuronal de Grafos (GNN) – Explicado paso a paso

Este script entrena una GNN (o una GAT) para predecir propiedades moleculares usando grafos derivados de SMILES.

Cada bloque incluye explicaciones pensadas para quien ya conoce Deep Learning pero es nuevo en aplicaciones de redes sobre grafos moleculares.
"""

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

# --------------------------------------------------
# 1. FUNCIONES AUXILIARES PARA MANEJO DE LOTES DE GRAFOS
# --------------------------------------------------
def collate_fn(batch):
    """
    Junta varios grafos moleculares en un batch grande para entrenamiento eficiente.
    - Ajusta los índices de los nodos y aristas para que todos convivan en el mismo tensor.
    - Devuelve un diccionario con todos los tensores concatenados, listo para el modelo.
    """
    batch_atom_features = []
    batch_edges = []
    batch_edge_features = []
    batch_labels = []
    batch_indices = []
    
    node_offset = 0
    for i, graph in enumerate(batch):
        num_atoms = graph['num_atoms']
        # Matriz de features de los átomos de esta molécula
        batch_atom_features.append(graph['atom_features'])
        # Se suman los offsets al índice de las aristas para que no se solapen entre moléculas
        edges = graph['edges'] + node_offset
        batch_edges.append(edges)
        batch_edge_features.append(graph['edge_features'])
        batch_labels.append(graph['label'])
        # Batch indices sirven para el readout global al final
        batch_indices.append(torch.full((num_atoms,), i, dtype=torch.long))
        node_offset += num_atoms
    return {
        'atom_features': torch.cat(batch_atom_features, dim=0),
        'edges': torch.cat(batch_edges, dim=1),
        'edge_features': torch.cat(batch_edge_features, dim=0),
        'labels': torch.cat(batch_labels, dim=0),
        'batch_idx': torch.cat(batch_indices, dim=0)
    }

# --------------------------------------------------
# 2. PIPELINE PRINCIPAL DE ENTRENAMIENTO
# --------------------------------------------------
def train():
    print("Cargando y preprocesando dataset...")
    # Carga SMILES y labels experimentales
    smiles_list, labels = load_real_dataset()

    # OPCIONAL: visualizar cómo es una molécula convertida (solo el primero)
    visualize_molecule(smiles_list[0], title="molecule_0", save_path="molecule.png")

    print(f"Procesando {len(smiles_list)} moléculas. Conversión a grafos...")
    # Convierte cada molécula en un grafo con features estructurales
    dataset = create_dataset(smiles_list, labels)
    print(f"Moleculas válidas convertidas a grafos: {len(dataset)}")

    # Separación entrenamiento/validación (80/20)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # DataLoader permite procesar lotes pequeños de grafos (por eficiencia de GPU)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # ------------------------------
    # CONFIGURACIÓN DEL MODELO
    # ------------------------------
    node_dim = 4  # n° de features por átomo (depende de preprocessing)
    edge_dim = 6  # n° de features por enlace
    hidden_dim = 64  # tamaño del embedding intermedio
    output_dim = 1   # salida: un valor de propiedad por molécula

    use_gat = False # Cambia a True para usar el modelo de atención en grafos (GAT)

    if use_gat:
        # GAT es un modelo de atención: aprende qué átomos vecinos son más relevantes
        model = MolecularGAT(node_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2)
    else:
        # GNN convencional (message passing): todos los vecinos igual de importantes
        model = MolecularGNN(node_dim, edge_dim, hidden_dim, output_dim, n_layers=3)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Como es regresión, usamos error cuadrático (MAE también sería válido)

    epochs = 100
    train_losses = []
    val_losses = []

    print("Comienza el entrenamiento...! (Verás un reporte cada 10 epochs)")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            # Preparamos el batch para alimentar a la red
            x = batch['atom_features']                 # (n_atoms_total, node_dim)
            edge_index = batch['edges']                # (2, n_edges_total)
            edge_attr = batch['edge_features']         # (n_edges_total, edge_dim)
            batch_idx = batch['batch_idx']             # índice de batch para pooling global
            targets = batch['labels']                  # (n_molecules,)

            if use_gat:
                # Para GAT hay que transformar edge_index en matriz densa de adyacencia
                num_nodes = x.size(0)
                adj = torch.zeros((num_nodes, num_nodes), device=x.device)
                adj[edge_index[0], edge_index[1]] = 1.0
                preds = model(x, adj, batch_idx).view(-1)
            else:
                preds = model(x, edge_index, edge_attr, batch_idx).view(-1)

            # Calcula el error de predicción
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ------------------------------
        # VALIDACIÓN AL FINAL DE CADA EPOCH
        # ------------------------------
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

    # Tras el entrenamiento, generamos visualizaciones útiles
    print("Guardando visualizaciones de métricas...")
    plot_training_metrics(train_losses, val_losses, save_path="training_metrics.png")
    plot_true_vs_predicted(all_true, all_preds, save_path="true_vs_predicted.png")
    print("Listo. Observa los gráficos y prueba cambiando hiperparámetros.")

if __name__ == '__main__':
    train()
