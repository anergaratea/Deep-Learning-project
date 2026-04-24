import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import os

def plot_training_metrics(train_losses, val_losses, save_path="metrics.png"):
    """
    Plots training and validation loss curves.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GNN Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")
    plt.close()

def visualize_molecule(smiles, title="Molecule", save_path="molecule.png"):
    """
    Visualizes a molecule directly from its SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300), kekulize=True)
        img.save(save_path)
        print(f"Molecule {title} saved to {save_path}")
    else:
        print(f"Failed to decode SMILES: {smiles}")

def plot_true_vs_predicted(y_true, y_pred, save_path="true_vs_pred.png"):
    """
    Plots True vs Predicted values to evaluate regression manually.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Model Predictions vs True Properties")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
