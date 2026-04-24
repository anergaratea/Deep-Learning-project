import os
import collections
import numpy as np
import torch
from rdkit import Chem

def get_atom_features(atom):
    """
    Extracts features from an RDKit atom object.
    
    Returns:
        np.array: Vector of features for the given atom.
    """
    # Atom symbol (One-hot encoding for common elements)
    symbols = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'Unknown']
    symbol = atom.GetSymbol()
    symbol_idx = symbols.index(symbol) if symbol in symbols else symbols.index('Unknown')
    
    # Degree (number of bonds)
    degree = atom.GetDegree()
    
    # Implicit valence
    implicit_valence = atom.GetImplicitValence()
    
    # Needs to be aromatic?
    is_aromatic = 1 if atom.GetIsAromatic() else 0

    features = [symbol_idx, degree, implicit_valence, is_aromatic]
    return np.array(features, dtype=np.float32)


def get_bond_features(bond):
    """
    Extracts features from an RDKit bond object.
    """
    bond_type = bond.GetBondType()
    features = [
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]
    return np.array(features, dtype=np.float32)


def smiles_to_graph(smiles):
    """
    Converts a SMILES string to a graph representation (node features, edge features, adjacency list).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get node features
    num_atoms = mol.GetNumAtoms()
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    atom_features = np.array(atom_features)

    # Get edge features and adjacency list
    edges = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        features = get_bond_features(bond)
        
        edges.append((i, j))
        edge_features.append(features)
        
        edges.append((j, i))
        edge_features.append(features)

    return {
        'num_atoms': num_atoms,
        'atom_features': torch.tensor(atom_features, dtype=torch.float32),
        'edges': torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long),
        'edge_features': torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.empty((0, 6), dtype=torch.float32)
    }

def create_dataset(smiles_list, labels):
    """
    Creates a list of graph dictionaries from lists of SMILES and corresponding labels.
    """
    dataset = []
    for smiles, label in zip(smiles_list, labels):
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graph['label'] = torch.tensor([label], dtype=torch.float32)
            dataset.append(graph)
    return dataset
