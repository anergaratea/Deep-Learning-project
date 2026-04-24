import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_morgan_fingerprint(smiles, radius=2, nBits=2048):
    """
    Computes Morgan Fingerprints using RDKit array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((nBits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def train_rf_baseline():
    """
    Trains a Random Forest regressor on Morgan Fingerprints as baseline.
    """
    # Same Dataset 
    smiles_list = ["O", "C", "CCO", "c1ccccc1", "CCC", "CC", "CC(=O)O", "CCN", "c1ccncc1", "CCCl"]
    labels = [-0.5, 1.1, -0.3, 2.1, 1.8, 1.0, -0.2, 0.5, 2.0, 1.5]

    X = np.array([smiles_to_morgan_fingerprint(s) for s in smiles_list])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Random Forest Baseline MAE: {mae:.4f}")
    print(f"Random Forest Baseline RMSE: {rmse:.4f}")

if __name__ == "__main__":
    train_rf_baseline()
