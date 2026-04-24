import urllib.request
import os
import pandas as pd

def download_freesolv_dataset(data_dir="data"):
    """
    Downloads the FreeSolv dataset containing molecules and experimental hydration free energy.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    filepath = os.path.join(data_dir, "freesolv.csv")
    
    if not os.path.exists(filepath):
        print(f"Downloading FreeSolv dataset to {filepath}...")
        # FreeSolv repository from deepchem
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")
    else:
        print("Dataset already exists.")

    return filepath

def load_real_dataset():
    """
    Loads SMILES and target values from FreeSolv dataset
    """
    filepath = download_freesolv_dataset()
    df = pd.read_csv(filepath)
    
    # FreeSolv column names: 'smiles' for the molecule, 'expt' for hydration free energy
    smiles_list = df['smiles'].tolist()
    labels = df['expt'].tolist()
    
    return smiles_list, labels

if __name__ == "__main__":
    s, l = load_real_dataset()
    print(f"Loaded {len(s)} molecules.")
    print("Example:", s[0], "-->", l[0])
