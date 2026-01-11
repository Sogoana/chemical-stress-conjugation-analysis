import pandas as pd
import os
import numpy as np
import pubchempy as pcp
import requests

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


BASE_DIR = "data/raw"
RESULTS_DIR = "data/processed"

# Load data
cids = pd.read_excel(
    os.path.join(base_dir, "cids.xlsx"),
    sheet_name="cids"
)

# Normalization
cids['CID'] = (
    cids['CID']
    .astype(str)                   # convert to a string
    .str.strip()                   # remove blank spaces
    .str.replace('\xa0', '', regex=False)
    .str.replace('\u00A0', '', regex=False)
    .str.replace(r'\.0$', '', regex=True)  # remove '.0'
    .replace(['', 'nan', 'NaN'], pd.NA)   # replace empty spaces to NA
    .dropna()                       # remove NA
    .astype(int)                    # convert to int
)

def get_smiles(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES/JSON"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['PropertyTable']['Properties'][0]['SMILES']
    except Exception as e:
        print(f"Error fetching SMILES for CID {cid}: {e}")
        return None

from tqdm import tqdm
tqdm.pandas()
cids['SMILES'] = cids['CID'].progress_apply(get_smiles)

# Obtain the physicochemical and chemical features from a list of SMILES.
data = []
morgan_gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)

for smi in cids['SMILES']:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    desc = {
        'SMILES': smi,
        'MoWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'RingCount': Descriptors.RingCount(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'Chi0': Descriptors.Chi0n(mol),
        'Chi1': Descriptors.Chi1n(mol),
        }
    fp = morgan_gen.GetFingerprint(mol)
    for i in range(fp.GetNumBits()):
        desc[f'fp_{i}'] = int(fp.GetBit(i))
    maccs = MACCSkeys.GenMACCSKeys(mol)
    for i in range(1, maccs.GetNumBits()):
        desc[f'maccs_{i}'] = int(maccs.GetBit(i))
    data.append(desc)

df_descriptors = pd.DataFrame(data)
print(df_descriptors.head())

df_descriptors.to_csv(os.path.join(RESULTS_DIR, "features_processed.csv"), index=False)
