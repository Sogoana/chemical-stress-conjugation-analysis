import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import random
from sklearn.cluster import KMeans


# reproducibilidad
np.random.seed(42)
random.seed(42)

# --- Parameters --- Num. PCs obtained after running PCA Snippet
N_PCS = 30  

# --- Load data ---

BASE_DIR = "data/processed"
RESULTS_DIR = "data/processed"
cids = pd.read_csv(os.path.join(base_dir, "unified_df.csv"))

df = df[df['Concentration'] != 0].copy()

# --- Eliminar los features muy correlacionados ---

def remove_correlated_fp_features(df_features, threshold=0.9):   # Removes highly correlated features (from morgan fp)

    fp_cols = [c for c in df_features.columns if c.startswith("fp")]
    other_cols = [c for c in df_features.columns if not c.startswith("fp")]

    if len(fp_cols) == 0:
        return df_features

    # Correlation matrix (FPs)
    corr_matrix = df_features[fp_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Columns to be removed
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    # Maintain those columnds with inrelated FPs features + other columns
    filtered = df_features.drop(columns=to_drop, errors='ignore')
    return filtered

# --- Define feature groups ---

physico_features = ['MoWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                    'FractionCSP3', 'NumRotatableBonds', 'RingCount', 'BertzCT', 'Chi0', 'Chi1']

Group1_features = ['AM', 'BCIDE', 'FDADD', 'NMAT', 'INDTCHEM', 'PHARMA', 'DISBYP']

Group2_features = ['ACONV', 'ADD', 'ANLGC', 'ANTCAK', 'ANTDEP', 'ANTHIST', 'ARTSW', 'BB',
                   'DISNF', 'EDC', 'FUNG', 'HAN', 'IL', 'INCHEM', 'LR', 'MENP', 'MEONP',
                   'NSAID', 'ORGCHEM', 'PIGM', 'PPP', 'PRES', 'SOLV', 'XRAY']

Concentration = ['Concentration']

metadata = ['Species_relation', 'Time', 'Temperature']

# --- Split global antes de PCA/Clustering ---
X = df.drop(columns=["Fold"])
y = df["Fold"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- PCA on MACCS ---
maccs_cols = [c for c in X_train.columns if c.startswith("maccs_")]
if len(maccs_cols) > 0:
    pca = PCA(n_components=min(N_PCS, len(maccs_cols)), svd_solver="full")
    X_train_pca = pca.fit_transform(X_train[maccs_cols])
    X_val_pca   = pca.transform(X_val[maccs_cols])

    # quitar MACCS originales
    X_train = X_train.drop(columns=maccs_cols).copy()
    X_val   = X_val.drop(columns=maccs_cols).copy()

    for i in range(X_train_pca.shape[1]):
        X_train[f"PC{i+1}"] = X_train_pca[:, i]
        X_val[f"PC{i+1}"]   = X_val_pca[:, i]

# --- Clustering (based on PCs) ---
pcs_train = X_train[[c for c in X_train.columns if c.startswith("PC")]]
pcs_val   = X_val[[c for c in X_val.columns if c.startswith("PC")]]

kmeans = KMeans(n_clusters=2, random_state=42)
X_train["cluster"] = kmeans.fit_predict(pcs_train)
X_val["cluster"]   = kmeans.predict(pcs_val)

# --- Scaling physicochemical features ---
scaler = StandardScaler()
X_train[physico_features] = scaler.fit_transform(X_train[physico_features])
X_val[physico_features]   = scaler.transform(X_val[physico_features])

# --- Remove highly correlated features fp_ (decision based on just training data set) ---
X_train = remove_correlated_fp_features(X_train, threshold=0.9)
X_val   = X_val[X_train.columns]  # mantener solo columnas finales de train

# --- Log-transform ---
cte = 1e-9
y_train = np.log10(y_train + cte)
y_val   = np.log10(y_val + cte)

if "Concentration" in X_train.columns:
    X_train["Concentration"] = np.log10(X_train["Concentration"] + cte)
    X_val["Concentration"]   = np.log10(X_val["Concentration"] + cte)

# --- Save ---
os.makedirs(RESULTS_DIR, exist_ok=True)

X_train.to_csv(os.path.join(RESULTS_DIR, "X_train.csv"), index=False)
X_val.to_csv(os.path.join(RESULTS_DIR, "X_val.csv"), index=False)
y_train.to_csv(os.path.join(RESULTS_DIR, "y_train.csv"), index=False)
y_val.to_csv(os.path.join(RESULTS_DIR, "y_val.csv"), index=False)