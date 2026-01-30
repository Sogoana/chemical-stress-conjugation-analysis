"""
premodel.py

Purpose
-------
1. Selection of the number of principal components (PCs) for MACCS fingerprints
and preprocessing of chemical descriptors for downstream modeling.
2. Data split: validation and training datasets

Methodological notes
--------------------
- PCA is fitted exclusively on the training set.
- Number of PCs is selected to explain ≥90% cumulative variance.
- MACCS fingerprints are replaced by PCA components.
- KMeans clustering is performed on PCA components (k=2).
- Physicochemical descriptors are standardized (z-score).
- Highly correlated fingerprint features (|r| > 0.9) are removed
  based solely on the training set.
- Log10 transformation is applied to target variable and concentration.

Reproducibility
---------------
- Random seed fixed to 42.
- scikit-learn version: 1.7.0
- KMeans parameters: init="k-means++", n_init=10

Input
-----
- data/processed/unified_df.csv

Output
------
- data/processed/X_train.csv
- data/processed/X_val.csv
- data/processed/y_train.csv
- data/processed/y_val.csv
- results/plots/PCA_MACCs.pdf
"""

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import random
from pathlib import Path


# --- Set seed ---
np.random.seed(42)
random.seed(42)

# --- Figure output ---
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

# --- Load data and define input and output paths ---
OUTDIR_fig = Path("/results/plots")
BASE_DIR = Path("data/processed")

df = pd.read_csv(BASE_DIR / "unified_df.csv")

# -----------------
# STEP 1. Filter concentration = 0.
# -----------------
df = df[df['Concentration'] != 0].copy()

# -----------------
# STEP 2. Define y and X for PCA and data split into training and validation.
# -----------------
X = df.drop(columns=["Fold"]).copy()
y = df["Fold"].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------
# STEP 3. PCA over trainning set using MACCs
# -----------------
maccs_features = [c for c in X_train.columns if c.startswith('maccs_')]
X_train_maccs = X_train[maccs_features].values
pca = PCA().fit(X_train_maccs)

explained_var = np.cumsum(pca.explained_variance_ratio_)
K = np.argmax(explained_var >= 0.9) + 1
print(f"Number of PCs able to explain >=90%: {K}")

# --- Save graphics ---
fig, ax = plt.subplots(figsize=(4.5, 4))

ax.plot(range(1, len(explained_var) + 1), explained_var, marker='o')
ax.axhline(0.9, linestyle='--', color='#d62728', linewidth=1.2)

ax.axvline(K, linestyle=':', color='black', linewidth=1)
ax.scatter(K, explained_var[K-1], color='black', s=40)

x_text = K + 0.6
y_text = 0.9 + 0.03

ax.text(
    K + 0.6, 0.9 + 0.03,
    f"K = {K}",
    color="black",
    fontsize=9,
    ha='left',
    va='bottom',
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8
    )
)

ax.set_xlabel('Number of components')
ax.set_ylabel('Cumulative explained variance')

fig.savefig(OUTDIR_fig / "PCA_MACCs.pdf", bbox_inches="tight")
plt.close(fig)

# --- No. PCs election ---
N_PCS = K

# -----------------
# STEP 4. Function def. Remove correlated Morgan FPs
# -----------------
def remove_correlated_fp_features(df_features, threshold=0.9):
    """
    Removes highly correlated FPs
    """
    fp_cols = [c for c in df_features.columns if c.startswith("fp")]
    if len(fp_cols) == 0:
        return df_features
    # Correlation matrix (FPs)
    corr_matrix = df_features[fp_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Remove columns
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    filtered = df_features.drop(columns=to_drop, errors='ignore')
    return filtered

# --- PCA on MACCS ---
maccs_cols = [c for c in X_train.columns if c.startswith("maccs_")]
if len(maccs_cols) > 0:
    pca = PCA(n_components=min(N_PCS, len(maccs_cols)), svd_solver="full")
    X_train_pca = pca.fit_transform(X_train[maccs_cols])
    X_val_pca   = pca.transform(X_val[maccs_cols])

    # remove original MACCs
    X_train = X_train.drop(columns=maccs_cols).copy()
    X_val   = X_val.drop(columns=maccs_cols).copy()

    for i in range(X_train_pca.shape[1]):
        X_train[f"PC{i+1}"] = X_train_pca[:, i]
        X_val[f"PC{i+1}"]   = X_val_pca[:, i]

# --- Clustering (based on PCs) ---
pcs_train = X_train[[c for c in X_train.columns if c.startswith("PC")]]
pcs_val   = X_val[[c for c in X_val.columns if c.startswith("PC")]]

kmeans = KMeans(n_clusters=2, random_state=42, init="k-means++", n_init=10)
X_train["cluster"] = kmeans.fit_predict(pcs_train)
X_val["cluster"]   = kmeans.predict(pcs_val)

# -----------------
# STEP 5. Scale continuous variables: physicochemical features
# -----------------
physico_features = ['MoWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                    'FractionCSP3', 'NumRotatableBonds', 'RingCount', 'BertzCT', 'Chi0', 'Chi1']
scaler = StandardScaler()
X_train[physico_features] = scaler.fit_transform(X_train[physico_features])
X_val[physico_features]   = scaler.transform(X_val[physico_features])

# --- Remove high-correlated features (decision based just on training set)
X_train = remove_correlated_fp_features(X_train, threshold=0.9)
X_val   = X_val[X_train.columns]  # mantener solo columnas finales de train

# --- Log-transform ---

cte = 1e-2
y_train = np.log10(y_train + cte)
y_val   = np.log10(y_val + cte)

if "Concentration" in X_train.columns:
    X_train["Concentration"] = np.log10(X_train["Concentration"])
    X_val["Concentration"]   = np.log10(X_val["Concentration"])

# --- Save ---
X_train.to_csv(BASE_DIR / "X_train.csv", index=False)
X_val.to_csv(BASE_DIR / "X_val.csv", index=False)
y_train.to_csv(BASE_DIR / "y_train.csv", index=False)
y_val.to_csv(BASE_DIR / "y_val.csv", index=False)