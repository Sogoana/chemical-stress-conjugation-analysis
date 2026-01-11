# SNIPPET-PCA selection PC number.
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# Set seed
np.random.seed(42)
random.seed(42)

BASE_DIR = "data/raw"

# Load data
cluster_df = pd.read_csv(os.path.join(BASE_DIR, "cids"))

y_cluster = cluster_df['Fold'].copy()

    # Define feature candidates (sin eliminar maccs todavía)
X_cluster = cluster_df.drop(columns=['Fold']).copy()

    # Split validation and training dataset before PCA
X_train, X_val, y_train, y_val = train_test_split(
    X_cluster, y_cluster, test_size=0.2, random_state=42
)

# ---- PCA sobre training set únicamente ----
maccs_features = [c for c in X_train.columns if c.startswith('maccs_')]

X_train_maccs = X_train[maccs_features].values

pca = PCA().fit(X_train_maccs)

# Varianza acumulada
explained_var = np.cumsum(pca.explained_variance_ratio_)

# Gráfico
plt.plot(range(1, len(explained_var)+1), explained_var, marker='o')
plt.axhline(0.9, color='r', linestyle='--')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza explicada acumulada')
plt.title('Curva PCA sobre maccs (training set)')
plt.show()

# Encuentra el menor K que explica >= 90%
K = np.argmax(explained_var >= 0.9) + 1
print(f"Número recomendado de PCs para explicar >=90%: {K}")