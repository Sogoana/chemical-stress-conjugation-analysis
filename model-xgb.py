import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


# Parameters
base_dir = "/data"
run_shap = True
results_all_clusters = []

# -- Cargar los datos --
X_train = pd.read_csv(os.path.join(base_dir, "X_train.csv"))
X_val   = pd.read_csv(os.path.join(base_dir, "X_val.csv"))
y_train = pd.read_csv(os.path.join(base_dir, "y_train.csv")).squeeze()
y_val   = pd.read_csv(os.path.join(base_dir, "y_val.csv")).squeeze()

# Loop por clusters
for cluster_id in ["all"]:
    if cluster_id == "all":
        cluster_folder = "all_data_xgb"
        cluster_path = os.path.join(base_dir, cluster_folder)
        os.makedirs(os.path.join(cluster_path, "importances"), exist_ok=True)

        X_train_cluster = X_train.copy()
        y_train_cluster = y_train.copy()
        X_val_cluster = X_val.copy()
        y_val_cluster = y_val.copy()
    else:
        cluster_folder = f"cluster_{cluster_id}_xgb"
        cluster_path = os.path.join(base_dir, cluster_folder)
        os.makedirs(os.path.join(cluster_path, "importances"), exist_ok=True)

        # Filtrar por cluster
        X_train_cluster = X_train[X_train["cluster"] == cluster_id].copy()
        y_train_cluster = y_train.loc[X_train_cluster.index]

        X_val_cluster = X_val[X_val["cluster"] == cluster_id].copy()
        y_val_cluster = y_val.loc[X_val_cluster.index]

    # --- definir feature sets (los mismos que en tu script original) ---
    physico_features = ['MoWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                        'FractionCSP3', 'NumRotatableBonds', 'RingCount',
                        'BertzCT', 'Chi0', 'Chi1']
    fp_features     = [c for c in X_train_cluster.columns if c.startswith("fp")]
    maccs_features  = [c for c in X_train_cluster.columns if c.startswith("PC")]
    Group1_features = ['AM', 'BCIDE', 'FDADD', 'NMAT', 'INDTCHEM', 'PHARMA', 'DISBYP', 'NATPROD']
    Group2_features = ['ACONV', 'ADD', 'ANLGC', 'ANTCAK', 'ANTDEP', 'ANTHIST', 'ARTSW', 'BB',
                       'DISNF', 'EDC', 'FUNG', 'IL', 'INCHEM', 'LR', 'MENP', 'MEONP',
                       'NSAID', 'ORGCHEM', 'PIGM', 'PPP', 'PRES', 'SOLV', 'XRAY', 'RUBBADD']
    Concentration = ['Concentration']
    metadata = ['Species_relation']

     # --- lista de feature sets y labels (48 combinaciones) ---
    feature_sets = [
        Concentration + Group1_features,
        Concentration + Group1_features + physico_features,
        Concentration + Group1_features + fp_features,
        Concentration + Group1_features + maccs_features,
        Concentration + Group1_features + physico_features + fp_features,
        Concentration + Group1_features + physico_features + maccs_features,
        Concentration + Group1_features + fp_features + maccs_features,
        Concentration + Group1_features + physico_features + fp_features + maccs_features,

        Concentration + Group2_features,
        Concentration + Group2_features + physico_features,
        Concentration + Group2_features + fp_features,
        Concentration + Group2_features + maccs_features,
        Concentration + Group2_features + physico_features + fp_features,
        Concentration + Group2_features + physico_features + maccs_features,
        Concentration + Group2_features + fp_features + maccs_features,
        Concentration + Group2_features + physico_features + fp_features + maccs_features,

        Concentration + Group1_features + Group2_features,
        Concentration + Group1_features + Group2_features + physico_features,
        Concentration + Group1_features + Group2_features + fp_features,
        Concentration + Group1_features + Group2_features + maccs_features,
        Concentration + Group1_features + Group2_features + physico_features + fp_features,
        Concentration + Group1_features + Group2_features + physico_features + maccs_features,
        Concentration + Group1_features + Group2_features + fp_features + maccs_features,
        Concentration + Group1_features + Group2_features + physico_features + fp_features + maccs_features,

        Concentration + Group1_features + metadata,
        Concentration + Group1_features + physico_features + metadata,
        Concentration + Group1_features + fp_features + metadata,
        Concentration + Group1_features + maccs_features + metadata,
        Concentration + Group1_features + physico_features + fp_features + metadata,
        Concentration + Group1_features + physico_features + maccs_features + metadata,
        Concentration + Group1_features + fp_features + maccs_features + metadata,
        Concentration + Group1_features + physico_features + fp_features + maccs_features + metadata,

        Concentration + Group2_features + metadata,
        Concentration + Group2_features + physico_features + metadata,
        Concentration + Group2_features + fp_features + metadata,
        Concentration + Group2_features + maccs_features + metadata,
        Concentration + Group2_features + physico_features + fp_features + metadata,
        Concentration + Group2_features + physico_features + maccs_features + metadata,
        Concentration + Group2_features + fp_features + maccs_features + metadata,
        Concentration + Group2_features + physico_features + fp_features + maccs_features + metadata,

        Concentration + Group1_features + Group2_features + metadata,
        Concentration + Group1_features + Group2_features + physico_features + metadata,
        Concentration + Group1_features + Group2_features + fp_features + metadata,
        Concentration + Group1_features + Group2_features + maccs_features + metadata,
        Concentration + Group1_features + Group2_features + physico_features + fp_features + metadata,
        Concentration + Group1_features + Group2_features + physico_features + maccs_features + metadata,
        Concentration + Group1_features + Group2_features + fp_features + maccs_features + metadata,
        Concentration + Group1_features + Group2_features + physico_features + fp_features + maccs_features + metadata,
        
        Concentration,
        physico_features,
        fp_features,
        maccs_features,

        Concentration + physico_features,
        Concentration + fp_features,
        Concentration + maccs_features,

        Concentration + physico_features + fp_features,
        Concentration + physico_features + maccs_features,
        Concentration + physico_features + fp_features + maccs_features,
        Concentration + fp_features + maccs_features,

        Concentration + metadata,
        physico_features + metadata,
        fp_features + metadata,
        maccs_features + metadata,

        Concentration + physico_features + metadata,
        Concentration + fp_features + metadata,
        Concentration + maccs_features + metadata,

        Concentration + physico_features + fp_features + metadata,
        Concentration + physico_features + maccs_features + metadata,
        Concentration + physico_features + fp_features + maccs_features + metadata,
        Concentration + fp_features + maccs_features + metadata
    ]

    feature_set_labels = [
        'G1',
        'G1 + Physicochemical',
        'G1 + Fps',
        'G1 + PCs',
        'G1 + Physicochemical + Fps',
        'G1 + Physicochemical + PCs',
        'G1 + Fps + PCs',
        'G1 + Physicochemical + Fps + PCs',

        'G2',
        'G2 + Physicochemical',
        'G2 + Fps',
        'G2 + PCs',
        'G2 + Physicochemical + Fps',
        'G2 + Physicochemical + PCs',
        'G2 + Fps + PCs',
        'G2 + Physicochemical + Fps + PCs',

        'G1 + G2',
        'G1 + G2 + Physicochemical',
        'G1 + G2 + Fps',
        'G1 + G2 + PCs',
        'G1 + G2 + Physicochemical + Fps',
        'G1 + G2 + Physicochemical + PCs',
        'G1 + G2 + Fps + PCs',
        'G1 + G2 + Physicochemical + Fps + PCs',

        'G1 + species',
        'G1 + species + Physicochemical',
        'G1 + species + Fps',
        'G1 + species + PCs',
        'G1 + species + Physicochemical + Fps',
        'G1 + species + Physicochemical + PCs',
        'G1 + species + Fps + PCs',
        'G1 + species + Physicochemical + Fps + PCs',

        'G2 + species',
        'G2 + species + Physicochemical',
        'G2 + species + Fps',
        'G2 + species + PCs',
        'G2 + species + Physicochemical + Fps',
        'G2 + species + Physicochemical + PCs',
        'G2 + species + Fps + PCs',
        'G2 + species + Physicochemical + Fps + PCs',

        'G1 + G2 + species',
        'G1 + G2 + species + Physicochemical',
        'G1 + G2 + species + Fps',
        'G1 + G2 + species + PCs',
        'G1 + G2 + species + Physicochemical + Fps',
        'G1 + G2 + species + Physicochemical + PCs',
        'G1 + G2 + species + Fps + PCs',
        'G1 + G2 + species + Physicochemical + Fps + PCs',

        'Concentration',
        'Physicochemical',
        'Fps',
        'PCs',

        'Concentration + physicochemical',
        'Concentration + Fps',
        'Concentration + PCs',

        'Concentration + physicochemical + Fps',
        'Concentration + physicochemical + PCs',
        'Concentration + physicochemical + FPs + PCs',
        'Concentration + Fps + PCs',

        'Concentration + species',
        'Physicochemical + species',
        'Fps + species',
        'PCs + species',

        'Concentration + physicochemical + species',
        'Concentration + Fps + species',
        'Concentration + PCs + species',

        'Concentration + physicochemical + Fps + species',
        'Concentration + physicochemical + PCs + species',
        'Concentration + physicochemical + FPs + PCs + species',
        'Concentration + Fps + PCs + species'
    ]

    results = []

    # Loop sobre cada feature set
    for idx, (selected_features, feature_code) in enumerate(
        tqdm(zip(feature_sets, feature_set_labels),
             desc=f"{cluster_folder} Feature Sets",
             total=len(feature_sets))
    ):
        available_features = [f for f in selected_features if f in X_train_cluster.columns]
        if not available_features:
            continue

        X_train_sel = X_train_cluster[available_features].copy()
        X_val_sel   = X_val_cluster[available_features].copy()
        y_train_sel = y_train_cluster
        y_val_sel   = y_val_cluster

        # --- XGBoost con GridSearch ---
        xgb = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        grid_search = GridSearchCV(
            xgb, param_grid,
            scoring='neg_root_mean_squared_error',
            cv=3, n_jobs=-1
        )
        grid_search.fit(X_train_sel, y_train_sel)
        best_xgb = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # --- predicciones ---
        y_train_pred = best_xgb.predict(X_train_sel)
        y_val_pred = best_xgb.predict(X_val_sel)

        train_rmse = np.sqrt(mean_squared_error(y_train_sel, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val_sel, y_val_pred))
        train_r2 = r2_score(y_train_sel, y_train_pred)
        val_r2 = r2_score(y_val_sel, y_val_pred)
        overfit = train_r2 - val_r2

        # --- Importancias ---
        importance_df = pd.DataFrame({
            'Feature': X_train_sel.columns,
            'Importance': best_xgb.feature_importances_
        })
        importance_df.to_csv(
            os.path.join(cluster_path, "importances", f"importances_set_{idx+1}.csv"),
            index=False
        )
         # --- Bias ---
        n_train = len(y_train_sel)
        n_val = len(y_val_sel)
        n_total = n_train + n_val
        bias_xgb = ((y_train_sel - y_train_pred).sum() + (y_val_sel - y_val_pred).sum()) / n_total

        # --- SHAP ---
        if run_shap:
            explainer = shap.TreeExplainer(best_xgb)
            shap_values = explainer.shap_values(X_train_sel, check_additivity=False)
            shap_df = pd.DataFrame({
                'Feature': X_train_sel.columns,
                'MeanAbsSHAP': np.abs(shap_values).mean(axis=0)
            })
            shap_df.to_csv(
                os.path.join(cluster_path, "importances", f"shap_values_set_{idx+1}.csv"),
                index=False
            )
        # --- Predicted vs. Real plot (Validation set, log scale) ---
        plt.figure(figsize=(6, 6))
        plt.scatter(y_val_sel, y_val_pred, alpha=0.6)
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(
            [y_val_sel.min(), y_val_sel.max()],
            [y_val_sel.min(), y_val_sel.max()],
            "r--", lw=2)
        plt.xlabel("Real Values (log scale)")
        plt.ylabel("Predicted Values (XGB, log scale)")
        plt.title(f"Predicted vs. Real - {feature_code}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(cluster_path, "importances", f"pred_vs_real_XGB_set_{idx+1}.png"),
            dpi=300, bbox_inches="tight"
            )
        plt.close()

        # --- SHAP plots ---
        if run_shap:
            plt.figure()
            shap.summary_plot(
                shap_values, X_train_sel, show=False, plot_type="dot"
                )
            plt.tight_layout()
            plt.savefig(
                os.path.join(cluster_path, "importances", f"shap_dotplot_XGB_set_{idx+1}.png"),
                dpi=300, bbox_inches="tight"
                )
            plt.close()

        results.append({
            'Cluster': cluster_folder,
            'Feature_set_index': idx + 1,
            'Feature_set_code': feature_code,
            'Num_features': len(available_features),
            'Best_XGB_params': best_params,
            'GridSearch_best_RMSE_xgb': -best_score,
            'Train_RMSE_xgb': train_rmse,
            'Val_RMSE_xgb': val_rmse,
            'Train_R2_xgb': train_r2,
            'Val_R2_xgb': val_r2,
            'Overfitting_xgb': overfit,
            'Bias_xgb': bias_xgb
        })

    # Guardar resultados del cluster
    pd.DataFrame(results).to_csv(os.path.join(cluster_path, "summary_results_xgb.csv"), index=False)
    results_all_clusters.extend(results)

# Guardar resumen general
pd.DataFrame(results_all_clusters).to_csv(os.path.join(base_dir, "summary_all_clusters_xgb.csv"), index=False)
