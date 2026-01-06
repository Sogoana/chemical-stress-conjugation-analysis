import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
base_dir = "/project/data"
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
        cluster_folder = "all_data"
        cluster_path = os.path.join(base_dir, cluster_folder)
        os.makedirs(os.path.join(cluster_path, "importances"), exist_ok=True)
        os.makedirs(os.path.join(cluster_path, "coefficients"), exist_ok=True)

        X_train_cluster = X_train.copy()
        y_train_cluster = y_train.copy()
        X_val_cluster = X_val.copy()
        y_val_cluster = y_val.copy()
    else:
        cluster_folder = f"cluster_{cluster_id}"
        cluster_path = os.path.join(base_dir, cluster_folder)
        os.makedirs(os.path.join(cluster_path, "importances"), exist_ok=True)
        os.makedirs(os.path.join(cluster_path, "coefficients"), exist_ok=True)

        # Filtrar por cluster
        X_train_cluster = X_train[X_train["cluster"] == cluster_id].copy()
        y_train_cluster = y_train.loc[X_train_cluster.index]

        X_val_cluster = X_val[X_val["cluster"] == cluster_id].copy()
        y_val_cluster = y_val.loc[X_val_cluster.index]
    # --- definicion de los feature sets ---
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
        # Usar solo features disponibles
        available_features = [f for f in selected_features if f in X_train_cluster.columns]
        if not available_features:
            continue

        X_train_sel = X_train_cluster[available_features].copy()
        X_val_sel   = X_val_cluster[available_features].copy()
        y_train_sel = y_train_cluster
        y_val_sel   = y_val_cluster

        # --- Random Forest con GridSearch ---
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid_search = GridSearchCV(
            rf, param_grid,
            scoring='neg_root_mean_squared_error',
            cv=3, n_jobs=-1
        )
        grid_search.fit(X_train_sel, y_train_sel)
        best_rf = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        y_train_pred_rf = best_rf.predict(X_train_sel)
        y_val_pred_rf = best_rf.predict(X_val_sel)

        train_rmse_rf = np.sqrt(mean_squared_error(y_train_sel, y_train_pred_rf))
        val_rmse_rf = np.sqrt(mean_squared_error(y_val_sel, y_val_pred_rf))
        train_r2_rf = r2_score(y_train_sel, y_train_pred_rf)
        val_r2_rf = r2_score(y_val_sel, y_val_pred_rf)
        overfit_rf = train_r2_rf - val_r2_rf
        # --- Predicted vs. Real plot (Validation set) ---
        plt.figure(figsize=(6, 6))
        plt.scatter(y_val_sel, y_val_pred_rf, alpha=0.6)
        plt.plot(
            [y_val_sel.min(), y_val_sel.max()],
            [y_val_sel.min(), y_val_sel.max()],
            "r--", lw=2
            )
        plt.xlabel("Real Values")
        plt.ylabel("Predicted Values (RF)")
        plt.title(f"Predicted vs. Real - {feature_code}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(cluster_path, "importances", f"pred_vs_real_RF_set_{idx+1}.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()

        # --- Importancias ---
        importance_df = pd.DataFrame({
            'Feature': X_train_sel.columns,
            'Importance': best_rf.feature_importances_
        })
        importance_df.to_csv(
            os.path.join(cluster_path, "importances", f"importances_set_{idx+1}.csv"),
            index=False
        )

        # --- SHAP ---
        if run_shap:
            explainer = shap.TreeExplainer(best_rf)
            shap_values = explainer.shap_values(X_train_sel, check_additivity=False)
            shap_df = pd.DataFrame({
                'Feature': X_train_sel.columns,
                'MeanAbsSHAP': np.abs(shap_values).mean(axis=0)
            })
            shap_df.to_csv(
                os.path.join(cluster_path, "importances", f"shap_values_set_{idx+1}.csv"),
                index=False
            )
        # --- SHAP plots ---
        if run_shap:
            plt.figure()
            shap.summary_plot(
                shap_values, X_train_sel, show=False, plot_type="dot"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(cluster_path, "importances", f"shap_dotplot_set_{idx+1}.png"),
                dpi=300, bbox_inches="tight"
            )
            plt.close()

        # --- Linear Regression ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_val_scaled = scaler.transform(X_val_sel)

        linreg = LinearRegression()
        linreg.fit(X_train_scaled, y_train_sel)
        y_train_pred_lin = linreg.predict(X_train_scaled)
        y_val_pred_lin = linreg.predict(X_val_scaled)

        train_rmse_lin = np.sqrt(mean_squared_error(y_train_sel, y_train_pred_lin))
        val_rmse_lin = np.sqrt(mean_squared_error(y_val_sel, y_val_pred_lin))
        train_r2_lin = r2_score(y_train_sel, y_train_pred_lin)
        val_r2_lin = r2_score(y_val_sel, y_val_pred_lin)
        overfit_lr = train_r2_lin - val_r2_lin

        coef_df = pd.DataFrame({
            'Feature': X_train_sel.columns,
            'Coefficient': linreg.coef_
        })
        coef_df.to_csv(
            os.path.join(cluster_path, "coefficients", f"coefficients_set_{idx+1}.csv"),
            index=False
        )

        # --- Bias ---
        n_train = len(y_train_sel)
        n_val = len(y_val_sel)
        n_total = n_train + n_val
        bias_rf = ((y_train_sel - y_train_pred_rf).sum() + (y_val_sel - y_val_pred_rf).sum()) / n_total
        bias_lr = ((y_train_sel - y_train_pred_lin).sum() + (y_val_sel - y_val_pred_lin).sum()) / n_total

        results.append({
            'Cluster': cluster_folder,
            'Feature_set_index': idx + 1,
            'Feature_set_code': feature_code,
            'Num_features': len(available_features),
            'Total_samples': n_total,
            'Train_samples': n_train,
            'Val_samples': n_val,
            'Best_RF_params': best_params,
            'GridSearch_best_RMSE': -best_score,
            'Train_RMSE_RF': train_rmse_rf,
            'Val_RMSE_RF': val_rmse_rf,
            'Train_R2_RF': train_r2_rf,
            'Val_R2_RF': val_r2_rf,
            'Overfitting_RF': overfit_rf,
            'Train_RMSE_LR': train_rmse_lin,
            'Val_RMSE_LR': val_rmse_lin,
            'Train_R2_LR': train_r2_lin,
            'Val_R2_LR': val_r2_lin,
            'Overfitting_LR': overfit_lr,
            'Bias_RF': bias_rf,
            'Bias_LR': bias_lr
        })

    # Guardar resultados del cluster
    pd.DataFrame(results).to_csv(os.path.join(cluster_path, "summary_results.csv"), index=False)
    results_all_clusters.extend(results)

# Guardar resumen general
pd.DataFrame(results_all_clusters).to_csv(os.path.join(base_dir, "summary_all_clusters.csv"), index=False)
