import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import shap

# -------------------------------
# PARAMETERS
# -------------------------------
BASE_DIR = "data/processed"   # change to your path
RESULTS_DIR = "results"
CLUSTERS = ["all"]  # Add cluster IDs if needed
RUN_MODELS = True
RUN_PLOTS = True
RUN_SHAP = True

# Feature set control: None = run all, or provide 1-based index to run only that set
FEATURE_SET_IDX = None

# -------------------------------
# FUNCTION DEFINITIONS
# -------------------------------

def load_data(base_dir):
    X_train = pd.read_csv(os.path.join(base_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(base_dir, "X_val.csv"))
    y_train = pd.read_csv(os.path.join(base_dir, "y_train.csv")).squeeze()
    y_val = pd.read_csv(os.path.join(base_dir, "y_val.csv")).squeeze()
    return X_train, X_val, y_train, y_val

def make_cluster_dirs(base_dir, cluster_id):
    if cluster_id == "all":
        folder = "all_data"
    else:
        folder = f"cluster_{cluster_id}"
    path = os.path.join(base_dir, folder)
    os.makedirs(os.path.join(path, "importances"), exist_ok=True)
    os.makedirs(os.path.join(path, "coefficients"), exist_ok=True)
    return path, folder

def get_feature_sets(X):
    physico_features = ['MoWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                        'FractionCSP3', 'NumRotatableBonds', 'RingCount',
                        'BertzCT', 'Chi0', 'Chi1']
    fp_features     = [c for c in X.columns if c.startswith("fp")]
    maccs_features  = [c for c in X.columns if c.startswith("PC")]
    Group1_features = ['AM', 'BCIDE', 'FDADD', 'NMAT', 'INDTCHEM', 'PHARMA', 'DISBYP', 'NATPROD']
    Group2_features = ['ACONV', 'ADD', 'ANLGC', 'ANTCAK', 'ANTDEP', 'ANTHIST', 'ARTSW', 'BB',
                       'DISNF', 'EDC', 'FUNG', 'IL', 'INCHEM', 'LR', 'MENP', 'MEONP',
                       'NSAID', 'ORGCHEM', 'PIGM', 'PPP', 'PRES', 'SOLV', 'XRAY', 'RUBBADD']
    Concentration = ['Concentration']
    metadata = ['Species_relation']

    # List of feature sets and labels (simplified example; extend as needed)
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
    return feature_sets, feature_set_labels

def train_model(model_name, X_train, y_train, X_val, y_val):
    results = {}
    if model_name == "RF":
        rf = RandomForestRegressor(random_state=42)
        param_grid = {'n_estimators':[50,100],
                      'max_depth':[5,10]
                      ,'min_samples_split':[2,5],
                      'min_samples_leaf':[1,2]
                      }
        grid = GridSearchCV(rf, param_grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        results.update({'best_params': grid.best_params_, 'grid_best_score': -grid.best_score_})
    elif model_name == "XGB":
        xgb = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1)
        param_grid = {'n_estimators':[100,200],
                      'max_depth':[3,5,7],
                      'learning_rate':[0.01,0.1, 0.2],
                      'subsample':[0.8,1.0],
                      'colsample_bytree':[0.8,1.0]
                      }
        grid = GridSearchCV(xgb, param_grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        results.update({'best_params': grid.best_params_, 'grid_best_score': -grid.best_score_})
    elif model_name == "LR":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        best_model = lr
    else:
        raise ValueError("Unknown model")
    
    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    overfit = train_r2 - val_r2

    results.update({'train_rmse': train_rmse, 'val_rmse': val_rmse,
                    'train_r2': train_r2, 'val_r2': val_r2, 'overfit': overfit,
                    'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred,
                    'model': best_model})
    return results

def plot_predicted_vs_real(y_val, y_val_pred, model_name, feature_code, save_path, log_scale=True):
    plt.figure(figsize=(6,6))
    plt.scatter(y_val, y_val_pred, alpha=0.6)
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
    plt.xlabel("Real Values")
    plt.ylabel(f"Predicted Values ({model_name})")
    plt.title(f"{model_name} Predicted vs. Real - {feature_code}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def compute_shap(model, X_train_sel, save_path_prefix):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_sel, check_additivity=False)
    shap_df = pd.DataFrame({'Feature': X_train_sel.columns, 'MeanAbsSHAP': np.abs(shap_values).mean(axis=0)})
    shap_df.to_csv(save_path_prefix + "_shap.csv", index=False)
    # SHAP dot plot
    plt.figure()
    shap.summary_plot(shap_values, X_train_sel, show=False, plot_type="dot")
    plt.tight_layout()
    plt.savefig(save_path_prefix + "_shap_dot.png", dpi=300, bbox_inches="tight")
    plt.close()

# -------------------------------
# MAIN LOOP
# -------------------------------

def main():
    X_train, X_val, y_train, y_val = load_data(BASE_DIR)
    results_all_clusters = []

    for cluster_id in CLUSTERS:
        cluster_path, cluster_folder = make_cluster_dirs(RESULTS_DIR, cluster_id)

        # Optional: filter cluster-specific rows here
        X_train_cluster = X_train.copy()
        y_train_cluster = y_train.copy()
        X_val_cluster = X_val.copy()
        y_val_cluster = y_val.copy()

        feature_sets, feature_labels = get_feature_sets(X_train_cluster)

        results_cluster = []

        for idx, (features, label) in enumerate(tqdm(zip(feature_sets, feature_labels),
                                                  total=len(feature_sets),
                                                  desc=f"{cluster_folder} Feature Sets")
                                                  ):
            print(idx + 1, label)
            # Skip if a specific feature set index is set
            if FEATURE_SET_IDX is not None and idx+1 != FEATURE_SET_IDX:
                continue
            available_features = [f for f in features if f in X_train_cluster.columns]
            if not available_features:
                continue

            X_train_sel = X_train_cluster[available_features].copy()
            X_val_sel = X_val_cluster[available_features].copy()
            y_train_sel = y_train_cluster
            y_val_sel = y_val_cluster

            # TRAIN MODELS
            for model_name in ["LR", "RF", "XGB"]:
                if RUN_MODELS:
                    res = train_model(model_name, X_train_sel, y_train_sel, X_val_sel, y_val_sel)

                # Save Predicted vs Real plot
                if RUN_PLOTS:
                    plot_path = os.path.join(cluster_path, "importances", f"pred_vs_real_{model_name}_set_{idx+1}.png")
                    plot_predicted_vs_real(y_val_sel, res['y_val_pred'], model_name, label, plot_path, log_scale=True)

                # SHAP / Feature importances
                if model_name in ["RF","XGB"] and RUN_SHAP:
                    compute_shap(res['model'], X_train_sel, os.path.join(cluster_path, "importances", f"{model_name}_set_{idx+1}"))
                elif model_name=="LR":
                    coef_df = pd.DataFrame({'Feature': X_train_sel.columns, 'Coefficient': res['model'].coef_})
                    coef_df.to_csv(os.path.join(cluster_path, "coefficients", f"coefficients_set_{idx+1}.csv"), index=False)

                # Combine results
                results_cluster.append({
                    'Cluster': cluster_folder,
                    'Feature_set_index': idx+1,
                    'Feature_set_code': label,
                    'Model': model_name,
                    **{k:v for k,v in res.items() 
                       if k not in ['y_train_pred','y_val_pred','model']}
                })

        # Save cluster summary
        pd.DataFrame(results_cluster).to_csv(os.path.join(cluster_path, "summary_results.csv"), index=False)
        results_all_clusters.extend(results_cluster)

    # Save overall summary
    pd.DataFrame(results_all_clusters).to_csv(os.path.join(RESULTS_DIR, "summary_all_clusters.csv"), index=False)

if __name__ == "__main__":

    main()
