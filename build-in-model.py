from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------
# Paths & parameters
# ----------------------------
BASE_DIR = Path("data/INPUT")
RUN_SHAP = True
RUN_PLOTS = False

# ----------------------------
# Load data
# ----------------------------
X_train = pd.read_csv(BASE_DIR / "X_train.csv")
X_val   = pd.read_csv(BASE_DIR / "X_val.csv")
y_train = pd.read_csv(BASE_DIR / "y_train.csv").squeeze()
y_val   = pd.read_csv(BASE_DIR / "y_val.csv").squeeze()

results_all = []

# ----------------------------
# Loop over feature sets
# ----------------------------
for idx, (selected_features, feature_code) in enumerate(
    tqdm(zip(feature_sets, feature_set_labels),
         total=len(feature_sets),
         desc="Feature sets")
):
    available = [f for f in selected_features if f in X_train.columns]
    if not available:
        continue

    Xtr = X_train[available]
    Xva = X_val[available]

    row = {
        "Feature_set_index": idx + 1,
        "Feature_set_code": feature_code,
        "Num_features": len(available),
        "Train_samples": len(y_train),
        "Val_samples": len(y_val)
    }

    # ==========================================================
    # LINEAR REGRESSION
    # ==========================================================
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)

    lr = LinearRegression()
    lr.fit(Xtr_s, y_train)

    ytr_lr = lr.predict(Xtr_s)
    yva_lr = lr.predict(Xva_s)

    row.update({
        "Train_RMSE_lr": np.sqrt(mean_squared_error(y_train, ytr_lr)),
        "Val_RMSE_lr": np.sqrt(mean_squared_error(y_val, yva_lr)),
        "Train_R2_lr": r2_score(y_train, ytr_lr),
        "Val_R2_lr": r2_score(y_val, yva_lr),
        "Bias_lr": ((y_train - ytr_lr).sum() + (y_val - yva_lr).sum()) /
                   (len(y_train) + len(y_val))
    })

    # ==========================================================
    # RANDOM FOREST
    # ==========================================================
    rf = RandomForestRegressor(random_state=42)
    rf_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10]
    }

    rf_gs = GridSearchCV(
        rf, rf_grid,
        scoring="neg_root_mean_squared_error",
        cv=3, n_jobs=-1
    )
    rf_gs.fit(Xtr, y_train)
    best_rf = rf_gs.best_estimator_

    ytr_rf = best_rf.predict(Xtr)
    yva_rf = best_rf.predict(Xva)

    row.update({
        "Train_RMSE_rf": np.sqrt(mean_squared_error(y_train, ytr_rf)),
        "Val_RMSE_rf": np.sqrt(mean_squared_error(y_val, yva_rf)),
        "Train_R2_rf": r2_score(y_train, ytr_rf),
        "Val_R2_rf": r2_score(y_val, yva_rf),
        "Bias_rf": ((y_train - ytr_rf).sum() + (y_val - yva_rf).sum()) /
                   (len(y_train) + len(y_val)),
        "Best_RF_params": rf_gs.best_params_
    })

    # ==========================================================
    # XGBOOST
    # ==========================================================
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    xgb_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1]
    }

    xgb_gs = GridSearchCV(
        xgb, xgb_grid,
        scoring="neg_root_mean_squared_error",
        cv=3, n_jobs=-1
    )
    xgb_gs.fit(Xtr, y_train)
    best_xgb = xgb_gs.best_estimator_

    ytr_xgb = best_xgb.predict(Xtr)
    yva_xgb = best_xgb.predict(Xva)

    row.update({
        "Train_RMSE_xgb": np.sqrt(mean_squared_error(y_train, ytr_xgb)),
        "Val_RMSE_xgb": np.sqrt(mean_squared_error(y_val, yva_xgb)),
        "Train_R2_xgb": r2_score(y_train, ytr_xgb),
        "Val_R2_xgb": r2_score(y_val, yva_xgb),
        "Bias_xgb": ((y_train - ytr_xgb).sum() + (y_val - yva_xgb).sum()) /
                    (len(y_train) + len(y_val)),
        "Best_XGB_params": xgb_gs.best_params_
    })

    results_all.append(row)

# ----------------------------
# Save unified summary
# ----------------------------
summary_df = pd.DataFrame(results_all)
summary_df.to_csv(BASE_DIR / "summary_results_all_models.csv", index=False)
