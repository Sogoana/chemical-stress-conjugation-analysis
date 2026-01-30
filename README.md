# chemical-stress-conjugation-analysis
Python-based analysis of extracted RP4 plasmid conjugation data under chemical stress conditions.

This repository contains the code and workflow for the analysis of chemical stresss conjugation effects using molecular chemical descriptors, fingerprints and integrating them into Machine-learning models.
The project is designed to be reproducible, modular and extensible with clear separation between data processing, feature generation, data split, possible clustering and model training / evaluation

## Project status

This repository is under active development.
Results, feature sets, and models may change.
Stable versions used in publications are archived on Zenodo.

# Project structure
```text
chemical-stress-conjugation-analysis/
├── README.md
│
├── data/
│   ├── raw/                    # Raw input data (e.g. original .xlsx files)
│   └── processed/              # Processed datasets (train/validation CSVs)
│
├── scripts/
│   ├── features.py             # Feature calculation (physicochemical, fingerprints, MACCS)
│   ├── data_split.py           # Feature dimensionality reduction, train/validation split, clustering
│   └── models.py               # Model training, evaluation, and SHAP analysis
│
├── results/
│   ├── all_data/                # Data set used (i.e., all data, Cluster X)
│   │   ├── importances/         # Feature importance and SHAP outputs
│   │   ├── coefficients/        # Linear model coefficients
│   │   └── summary_results.csv  # All data combining, if used, all different data sets
│
└── environment.yml              # Conda environment specification
```

# Environment Setup
```text
I. This project uses Conda for environment management.
II. Install dependencies included in environment.yml .
```

# Workflow overview
### I. CID and feature generation
Script: scripts/features.py
### II. Create unified dataframe
Matrix was created including metadata, numerical response variable and features obtained from CID (I.). In addition usage multi group was assigned to each CID that will be used as feature input in the model (IV.).
### III. Data splitting
First: snippet on PCA is run to find the minimum number of principal components (PCs) needed to explain at least 90% of the data
Second: dimensionality reduction and data split on training and validation dataset for posterior model training and evaluation (IV). 
Script: scripts/data_split.py .
### IV. Model training and evaluation
Script: scripts/model.py

# Model information
Models implemented:
- Linear Regression (LR)
- Random Forest Regressor (RF)
- eXtreme Gradient Boosting Regressor (XGB)

Features used:
- Concentration
- Physicochemical descriptors
- Extended-Connectivity Fingerprints (ECPF4)
- Molecular ACCess System
- Chemical use category
- Species involved in the experiments (metadata)

NOTE: models are trained on multiple feature-set combinations on the features included above.

# OUTPUT
- Performance metrics (RMSE, R2, Overfitting estimate, Bias)
- Predicted vs. real plots
- Feature importance and SHAP values (RF and XGB)
- Coefficients (LR)

Results are saved under:
```text
results/
```

# Reproducibility
- The randomness introduced by functions is prevented by fixing random seeds (42) when needed.
- Dependencies are pinned via Conda
- Scripts are modular and can be run independently

NOTE: 
The project is currently developed and tested on macOS 14.5 (23F79).

# Author
Ana Rey Sogo

# License
This project is intended for academic and research use only.

- **Code** in this repository is released under the MIT License (see the `LICENSE` file).
- **Data** (extracted conjugation datasets) are licensed under the
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
  license. Please attribute the original sources.