# chemical-stress-conjugation-analysis
Python-based analysis of extracted RP4 plasmid conjugation data under chemical stress conditions.

This repository contains the code and workflow for the analysis of chemical stresss conjugation effects using molecular chemical descriptors, fingerprints and integrating them into Machine-learning models.
The project is designed to be reproducible, modular and extensible with clear separation between data processing, feature generation, data split, possible clustering and model training / evaluation

# Project structure
chemical-stress-conjugation-analysis/
    README.md
    data/
        raw/
        processed/
    scripts/
        features.py   # Feature calculation (physicochemical, fingerprints, MACCS)
        snippet_PCS.py   # Calculation of number of Principal Components for data_split.py
        data_split.py   # feature dimensionality reduction, Train/validation splitting and clustering
        models.py   # Model training, evaluation, and SHAP analysis
    results/
        all_data/
            importances/   # Feature importance & SHAP outputs
            coefficients/   # Linear model coefficients
            summary_results.csv
        summary_all_clusters.csv
    environment.yml   # Conda environment specification


# Environment Setup
I. This project uses Conda for environment management.
II. Install dependencies included in environment.yml .

# Workflow overview
## I. CID and feature generation
python scripts/features.py
## II. Create unified dataframe
Matrix was created including metadata, numerical response variable and generated features and categorization by use included in doc. MultiGroup.
## III. Data splitting
First. snippet on PCA is run to find the minimum number of principal components (PCs) needed to explain at least 90% of the data. This parameter should be changed in the data_split.py .
Second. python scripts/data_split.py
### IV. Model training and evaluation
python scripts/models.py

# Model information
Models implemented:
- Linear Regression (LR)
- Random Forest Regressor (RF)
- XGBoost Regressor (XGB)

Features used:
- Concentration
- Physicochemical descriptors
- Extended-Connectivity Fingerprints (ECPF4)
- Molecular ACCess System
- Chemical group by use
- Species involved in the experiments (metadata)

NOTE: models are trained on multiple feature-set combinations on the features included above.

# OUTPUT
- Performance metrics (RMSE, R2, Overfitting estimate, Bias)
- Predicted vs. real plots
- Feature importance and SHAP values (RF and XGBoost)
- Coefficients (LR)

Results are saved under:

results/

# Reproducibility
- The randomness introduced by functions is prevented by fixing random seeds (42) when needed.
- Dependencies are pinned via Conda
- Scripts are modular and can be run independently

NOTE: 
The project is currently developed and tested on macOS 14.5 (23F79)
Large transitory data files were not included in the repository

# Author
Ana Rey Sogo

# License
This project is intended. for academic and research use