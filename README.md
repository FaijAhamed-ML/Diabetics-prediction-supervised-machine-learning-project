# IT2011 — Progress Review I: Data Preprocessing and EDA
## Diabetes and LifeStyle Dataset

**Module:** IT2011 - Artificial Intelligence and Machine Learning
**Year 2, Semester 1 (2026)** — Faculty of Computing

---

## 1. Project Overview

This deliverable applies data cleaning, preprocessing, and exploratory data analysis (EDA) to
the assigned **Diabetes and LifeStyle Dataset** (97,297 records, 31 columns), covering
demographic, lifestyle, and clinical features used to predict a patient's **diabetes stage**
(`No Diabetes`, `Pre-Diabetes`, `Gestational`, `Type 2`, `Type 1`).

Each group member independently implemented and validated one preprocessing technique, and the
group integrated all five into a single, reproducible pipeline (`2026-Y2-S1-MLB-WEB2G1
-02_pipeline.ipynb`) that
also trains a baseline KNN classifier to confirm the processed data is model-ready.

## 2. Dataset

- **File:** `data/raw/Diabetes_and_LifeStyle_Dataset_.csv`
- **Rows / Columns:** 97,297 x 31
- **Target:** `diabetes_stage` (5 classes, imbalanced — `Type 2` and `Pre-Diabetes` dominate)
- **Feature groups:** demographics (age, gender, ethnicity, education, income, employment),
  lifestyle (smoking, alcohol, physical activity, diet score, sleep, screen time), and clinical
  measurements (BMI, blood pressure, cholesterol panel, glucose, insulin, HbA1c).

## 3. Group Member Roles

| IT Number | Member | Preprocessing Technique | Notebook |
|---|---|---|---|
| `IT25101611` | Thamoddaya W.M.R. | Handling missing data (verification + validated imputer) | `notebooks/IT25101611_Missing_Data_Handling.ipynb` |
| `IT25101557` | Hiruni kawya | Encoding categorical variables (ordinal + one-hot) | `notebooks/IT25101557_Encoding_Categorical_Variables.ipynb` |
| `IT_Number` | Member 3 | Outlier detection & treatment (IQR capping) | `notebooks/IT_Number_Outlier_Removal.ipynb` |
| `IT25101505` | FAIJ AHAMED.ML | Normalization / scaling (StandardScaler) | `notebooks/IT25101505_Normalization_Scaling.ipynb` |
| `IT25101565` | Dharshika.T | Feature engineering & selection (leakage removal, correlation-based selection, new feature) | `notebooks/IT25101565_Feature_Engineering.ipynb` |
| `IT_Number` | Member 6 | Dimensionality reduction (PCA, scree plot, 2D class-separability projection) | `notebooks/IT_Number_Dimensionality_Reduction.ipynb` |


## 4. Pipeline Flow

```
raw CSV
  -> Stage 1: Missing data check & duplicate removal      (Thamoddaya)
  -> Stage 2: Categorical encoding (ordinal + one-hot)     (Hiruni kawya)
  -> Stage 3: Outlier detection & IQR capping              (Member 3)
  -> Stage 4: StandardScaler normalization                 (FAIJ AHAMED-LEAD)
  -> Stage 5: Leakage removal + feature selection/engineering (Dharshika)
  -> Stage 6: Dimensionality reduction / PCA (Member 6, optional alt. representation)
  -> Train/test split -> SMOTE (train only) -> baseline KNN model (on Stage 5 features)
  -> results/outputs/final_processed_dataset.csv
```

Each individual notebook reads the previous stage's checkpoint CSV from `results/outputs/` and
writes its own checkpoint forward, so the six notebooks can also be run independently in order
1 -> 6. `2026-Y2-S1-MLB-WEB2G1-02_pipeline.ipynb` re-implements the same six stages end-to-end in one run and adds
the model-training step, serving as the "integrated" deliverable. Stage 6 (PCA) produces an
optional compressed dataset (`stage6_pca_reduced.csv`); the primary KNN model is trained on
Stage 5's original engineered features to preserve clinical interpretability (see Member 6's
notebook for the reasoning).

## 5. Key Findings (EDA)

- The dataset arrived **clean**: 0 explicit missing values, 0 duplicate rows, 0 implausible
  clinical readings.
- The target class is **imbalanced**: `Type 2` (~60%) and `Pre-Diabetes` (~32%) dominate, while
  `Gestational` (~0.3%) and `Type 1` (~0.1%) are rare — addressed with SMOTE on the training
  split only.
- `hba1c`, `glucose_fasting`, and `glucose_postprandial` are by far the strongest predictors of
  `diabetes_stage`, consistent with clinical diagnostic criteria.
- `triglycerides`, `insulin_level`, and `hba1c` had the most IQR-flagged outliers; these were
  **capped, not deleted**, to avoid removing genuine diabetic patients from the minority classes.
- `diagnosed_diabetes` and `diabetes_risk_score` were dropped as **target-leakage** columns.
- PCA shows variance is fairly spread across components (little redundancy left after Member
  5's pruning); a 2D projection shows `No Diabetes` separates reasonably well while
  `Pre-Diabetes`/`Type 2` overlap heavily, matching the model's confusion pattern below.
- A baseline KNN classifier trained on the fully processed data reached **~72% test accuracy**
  (weighted F1 ≈ 0.74), confirming the pipeline output is usable for modeling. The rare
  `Gestational`/`Type 1` classes remain hard to predict given how few examples exist even after
  SMOTE — a good discussion point for the viva.

All supporting charts are saved in `results/eda_visualizations/`.

## 6. How to Run

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn jupyter

# Run an individual member's notebook (from the notebooks/ folder, in order 1 -> 5)
cd notebooks

# Run the full integrated pipeline (from the 2026-Y2-S1-MLB-WEB2G1-02/ root)
```

## 7. Repository Layout

```
Group_ID/
├── README.md
├── 2026-Y2-S1-MLB-WEB2G1-02_pipeline.ipynb
├── data/
│   ├── raw/                          # assigned dataset
│   └── external/                     # (unused — no external reference data was needed)
├── notebooks/
│   ├── IT25101611_Missing_Data_Handling.ipynb
│   ├── IT25101557_Encoding_Categorical_Variables.ipynb
│   ├── IT_Number_Outlier_Removal.ipynb
│   ├── IT25101505_Normalization_Scaling.ipynb
│   ├── IT25101565_Feature_Engineering.ipynb
│   └── IT_Number_Dimensionality_Reduction.ipynb
└── results/
    ├── eda_visualizations/           # PNG charts referenced above
    ├── logs/                         # (reserved for execution logs)
    └── outputs/                      # stage-by-stage + final processed CSVs
```
