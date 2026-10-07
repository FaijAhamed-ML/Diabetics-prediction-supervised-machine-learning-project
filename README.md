# 2026-Y2-S1-MLB-WEB2G1-02 — Diabetes Stage Prediction
## IT2011 Artificial Intelligence and Machine Learning — Full Project

**Module:** IT2011 — Artificial Intelligence and Machine Learning
**Year 2, Semester 1 (2026)** — Faculty of Computing
**Dataset:** Diabetes and LifeStyle Dataset (97,297 records, 31 columns)

---

## 1. Project Overview

This repository contains the group's complete coursework for IT2011, covering all three
deliverables:

1. **Progress Review I — Data Preprocessing & EDA** (`for Progress/`): six group members each
   implemented and validated one preprocessing technique, integrated into a single reproducible
   pipeline, cleaning and preparing the raw clinical + lifestyle dataset for modelling and
   producing a balanced train/test split.
2. **Progress Review II — Model Implementation & Comparison** (`for final/Final Implementation/`):
   each member trained one individual machine learning model on the Stage 6 processed data, and
   the group compared all six models' results in a shared discussion notebook.
3. **Final Model & GUI Demo** (`for final/finalmodel.ipynb`): the best-performing model family
   (Random Forest) is tuned with GridSearchCV and wrapped in a desktop **Tkinter GUI** that
   lets a user enter patient features (or generate random ones) and get a predicted diabetes
   stage with a confidence score.

The overall goal across all phases is to predict a patient's **diabetes stage**
(`No Diabetes`, `Pre-Diabetes`, `Gestational`, `Type 2` — `Type 1` dropped, see Section 3) from
demographic, lifestyle, and clinical features.

## 2. Repository Layout

```
aimly2s1/
├── README.md                          # this file
├── for Progress/                      # Progress Review I — preprocessing & EDA
│   ├── README.md                      # detailed write-up for this phase
│   ├── 2026-Y2-S1-MLB-WEB2G1-02_pipeline.ipynb   # integrated 6-stage pipeline + baseline KNN
│   ├── data/
│   │   ├── raw/                       # assigned dataset (CSV)
│   │   └── external/                  # (unused)
│   ├── notebooks/                     # one notebook per member, per technique
│   │   ├── 01.IT25101611_Missing_Data_Handling.ipynb
│   │   ├── 02.IT25101557_Encoding_Categorical_Variables.ipynb
│   │   ├── 03.IT25101492_Outlier_Removal.ipynb
│   │   ├── 04.IT25101505_Normalization_Scaling.ipynb
│   │   ├── 05.IT25101565_Feature_Engineering.ipynb
│   │   └── 06.IT25101588_Class_Balancing.ipynb
│   └── results/
│       ├── eda_visualizations/        # charts (missing data, outliers, scaling, correlations…)
│       ├── logs/                      # (reserved)
│       └── outputs/                   # stage-by-stage checkpoint CSVs + final processed dataset
└── for final/                         # Progress Review II + final model & GUI
    ├── finalmodel.ipynb               # tuned Random Forest + Tkinter prediction GUI (Section 5)
    ├── sample_outputs_with_GUI_For_testing/   # screenshots of the GUI being tested
    └── Final Implementation/          # individual models + comparison
        ├── group_comparison.ipynb     # group results table + discussion (see Section 4)
        └── notebooks/                 # one model per member, trained on Stage 6 data
            ├── 01.IT25101505_RandomForest_Model.ipynb
            ├── 02IT25101492._DecisionTree_Model.ipynb
            ├── 03.IT25101588_MLP_DeepLearning_Model.ipynb
            ├── 04.IT25101611_LogisticRegression_Model.ipynb
            ├── 05.IT25101557_KNN_Model.ipynb
            └── 06.IT25101565_KMeans_Clustering_Model.ipynb
```

## 3. Phase 1 — Preprocessing & EDA (`for Progress/`)

Six stages, each implemented and validated by a different member, chained into one pipeline:

```
raw CSV
  -> Stage 1: Missing data check & duplicate removal
  -> Stage 2: Categorical encoding (ordinal + one-hot)
  -> Stage 3: Outlier detection & IQR capping
  -> Stage 4: StandardScaler normalization
  -> Stage 5: Leakage removal + feature selection/engineering
  -> Stage 6: Class imbalance handling (drop Type 1, under-sample majority,
              SMOTE minorities, Tomek-link cleaning — training split only)
  -> baseline KNN model
  -> results/outputs/final_processed_dataset.csv
```

Key findings: the raw data was already clean (no missing values, no duplicates); `hba1c`,
`glucose_fasting`, and `glucose_postprandial` are the strongest predictors of diabetes stage;
the target was severely imbalanced and was rebalanced with a hybrid under-sampling + SMOTE +
Tomek-link approach; a baseline KNN reached ~70% test accuracy (weighted F1 ≈ 0.74) on the
untouched, imbalanced test set.

Full details, member roles, and reasoning are in `for Progress/README.md`.

## 4. Phase 2 — Individual Models & Group Comparison (`for final/Final Implementation/`)

Each member trained one model family on the shared `stage6_train_balanced.csv` /
`stage6_test_holdout.csv` split produced in Phase 1, so all results are directly comparable:

| Member | Model | Notebook |
|---|---|---|
| Ahamed M.L.F. | Random Forest | `01.IT25101505_RandomForest_Model.ipynb` |
| Lakshith R. | Decision Tree | `02IT25101492._DecisionTree_Model.ipynb` |
| Wikramasundara D.G.U.B. | MLP (Deep Learning, Keras/TensorFlow) | `03.IT25101588_MLP_DeepLearning_Model.ipynb` |
| Thamoddaya W.M.R. | Logistic Regression (multinomial) | `04.IT25101611_LogisticRegression_Model.ipynb` |
| Hiruni Kawya | K-Nearest Neighbors | `05.IT25101557_KNN_Model.ipynb` |
| Dharshika T. | K-Means Clustering (unsupervised) | `06.IT25101565_KMeans_Clustering_Model.ipynb` |

**Result ranking (weighted F1 on the untouched, imbalanced test set):**

Random Forest (0.905) > Decision Tree (0.865) > MLP (0.793) > Logistic Regression (0.767) ≈
PCA + Logistic Regression (0.739) > K-Means (unsupervised, Adjusted Rand Index ≈ 0.17)

**Summary of the group discussion** (`group_comparison.ipynb`):
- Non-linear models (Random Forest, Decision Tree, MLP) outperformed the linear baseline,
  confirming the diabetes-stage boundaries are not linear in the clinical feature space.
- The `Gestational` class (only 53 test rows) was the hardest to predict across every model.
- K-Means clusters did not align with the true clinical labels, since diabetes stages are
  defined by threshold ranges on markers like HbA1c rather than naturally separated groupings.
- Suggested future work: gradient-boosted trees (XGBoost/LightGBM), training the MLP on the
  full balanced set with a learning-rate schedule, and collecting more real `Gestational`
  examples rather than relying further on synthetic resampling.

## 5. Phase 3 — Final Model & GUI Demo (`for final/finalmodel.ipynb`)

Because Random Forest ranked first in the Phase 2 comparison, it was selected as the final
model and turned into an interactive demo.

**Model training**
- Loads `stage6_train_balanced.csv` and `stage6_test_holdout.csv` from
  `for Progress/results/outputs/` (target: `diabetes_stage_encoded`; classes
  `0 = No Diabetes`, `1 = Pre-Diabetes`, `2 = Gestational`, `3 = Type 2`).
- Draws a class-balanced training sample of up to **4,000 rows per class** (`random_state=42`)
  to keep tuning fast.
- Tunes a `RandomForestClassifier` with **GridSearchCV** (3-fold CV, scoring = weighted F1) over
  `n_estimators` ∈ {100, 200}, `max_depth` ∈ {10, 20, None}, `max_features` ∈ {sqrt, log2}.
- The best estimator is stored as `rf_tuned` and evaluated with accuracy, weighted F1,
  a classification report and a confusion matrix helper.

**Tkinter GUI — "Diabetes Stage Prediction"**
- One input field per model feature (19 features: age, physical activity, diet score, family /
  hypertension / cardiovascular history, BMI, waist-to-hip ratio, systolic & diastolic BP,
  heart rate, total / HDL / LDL cholesterol, triglycerides, fasting & postprandial glucose,
  insulin level, HbA1c, pulse pressure).
- **Predict Diabetes Stage** button — returns the predicted stage and the model's confidence
  (highest class probability, in %).
- **Generate Random Data** button — fills every field with a random value within the min/max
  range seen in the training data, so every class can be tried quickly.
- Non-numeric input shows an "Invalid input" error dialog.
- **Note:** the model was trained on the preprocessed (standardized) Stage 6 data, so the GUI
  expects **scaled feature values** (e.g. `1.58`, `-0.72`), not raw clinical units.

**Sample outputs:** screenshots of the GUI (empty form, random data, and predictions such as
`Type 2` at 85.50% confidence) are in `for final/sample_outputs_with_GUI_For_testing/`.

## 6. How to Run

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn tensorflow jupyter

# Phase 1 — preprocessing (run in order 1 -> 6, or run the integrated pipeline notebook)
cd "for Progress/notebooks"

# Phase 2 — models (each notebook loads results/outputs/stage6_*.csv from Phase 1)
cd "for final/Final Implementation/notebooks"

# Group comparison & discussion
# open "for final/Final Implementation/group_comparison.ipynb"

# Phase 3 — final tuned model + GUI (run all cells; the Tkinter window opens at the end)
# open "for final/finalmodel.ipynb"
# Requires a desktop environment with Tkinter (bundled with standard Python installs).
# The notebook reads the data via ../for progress/results/outputs/ — keep the folder
# structure from Section 2 (and note the folder-name casing on case-sensitive systems).
```

## 7. Group Member Roles (All Phases)

| IT Number | Member | Preprocessing (PR I) | Model (PR II) |
|---|---|---|---|
| IT25101611 | Thamoddaya W.M.R. | Missing data handling | Logistic Regression |
| IT25101557 | Hiruni Kawya | Encoding categorical variables | K-Nearest Neighbors |
| IT25101492 | Lakshith R. | Outlier detection & treatment | Decision Tree |
| IT25101505 | Ahamed M.L.F. (Lead) | Normalization / scaling | Random Forest (also the final tuned model + GUI) |
| IT25101565 | Dharshika T. | Feature engineering & selection | K-Means Clustering |
| IT25101588 | Wikramasundara D.G.U.B. (Uditha Banuka) | Class imbalance handling | MLP Deep Learning |
