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
group integrated all six into a single, reproducible pipeline (`2026-Y2-S1-MLB-WEB2G1
-02_pipeline.ipynb`) that also trains a baseline KNN classifier to confirm the processed data
is model-ready.

## 2. Dataset

- **File:** `data/raw/Diabetes_and_LifeStyle_Dataset_.csv`
- **Rows / Columns:** 97,297 x 31
- **Target:** `diabetes_stage` (5 classes as provided, severely imbalanced — `Type 2` and
  `Pre-Diabetes` dominate; `Type 1` is dropped during balancing, see Section 5)
- **Feature groups:** demographics (age, gender, ethnicity, education, income, employment),
  lifestyle (smoking, alcohol, physical activity, diet score, sleep, screen time), and clinical
  measurements (BMI, blood pressure, cholesterol panel, glucose, insulin, HbA1c).

## 3. Group Member Roles

| IT Number | Member | Preprocessing Technique | Notebook |
|---|---|---|---|
| `IT25101611` | Thamoddaya W.M.R. | Handling missing data (verification + validated imputer) | `notebooks/IT25101611_Missing_Data_Handling.ipynb` |
| `IT25101557` | Hiruni kawya | Encoding categorical variables (ordinal + one-hot) | `notebooks/IT25101557_Encoding_Categorical_Variables.ipynb` |
| `IT25101492` | Lakshith R. | Outlier detection & treatment (IQR capping) | `notebooks/IT25101492_Outlier_Removal.ipynb` |
| `IT25101505` | FAIJ AHAMED.ML | Normalization / scaling (StandardScaler) & LEAD | `notebooks/IT25101505_Normalization_Scaling.ipynb` |
| `IT25101565` | Dharshika.T | Feature engineering & selection (leakage removal, correlation-based selection, new feature) | `notebooks/IT25101565_Feature_Engineering.ipynb` |
| `IT25101588` | Uditha Banuka | Class imbalance handling (hybrid: under-sampling + SMOTE + Tomek-link cleaning) | `notebooks/06.IT25101588_Class_Balancing.ipynb` |


## 4. Pipeline Flow

```
raw CSV
  -> Stage 1: Missing data check & duplicate removal          (Thamoddaya)
  -> Stage 2: Categorical encoding (ordinal + one-hot)         (Hiruni kawya)
  -> Stage 3: Outlier detection & IQR capping                  (Lakshith)
  -> Stage 4: StandardScaler normalization                     (FAIJ AHAMED-LEAD)
  -> Stage 5: Leakage removal + feature selection/engineering  (Dharshika)
  -> Stage 6: Class imbalance handling — drop Type 1, then     (Uditha Banuka)
              under-sample majority + SMOTE minorities +
              Tomek-link clean (training split only)
  -> baseline KNN model (trained on Stage 6 balanced split,
     evaluated on Stage 6 untouched test split)
  -> results/outputs/final_processed_dataset.csv
```

Each individual notebook reads the previous stage's checkpoint CSV from `results/outputs/` and
writes its own checkpoint forward, so the six notebooks can also be run independently in order
1 -> 6. `2026-Y2-S1-MLB-WEB2G1-02_pipeline.ipynb` re-implements the same six stages end-to-end
in one run and adds the model-training step, serving as the "integrated" deliverable.

Stage 6 first splits the data (train/test) and then, on the **training split only**:

1. Drops `Type 1` — with only 94 training rows, forcing it up to the majority count
   (46,530) via SMOTE would require ~495x synthetic growth per real example, which mostly
   repeats a handful of directions in feature space rather than adding real signal.
2. Random-under-samples the majority class (`Type 2`) down to the size of the next-largest
   class (`Pre-Diabetes`, ~24,810), instead of using the full majority as the balancing target.
3. SMOTE-oversamples the minority classes (`No Diabetes`, `Gestational`) up to that same
   target.
4. Applies Tomek-link cleaning to remove ambiguous, closely-overlapping class-boundary pairs
   created by the resampling.

This produces `stage6_train_balanced.csv` (4 classes, ~23k–25k rows each) and
`stage6_test_holdout.csv` (untouched, still-imbalanced, 4 classes) — the baseline KNN model
trains on the former and is evaluated on the latter, so reported accuracy stays representative
of the real, imbalanced clinical population. See Member 6's notebook for the full before/after
class-distribution analysis and reasoning.

## 5. Key Findings (EDA)

- The dataset arrived **clean**: 0 explicit missing values, 0 duplicate rows, 0 implausible
  clinical readings.
- The target class is **severely imbalanced** across all 5 classes (training split):
  `Type 2` 46,530, `Pre-Diabetes` 24,810, `No Diabetes` 6,190, `Gestational` 214, `Type 1` 94.
- `hba1c`, `glucose_fasting`, and `glucose_postprandial` are by far the strongest predictors of
  `diabetes_stage`, consistent with clinical diagnostic criteria.
- `triglycerides`, `insulin_level`, and `hba1c` had the most IQR-flagged outliers; these were
  **capped, not deleted**, to avoid removing genuine diabetic patients from the minority classes.
- `diagnosed_diabetes` and `diabetes_risk_score` were dropped as **target-leakage** columns.
- `Type 1` was dropped before balancing — too few real examples (94) to resample without
  manufacturing near-duplicate synthetic data. The remaining 4 classes were balanced with a
  **hybrid** technique (under-sample majority + SMOTE minorities + Tomek-link cleaning) rather
  than plain SMOTE-to-majority, bringing all 4 classes to ~23,000–25,000 training rows each
  with a far smaller synthetic-growth factor for `Gestational` than matching the true majority
  would have required.
- A baseline KNN classifier trained on the balanced split reached **~70% test accuracy**
  (weighted F1 ≈ 0.74) on the untouched, imbalanced 4-class test set. `No Diabetes` recall rose
  sharply (0.90) at the cost of precision (0.47), and `Gestational` remains very hard to
  predict (53 test rows, 214 real training rows) even after balancing — a good discussion
  point for the viva on the limits of resampling versus needing more real minority-class data.

All supporting charts are saved in `results/eda_visualizations/`.

## 6. How to Run

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn jupyter

# Run an individual member's notebook (from the notebooks/ folder, in order 1 -> 6)
cd notebooks

# Run the full integrated pipeline (from the 2026-Y2-S1-MLB-WEB2G1-02/ root)
```

## 7. Repository Layout

```
2026-Y2-S1-MLB-WEB2G1-02/
├── README.md
├── 2026-Y2-S1-MLB-WEB2G1-02_pipeline.ipynb
├── data/
│   ├── raw/                          # assigned dataset
│   └── external/                     # (unused — no external reference data was needed)
├── notebooks/
│   ├── 01.IT25101611_Missing_Data_Handling.ipynb
│   ├── 02.IT25101557_Encoding_Categorical_Variables.ipynb
│   ├── 03.IT25101492_Outlier_Removal.ipynb
│   ├── 04.IT25101505_Normalization_Scaling.ipynb
│   ├── 05.IT25101565_Feature_Engineering.ipynb
│   └── 06.IT25101588_Class_Balancing.ipynb
└── results/
    ├── eda_visualizations/           # PNG charts referenced above
    ├── logs/                         # (reserved for execution logs)
    └── outputs/                      # stage-by-stage + final processed CSVs
```
