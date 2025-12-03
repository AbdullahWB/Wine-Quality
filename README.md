<!-- Replace this with your own banner image -->
<p align="center">
  <img src="assets/Feature Selection for Wine Quality Prediction.jpg" alt="Feature Selection for Wine Quality Prediction" width="100%">
</p>

<h1 align="center">Comparative Feature Selection for Wine Quality & Wine Type Prediction</h1>

<p align="center">
  <b>Machine Learning Research Project</b><br>
  Wine Quality • Feature Selection • Model Comparison
</p>

---

## 🍷 Project Overview

This repository contains my research work on **comparing feature selection techniques** for:

- **Wine quality prediction** using physicochemical properties of red wine, and
- **Wine type (cultivar) classification** using classic wine chemistry features.

The project is built around two well-known public datasets:

- The **UCI Wine Quality** dataset (red wine), where the goal is to classify wines into **good vs not-good** based on a numeric quality score.
- The **Wine** dataset (often used in scikit-learn), where the goal is to classify wines into **three cultivars** using 13 chemical attributes.

The main goals of this project are:

- ✅ Perform **clean EDA** on both datasets (distribution, correlations, target analysis).
- ✅ Build **baseline models** (Logistic Regression, Random Forest) without feature selection.
- ✅ Integrate and compare **feature selection methods** (e.g. univariate filters like SelectKBest) inside proper ML pipelines.
- ✅ Use **train/test splits, cross-validation, and evaluation metrics** (Accuracy, F1-macro, confusion matrices) in a reproducible way.
- ✅ Lay the foundation for a **conference-style comparative study** of feature selection methods on wine data.

This is a **research/learning project**, not a production recommendation system. The focus is on **methodology quality**: clean pipelines, fair comparisons, and reproducible experiments.

---

## 📦 Datasets

> ⚠️ Raw data files are **not** included in this repository (respecting dataset licenses).  
> To reproduce the experiments, download the datasets from their official sources and place them in the expected folders.

### 1️⃣ UCI Red Wine Quality

- Source: UCI Machine Learning Repository – _Wine Quality Data Set_
- Samples: 1,599 red wines
- Features: 11 physicochemical attributes (acidity, sulphates, alcohol, etc.)
- Original Target: integer quality score (0–10, typically 3–8) assigned by human experts
- In this project:
  - Target is converted to **binary**: `good = 1 if quality >= 7 else 0`
  - Used for **binary classification** (good vs not-good)

### 2️⃣ Classic Wine Dataset (Cultivar Classification)

- Origin: Wine recognition dataset originally from the UCI repository, commonly accessed through scikit-learn as `load_wine()`.
- Samples: 178 wines
- Features: 13 continuous attributes (alcohol, malic acid, ash, magnesium, phenols, color intensity, hue, etc.)
- Target: 3 classes corresponding to different wine cultivars
- In this project:
  - Target column is **`target`**
  - Used for **multiclass classification**

---

## 🧪 Methodology

The project is organized in **two main phases**, each implemented in separate Jupyter notebooks.

### 1️⃣ Exploratory Data Analysis (EDA)

Notebook: `01_eda.ipynb`

Key steps:

- Load **both datasets** and inspect:
  - Shape, column names, data types
  - Basic statistics (mean, std, min, max)
  - Class distributions (quality levels, good vs not-good, cultivar classes)
- Visualizations:
  - Histograms of features
  - Boxplots to inspect outliers
  - Correlation matrices (heatmaps) to explore relationships between:
    - Features themselves
    - Features and the target (e.g., alcohol vs quality)
- Early observations about:
  - Which features are most correlated with quality or target
  - Skewed distributions or potential outliers
  - Multicollinearity between some chemical measurements

EDA serves as the **foundation** for all later modeling and feature selection decisions.

---

### 2️⃣ Modeling & Feature Selection Experiments

Notebook: `02_experiments.ipynb` (or similar)

This phase implements a **research-style experimental pipeline**:

#### 🔹 Problem Definitions

- **Red wine dataset**:
  - Formulated as **binary classification**: good vs not-good.
- **Wine dataset**:
  - Formulated as **multiclass classification**: 3 cultivars.

#### 🔹 Data Splitting & Preprocessing

- Separate **train/test splits** (e.g., 80/20) with stratification on the target.
- Standardization (`StandardScaler`) applied within **scikit-learn Pipelines** for models that need scaling (e.g., Logistic Regression).
- All transformations are fit **only on the training data** to avoid data leakage.

#### 🔹 Baseline Models (No Feature Selection)

Baseline pipelines include:

- **Logistic Regression** (with scaling)
- **Random Forest Classifier**

For each dataset:

- Evaluate baselines with **Stratified K-Fold cross-validation** (e.g., 5 folds).
- Metrics:
  - Accuracy
  - F1-macro (handles class imbalance more fairly)
- Store results in tidy tables (CSV) for later comparison.

#### 🔹 Hyperparameter Tuning

- Use **GridSearchCV** for each model and dataset:
  - Logistic Regression: `C`, penalty, solver
  - Random Forest: number of trees, max depth, min samples split
- Optimize mainly for **F1-macro** on the training folds.
- Refit on the **full training split** using the best hyperparameters.

#### 🔹 Test Set Evaluation

- Evaluate tuned models on the **held-out test sets** for both datasets.
- Report and visualize:
  - Test Accuracy and F1-macro
  - Classification reports (per-class precision, recall, F1)
  - Confusion matrices (as plots)

These scores are the **reference baselines** for evaluating feature selection methods.

#### 🔹 First Feature Selection Pipeline (SelectKBest)

To begin the comparative feature selection study, the project integrates an initial FS method:

- **Univariate filter**: `SelectKBest` with ANOVA F-score (`f_classif`)
- Pipeline example:
  - `StandardScaler` → `SelectKBest(k)` → `LogisticRegression`
- Hyperparameters:
  - `k` (number of top features to keep) ∈ {3, 5, 7, all}
  - Logistic Regression `C` values

This is evaluated via **GridSearchCV**, just like the baselines, and compared on the test set to see:

- Does feature selection improve performance?
- How many features are actually needed for good predictions?

This simple FS setup is intended as a **starting point** for more advanced methods (e.g., RFE, L1-based selection, tree-based importance, genetic algorithms) in later work.

---

## ⚙️ Tech Stack

- **Language**: Python 3
- **Core Libraries**:
  - `pandas`, `numpy` – data handling
  - `matplotlib`, `seaborn` – visualization
  - `scikit-learn` – modeling, pipelines, feature selection, cross-validation
  - `joblib` – saving models and results

All experiments are designed to run in **Jupyter Notebook**, following a clean, step-by-step structure similar to research code in academic ML papers.

---

## 📂 Repository Structure (example)

> ⚠️ Adjust this section if your actual folder names differ.

```text
.
├── data/
│   ├── WineQuality-RedWine.csv      # UCI red wine quality data (local copy only)
│   └── wine_dataset.csv             # Classic wine dataset (local copy only)
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis for both datasets
│   └── 02_experiments.ipynb         # Modeling, CV, hyperparameter tuning, FS
├── results/
│   ├── baseline_cv_results.csv      # Cross-validation metrics for baselines
│   ├── test_results_tuned_models.csv
│   └── baseline_vs_fs_test_comparison.csv
├── models/
│   ├── red_wine_best_*.joblib       # Best-tuned models (red wine)
│   └── wine_dataset_best_*.joblib   # Best-tuned models (cultivar classification)
├── figures/
│   ├── f1_macro_baseline_vs_fs.png  # Comparison plots
│   └── f1_macro_by_dataset.png
├── assets/
│   └── banner.png                   # Project banner used in README
├── README.md
└── requirements.txt
```
