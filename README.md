# Credit Card Transaction Analysis & Fraud Detection System

An end-to-end Machine Learning and Association Rule Mining project for analyzing credit card transactions, detecting fraudulent behavior, comparing algorithms, and serving an interactive web application.

---

## 📁 Project Architecture

```text
CreditCardTransactionAnalysis/
│
├── data/
│   ├── fraudTrain.csv                 # Raw training dataset (1,296,675 records)
│   ├── fraudTest.csv                  # Raw testing dataset (555,719 records)
│   └── processed/
│       ├── fraudTrain_clean.csv       # Preprocessed and engineered training data
│       └── fraudTest_clean.csv        # Preprocessed and engineered testing data
│
├── notebooks/
│   ├── 01_preprocessing.ipynb        # Data wrangling & feature engineering
│   ├── 02_eda.ipynb                  # Exploratory data analysis & distributions
│   ├── 03_machine_learning.ipynb     # Model training & ROC-AUC comparison
│   └── 04_association_mining.ipynb   # Apriori vs FP-Growth rule mining
│
├── outputs/
│   ├── graphs/                       # All 13 exported analytical figures (300 DPI)
│   └── models/
│       ├── logistic_regression.joblib# Trained Logistic Regression pipeline
│       ├── random_forest.joblib      # Trained Random Forest pipeline
│       ├── model_comparison.json     # Quantitative evaluation metrics
│       └── fraud_association_rules.csv # Discovered high-lift fraud rules
│
├── app/
│   ├── app.py                        # Flask server with real-time inference & REST API
│   ├── templates/                    # Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── eda.html
│   │   ├── predict.html
│   │   └── patterns.html
│   └── static/graphs/                # Visual assets for the web dashboard
│
├── 01_preprocessing.py               # Standalone preprocessing script
├── 02_eda.py                         # Standalone EDA & visualization script
├── 03_machine_learning.py            # Standalone ML training & evaluation script
└── 04_association_mining.py          # Standalone association mining script
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
The project requires Python 3.10+ (tested and running on Python 3.14).
```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend flask joblib
```

### 2. Running Individual Pipeline Stages
You can run each stage independently via Python or open the corresponding Jupyter notebooks:

- **Stage 1 — Preprocessing & Feature Engineering**:
  ```bash
  python 01_preprocessing.py
  ```
- **Stage 2 — Exploratory Data Analysis & Plot Generation**:
  ```bash
  python 02_eda.py
  ```
- **Stage 3 — Machine Learning (Logistic Regression vs Random Forest)**:
  ```bash
  python 03_machine_learning.py
  ```
- **Stage 4 — Association Rule Mining (Apriori vs FP-Growth)**:
  ```bash
  python 04_association_mining.py
  ```

### 3. Launching the Web Application
```bash
python app/app.py
```
Open your browser at **`http://127.0.0.1:5000/`**.

---

## 📊 Key Findings & Results

### 1. Data Profile & Imbalance
- **Total Transactions**: 1,852,394 (1,296,675 Train + 555,719 Test)
- **Fraud Incidence**: 0.58% in Train (7,506 cases), confirming extreme class imbalance.

### 2. Feature Engineering Highlights
- **Haversine Distance**: Computed spatial distance between cardholder latitude/longitude and merchant coordinates.
- **Temporal Binning**: Extracted transaction hour, day of week, month, and weekend indicator.
- **PII Scrubbing**: Eliminated raw card numbers (`cc_num`), first/last names, unix timestamps, and street addresses to preserve customer privacy.

### 3. Machine Learning Comparison
| Model | Recall (Fraud) | Precision (Fraud) | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Class-Weighted)** | 72.0% | 2.1% | 0.04 | 0.9013 | Baseline |
| **Random Forest Classifier (Ensemble)** | **83.4%** | **45.9%** | **0.59** | **0.9947** | **Optimal** |

### 4. Association Rule Mining
- **Frequent Itemset Comparison**: Apriori and FP-Growth discovered identical frequent itemsets, but FP-Growth avoids candidate generation.
- **Top Fraud Antecedents**:
  - `cat=grocery_pos, hour=Late_Night` &rarr; `fraud=Yes` (**Lift: 8.76x**)
  - `amt=High, hour=Late_Night` &rarr; `fraud=Yes` (**Lift: 8.69x**)
