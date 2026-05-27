# Predictive Analytics for Proactive Disease Management in Public Health

**Course:** DATA 602 — Introduction to Data Analysis and Machine Learning  
**University:** University of Maryland, Baltimore County (UMBC)  
**Team:** Tharun Kumar Molapally · Rahul Reddy Kota · Prabhas Teja Penugonda

---

## Overview

This project builds a predictive analytics pipeline to identify individuals at high risk of chronic disease (diabetes) using real-world public health survey data. The goal is to enable proactive screening and early intervention — shifting healthcare from reactive to preventive.

We follow the **CRISP-DM framework** throughout: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation.

---

## Dataset

**Source:** [CDC Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/index.html)  
**File format:** SAS Transport (.XPT) — `LLCP2024.XPT`  
**Scale:** Large-scale annual public health survey of U.S. adults

Key features used:

| Variable | Description |
|----------|-------------|
| `DIABETE4` | Diabetes status (target) |
| `_BMI5` | Body Mass Index |
| `_AGE_G` | Age group |
| `INCOME3` | Income level |
| `_RFHYPE6` | High blood pressure indicator |
| `_RFCHOL3` | High cholesterol indicator |
| `SMOKDAY2` | Smoking frequency |
| `_TOTINDA` | Physical activity |
| `EXERANY2` | Any exercise in past 30 days |

> **Note:** The BRFSS XPT file is not included in this repo due to its size. Download it from the CDC BRFSS website and set the `BRFSS_XPT_PATH` environment variable or place it in the `data/` folder.

---

## Methodology

### Data Preparation
- Replaced BRFSS missing codes (7, 9, 77, 99, 777, 999) with `NaN`
- Binary target: `has_diabetes` (1 = Yes, 0 = No; prediabetes excluded)
- Median imputation for missing numerical features
- Standardization with `StandardScaler`
- 80/20 train-test split with stratification
- **SMOTE** applied on training data to address class imbalance

### Models Trained
| Model | Notes |
|-------|-------|
| Logistic Regression | Interpretable baseline |
| Random Forest | Ensemble, non-linear relationships |
| Gradient Boosting | Sequential boosting, best performer |

Hyperparameter tuning via **GridSearchCV**.

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (primary metric for imbalanced public health data)
- Precision-Recall curve

---

## Results

**Best Model: Gradient Boosting**

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.758 |
| Average Precision | ~0.353 |

Gradient Boosting achieved the highest discriminative ability across all thresholds, with the best balance of recall and precision — critical for public health screening where missing a high-risk case has real consequences.

### Top Predictive Features (Feature Importance)
1. **Age** — strongest predictor of disease risk
2. **BMI** — significant correlation
3. **Income** — key socioeconomic determinant
4. **Cardiovascular health indicators**

---

## Project Structure

```
Disease-Risk-Prediction-BRFSS/
├── Final_Project_602_Group_8.ipynb   # Full CRISP-DM pipeline notebook
├── report/
│   └── Predictive_Analytics_Disease_Management.pdf
├── README.md
└── requirements.txt
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost pyreadstat
```

---

## How to Run

1. Download `LLCP2024.XPT` from [CDC BRFSS Data](https://www.cdc.gov/brfss/annual_data/annual_data.htm)
2. Place it in the project folder or set: `export BRFSS_XPT_PATH=/path/to/LLCP2024.XPT`
3. Open and run the notebook: `jupyter notebook Final_Project_602_Group_8.ipynb`

---

## Course

DATA 602 — Introduction to Data Analysis and Machine Learning  
Instructor: Devin Fensterheim  
University of Maryland, Baltimore County
