# 📊 Data Loading & Exploratory Data Analysis (EDA)

> **Notebook :** `Chargement_Analyse_EDA.ipynb`

> **Objectif :** Charger les données, les explorer et réaliser une analyse exploratoire complète avant toute modélisation.
> **Dataset :** 45 528 clients · 19 colonnes · Target binaire : `credit_card_default`

---

## 1. Dataset Overview

The dataset aims to predict whether a customer will default on credit card payments.

### 📐 Dataset Size

| | Value |
|---|---|
| 🟦 Training samples | **45,528** |
| 🟩 Test samples | **11,383** |
| 🔢 Features | **18 input variables** |
| 🎯 Target variable | `credit_card_default` |

```
credit_card_default
  0 → No Default
  1 → Default
```

### 🔒 Sensitive Identifiers — Excluded

- `customer_id`
- `name`

> These columns were excluded from modeling and analytical steps to prevent data leakage and ensure ethical data usage.

---

## 2. Data Quality Assessment

### 🔍 Missing Values

The dataset presents a **low missing rate (<2%)**, indicating high data quality.

| Feature | Missing % |
|---|:---:|
| `no_of_children` | 1.7% |
| `owns_car` | 1.2% |
| `no_of_days_employed` | 1.0% |
| others | < 0.3% |

### ✅ Decision — Imputation Strategy

Instead of removing samples:

| Type | Strategy |
|---|---|
| Numerical features | **Median Imputation** |
| Categorical features | **Most Frequent Imputation** |

> This preserves dataset size while maintaining statistical robustness.

---

## 3. Target Distribution Analysis

### ⚖️ Class Distribution

| Class | Percentage |
|---|:---:|
| No Default `(0)` | **91.88%** |
| Default `(1)` | **8.12%** |

> The dataset is **highly imbalanced** with a ratio of approximately **11:1**.

### ⚠️ Implication

Accuracy becomes misleading since predicting only the majority class yields high accuracy.

```
✅ Primary evaluation metric → F1-score
```

---

## 4. Feature Distribution Analysis

Histogram analysis revealed:

### 🟢 Well-distributed Variables

| Variable | Status |
|---|---|
| `age` | ✅ Naturally scaled |
| `credit_score` | ✅ Naturally scaled |
| `credit_limit_used(%)` | ✅ Naturally scaled |

---

### 🔴 Highly Skewed Financial Variables

Strong right-skew detected in:

| Variable |
|---|
| `net_yearly_income` |
| `credit_limit` |
| `yearly_debt_payments` |
| `no_of_days_employed` |

> Extreme financial values exist (high-income customers).

### ✅ Decision — Log Transformation

Instead of removing observations:

```python
log1p(x)
```

| Benefit |
|---|
| 📉 Variance stabilization |
| 📈 Improved linear model performance |
| 🔇 Reduced outlier influence |

---

## 5. Outlier Analysis

Boxplot visualization confirmed the presence of financial outliers.

> These represent **realistic economic variability** rather than noise.

### ✅ Decision

Outliers were **retained** and handled via transformation rather than deletion.

---

## 6. Correlation Analysis

### 🚨 Critical Multicollinearity

```
credit_limit  ↔  net_yearly_income  =  0.99
```

This indicates redundant information.

| Action | Variable |
|---|---|
| ✅ Keep | `credit_limit` |
| ❌ Remove | `net_yearly_income` |

---

### 👨‍👩‍👧 Family-related Correlation

```
total_family_members  ↔  no_of_children  =  0.88
```

| Action | Variable |
|---|---|
| ✅ Keep | `total_family_members` |
| ❌ Drop | `no_of_children` |

---

## 7. Strong Predictive Indicators

EDA suggests strong behavioral predictors:

| Feature | Role |
|---|---|
| `prev_defaults` | 🔴 High impact |
| `default_in_last_6months` | 🔴 High impact |
| `credit_score` | 🟠 Strong signal |

> These variables are expected to contribute significantly to prediction performance.

---

## ✅ Conclusion

The dataset is clean, structured, and suitable for machine learning modeling after:

| Step | Action |
|---|---|
| 🪪 Identifiers | Removed (`customer_id`, `name`) |
| ⚖️ Imbalance | Handled via F1-score evaluation |
| 📉 Skewed variables | Transformed with `log1p` |
| 🔗 Multicollinearity | Reduced by dropping correlated features |
| 🔧 Preprocessing | Structured pipeline with `ColumnTransformer` |

> **Next step →** Building a robust preprocessing pipeline using `ColumnTransformer`.