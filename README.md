# 💳 Credit Card Default Prediction (Pipeline ML)

# 💳 Credit Risk Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blueviolet)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-success)]()
[![Imbalanced Data](https://img.shields.io/badge/Imbalanced--Data-F1%20Optimized-critical)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Un projet complet de Data Mining et d'apprentissage automatique pour prédire si un client sera en défaut de paiement sur sa carte de crédit (**classification binaire**).  
Le projet suit les **bonnes pratiques industrielles** : prétraitement sans fuite de données, évaluation adaptée au déséquilibre, optimisation du seuil, validation croisée, et artefacts exportables.

---

## 🎯 Objectif

Prédire `credit_card_default` :

- `0` → Pas de défaut
- `1` → Défaut de paiement

Le dataset étant **fortement déséquilibré (~8% de défauts)**, les métriques principales sont :
✅ **F1-score** et ✅ **PR-AUC (Average Precision)**.

---

## 📁 Structure du Projet

```
credit-risk-ml/
│
│   .gitignore
│   main.py
│   README.md
│   requirements.txt
│
├───artifacts/
│       credit_default_model.joblib
│       credit_default_xgb_pipeline.joblib
│       decision_threshold_lr.joblib
│       decision_threshold_xgb.joblib
│       preprocessor.joblib
│       submission.csv
│
├───data/
│       test.csv
│       train (1).csv
│
├───docs/
│       Chargement_Analyse_EDA.md
│       Modeling.md
│       Modeling_Final.md
│       Preprocessing.md
│
└───notebooks/
        Chargement_Analyse_EDA.ipynb
        Modeling.ipynb
        Modeling_Final.ipynb
        Preprocessing.ipynb
```

---

## 📊 Aperçu du Dataset

- Échantillons d'entraînement : **45 528**
- Échantillons de test : **11 383**
- Variable cible : `credit_card_default`

### 🔒 Identifiants Sensibles

Les colonnes suivantes sont exclues de la modélisation pour éviter toute fuite de données et garantir une utilisation éthique :

- `customer_id`
- `name`

---

## 🔎 Résumé de l'EDA (Points Clés)

### ✅ Qualité des Données

Taux de valeurs manquantes faible (**<2%**) — toutes les lignes sont conservées, avec imputation :

- Numériques → médiane
- Catégorielles → valeur la plus fréquente

### ⚖️ Déséquilibre des Classes

- Pas de défaut : **91.88%**
- Défaut : **8.12%**

Ratio ≈ **11:1** → l'accuracy est trompeuse → focus sur **F1** et **PR-AUC**.

### 📉 Asymétrie & Valeurs Aberrantes

Plusieurs variables financières présentent une forte asymétrie droite :
`credit_limit`, `yearly_debt_payments`, `no_of_days_employed`, etc.

Transformation **log1p** appliquée pour réduire l'asymétrie et limiter l'impact des valeurs extrêmes.

### 🔗 Corrélations

Corrélations élevées détectées :

- `credit_limit` ↔ `net_yearly_income` ≈ 0.99
- `total_family_members` ↔ `no_of_children` ≈ 0.88

→ Les variables redondantes ont été supprimées lors du prétraitement.

📌 Rapport EDA complet : `docs/Chargement_Analyse_EDA.md`

---

## ⚙️ Prétraitement (Pipeline Sans Fuite de Données)

Implémenté avec **ColumnTransformer + Pipeline (scikit-learn)**.

### Pipeline Numérique
- Imputation par la médiane
- Transformation `log1p`
- StandardScaler

### Pipeline Catégoriel
- Imputation par la valeur la plus fréquente
- OneHotEncoder (`handle_unknown="ignore"`)

📌 Rapport complet : `docs/Preprocessing.md`

---

## 🤖 Stratégie de Modélisation

Modèles testés avec le même pipeline de prétraitement :

- Régression Logistique (`balanced`)
- Random Forest (`balanced_subsample`)
- **XGBoost (modèle final)**

Objectif principal :
✅ Maximiser le **F1-score**, valider avec le **PR-AUC**, et optimiser le seuil de décision.

📌 Étapes de modélisation : `docs/Modeling.md` · `docs/Modeling_Final.md`

---

## 🏆 Résultats Finaux (XGBoost)

### Comparaison des Modèles (Validation @ seuil = 0.5)

| Modèle | F1@0.5 | Précision@0.5 | Rappel@0.5 | PR-AUC |
|---|:---:|:---:|:---:|:---:|
| Régression Logistique | 0.7744 | 0.6493 | 0.9594 | 0.9490 |
| Random Forest | 0.8567 | 0.8724 | 0.8417 | 0.9575 |
| **XGBoost** | **0.8627** | **0.9310** | 0.8038 | **0.9588** |

### Optimisation du Seuil de Décision (Maximisation du F1)

Le seuil par défaut 0.5 n'est pas optimal avec un dataset déséquilibré.

```
✅ Seuil optimal (XGBoost) = 0.37
```

| Métrique | Score |
|---|:---:|
| F1-score | **0.8693** |
| Précision | 0.8839 |
| Rappel | 0.8552 |

### Matrice de Confusion (XGBoost @ seuil = 0.37)

| | Prédit : Non-Défaut | Prédit : Défaut |
|---|:---:|:---:|
| **Réel : Non-Défaut** | TN = 8 284 | FP = 83 |
| **Réel : Défaut** | FN = 107 | TP = 632 |

### Validation Croisée (StratifiedKFold, k=5)

| Métrique | Score Moyen |
|---|:---:|
| PR-AUC | **0.9519** |
| F1-score | **0.8555** |

---

## 📦 Artefacts Sauvegardés (Reproductibles & Déployables)

| Fichier | Contenu |
|---|---|
| `artifacts/preprocessor.joblib` | Pipeline de prétraitement seul |
| `artifacts/credit_default_model.joblib` | Pipeline complet — Régression Logistique |
| `artifacts/credit_default_xgb_pipeline.joblib` | Pipeline complet — XGBoost |
| `artifacts/decision_threshold_lr.joblib` | Seuil optimal — Régression Logistique |
| `artifacts/decision_threshold_xgb.joblib` | Seuil optimal — XGBoost (0.37) |
| `artifacts/submission.csv` | Prédictions finales sur le test set |

**Logique d'inférence :**
1. Prédire les probabilités avec le pipeline sauvegardé
2. Appliquer le seuil de décision sauvegardé

---

## 🚀 Lancer le Projet

**1. Installer les dépendances**

```bash
pip install -r requirements.txt
```

**2. Exécuter les notebooks dans l'ordre**

```
notebooks/Chargement_Analyse_EDA.ipynb
notebooks/Preprocessing.ipynb
notebooks/Modeling.ipynb
notebooks/Modeling_Final.ipynb
```

**3. Soumission finale**

```
artifacts/submission.csv
```

---

## ✅ Utilisation Éthique des Données

- Les identifiants sensibles sont exclus de toutes les étapes de modélisation.
- Aucun profilage personnel direct n'est effectué.
- Le projet est strictement orienté prédiction du risque crédit à des fins éducatives.

---

## 👨‍💻 Auteur

**Ilyass Ait Cheikh**
