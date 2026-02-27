# ⚙️ Pipeline de Prétraitement des Données

> **Notebook :** `Preprocessing.ipynb`

> **Objectif :** Construire un pipeline de prétraitement reproductible et sans fuite de données, prêt pour la modélisation.

Pour garantir la reproductibilité et éviter toute fuite de données, le prétraitement a été implémenté avec :

```
ColumnTransformer + Pipeline  →  Scikit-Learn
```

---

## 1. Sélection des Features

Les colonnes suivantes ont été supprimées avant toute modélisation :

| Colonne | Raison |
|---|---|
| `customer_id` | Identifiant — fuite de données |
| `name` | Identifiant — fuite de données |
| `net_yearly_income` | Redondance — corrélation 0.99 avec `credit_limit` |
| `no_of_children` | Redondance — corrélation 0.88 avec `total_family_members` |

---

## 2. Traitement des Variables Numériques

### 📋 Variables Concernées

```
age · no_of_days_employed · total_family_members · migrant_worker
yearly_debt_payments · credit_limit · credit_limit_used(%)
credit_score · prev_defaults · default_in_last_6months
```

---

### Étape 1 — 🩹 Imputation par la Médiane

Gère les valeurs manquantes de façon robuste face aux valeurs aberrantes.

```python
SimpleImputer(strategy="median")
```

---

### Étape 2 — 📉 Transformation Logarithmique

Appliquée via :

```python
FunctionTransformer(log1p)
```

| Objectif |
|---|
| Réduire l'asymétrie des distributions |
| Compresser les valeurs extrêmes |
| Améliorer la stabilité des modèles |

---

### Étape 3 — 📏 Normalisation

```python
StandardScaler()
```

| Garantit |
|---|
| Moyenne nulle |
| Variance unitaire |

> Indispensable pour les modèles linéaires et les algorithmes à descente de gradient.

---

## 3. Traitement des Variables Catégorielles

### 📋 Variables Concernées

```
gender · owns_car · owns_house · occupation_type
```

---

### Étape 1 — 🩹 Imputation par la Valeur la Plus Fréquente

```python
SimpleImputer(strategy="most_frequent")
```

---

### Étape 2 — 🔠 Encodage One-Hot

```python
OneHotEncoder(handle_unknown="ignore")
```

| Bénéfice |
|---|
| ✅ Évite le biais ordinal |
| ✅ Gère les catégories inconnues sans erreur |

---

## 4. Architecture Finale du Pipeline

```
ColumnTransformer
│
├── 🔢 Pipeline Numérique
│   ├── SimpleImputer (médiane)
│   ├── FunctionTransformer (log1p)
│   └── StandardScaler
│
└── 🔤 Pipeline Catégoriel
    ├── SimpleImputer (most_frequent)
    └── OneHotEncoder (handle_unknown="ignore")
```

---

## ✅ Avantages du Pipeline

| Propriété | Détail |
|---|---|
| 🔒 Aucune fuite de données | Fit uniquement sur le train set |
| 🔁 Totalement reproductible | Même transformation à chaque exécution |
| 🔀 Compatible cross-validation | S'intègre directement dans `cross_val_score` |
| 🚀 Prêt pour le déploiement | Exportable avec `joblib` ou `pickle` |
| 🏭 Standard industriel | Conforme aux bonnes pratiques Scikit-Learn |

---

## ✅ Sortie du Pipeline

Le pipeline transforme les données brutes des clients en une **matrice numérique prête pour l'apprentissage automatique**, utilisable directement par n'importe quel classifieur.

> **Prochaine étape →** Entraînement et évaluation des modèles de classification.