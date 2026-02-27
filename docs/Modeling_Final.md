# 🤖 Modélisation Finale — Credit Card Default Prediction
> **Notebook :** `Modeling_Final.ipynb`

> **Objectif :** Prédire si un client sera en défaut de paiement (`credit_card_default` ∈ {0, 1}) sur un dataset fortement déséquilibré (~8% de défauts).

---

## 1. Découpage des Données

Un split stratifié a été utilisé pour préserver le ratio de défauts dans les deux ensembles :

| Ensemble | Taille | Ratio de défauts |
|---|:---:|:---:|
| 🟦 Entraînement | 36 422 lignes | ~0.0812 |
| 🟩 Validation | 9 106 lignes | ~0.0812 |

---

## 2. Prétraitement — Sans Fuite de Données

Un unique `ColumnTransformer` a été appliqué à tous les modèles.

### 🔢 Pipeline Numérique

| Étape | Méthode |
|---|---|
| Imputation | `SimpleImputer(strategy="median")` |
| Transformation | `log1p` — réduit l'asymétrie et l'impact des valeurs extrêmes |
| Normalisation | `StandardScaler()` |

### 🔤 Pipeline Catégoriel

| Étape | Méthode |
|---|---|
| Imputation | `SimpleImputer(strategy="most_frequent")` |
| Encodage | `OneHotEncoder(handle_unknown="ignore")` |

> Les identifiants sensibles (`customer_id`, `name`) ont été exclus de la modélisation.

---

## 3. Comparaison des Modèles (Validation @ seuil = 0.5)

| Modèle | F1 | Précision | Rappel | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Régression Logistique (`balanced`) | 0.7744 | 0.6493 | 0.9594 | 0.9939 | 0.9490 |
| Random Forest (`balanced_subsample`) | 0.8567 | 0.8724 | 0.8417 | 0.9954 | 0.9575 |
| **XGBoost** | **0.8627** | **0.9310** | 0.8038 | **0.9956** | **0.9588** |

> ✅ **XGBoost** obtient le meilleur PR-AUC et le meilleur F1 parmi tous les candidats.

---

## 4. Optimisation du Seuil de Décision

Le seuil par défaut (0.5) n'est pas optimal sur un dataset déséquilibré. Le seuil a été optimisé sur les données de validation en maximisant le **F1-score**.

```
Seuil optimal (XGBoost) → 0.37
```

| Métrique | Score |
|---|:---:|
| 🎯 F1-score | **0.8693** |
| 🔬 Précision | **0.8839** |
| 🔍 Rappel | **0.8552** |

### 🔲 Matrice de Confusion (XGBoost @ seuil = 0.37)

| | Prédit : Non-Défaut | Prédit : Défaut |
|---|:---:|:---:|
| **Réel : Non-Défaut** | TN = 8 284 | FP = 83 |
| **Réel : Défaut** | FN = 107 | TP = 632 |

> Cette configuration offre un excellent équilibre : **peu de fausses alertes** (précision élevée) et **détection efficace des vrais défauts** (rappel élevé).

---

## 5. Validation Croisée — StratifiedKFold (k = 5)

Pour garantir la stabilité et la généralisation du modèle :

### PR-AUC par Fold

| Fold | Score |
|---|:---:|
| Fold 1 | 0.9501 |
| Fold 2 | 0.9497 |
| Fold 3 | 0.9556 |
| Fold 4 | 0.9464 |
| Fold 5 | 0.9575 |
| **Moyenne** | **0.9519** |

### F1-score par Fold

| Fold | Score |
|---|:---:|
| Fold 1 | 0.8509 |
| Fold 2 | 0.8557 |
| Fold 3 | 0.8635 |
| Fold 4 | 0.8466 |
| Fold 5 | 0.8609 |
| **Moyenne** | **0.8555** |

> La faible variance inter-folds confirme l'absence de surapprentissage et la robustesse du modèle.

---

## 6. Artefacts Sauvegardés

| Fichier | Contenu |
|---|---|
| `artifacts/credit_default_xgb_pipeline.joblib` | Pipeline complet (prétraitement + XGBoost) |
| `artifacts/decision_threshold_xgb.joblib` | Seuil optimal (0.37) |
| `artifacts/submission.csv` | Prédictions sur le test set (format requis) |

---

## ✅ Conclusion

Un système de prédiction de défaut de crédit **prêt pour la production** a été développé grâce à :

| Composant | Choix |
|---|---|
| 🔒 Prétraitement | Pipeline sans fuite de données (`ColumnTransformer`) |
| ⚖️ Évaluation déséquilibre | F1-score + PR-AUC |
| 🚀 Modèle | Gradient Boosting avancé (XGBoost) |
| 🎯 Décision | Seuil optimisé sur la validation |
| ✅ Généralisation | Validation croisée stratifiée 5-folds |

> **Améliorations futures →** Calibration des probabilités, stacking de modèles, et optimisation des hyperparamètres par Bayesian Search.