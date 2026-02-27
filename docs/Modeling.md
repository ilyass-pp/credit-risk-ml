# 🤖 Modélisation  — Machine Learning
> **Notebook :** `Modeling.ipynb`

> **Objectif :** Prédire le défaut de paiement sur carte de crédit à partir d'un dataset financier fortement déséquilibré.

---

## 1. Pipeline de Modélisation

```
Données Brutes
    ↓
ColumnTransformer (prétraitement)
    ↓
Régression Logistique (class_weight="balanced")
    ↓
Prédiction de Probabilités
    ↓
Seuil de Décision Optimisé
    ↓
Classe Prédite
```

---

## 2. Performance Initiale — Baseline (Seuil = 0.5)

| Métrique | Score |
|---|:---:|
| F1-score | 0.774 |
| Précision | 0.65 |
| Rappel | 0.96 |
| ROC-AUC | **0.994** |

> Le modèle baseline détecte très bien les défauts mais génère de nombreux faux positifs en raison de la précision faible.

### 📊 Validation Croisée — 5 Folds Stratifiés

| Fold | F1 |
|---|:---:|
| Fold 1 | 0.7483 |
| Fold 2 | 0.7630 |
| Fold 3 | 0.7563 |
| Fold 4 | 0.7633 |
| Fold 5 | 0.7657 |
| **Moyenne** | **0.7593** |

> La stabilité inter-folds confirme l'absence de surapprentissage.

---

## 3. SMOTE — Suréchantillonnage

Une variante avec `SMOTE` a été testée pour compenser le déséquilibre des classes :

| Modèle | F1-score |
|---|:---:|
| Baseline (`class_weight="balanced"`) | 0.7744 |
| SMOTE + Régression Logistique | 0.7784 |

> Le gain apporté par SMOTE est marginal (+0.004). Le paramètre `class_weight="balanced"` s'avère suffisant et plus simple à maintenir.

---

## 4. Optimisation du Seuil de Décision

Le seuil de classification par défaut (0.5) n'est pas optimal en contexte déséquilibré.

Le seuil a été optimisé sur les données de validation en maximisant le **F1-score** :

```python
thresholds = np.linspace(0.05, 0.95, 181)
best_threshold = 0.95   →   Best F1 = 0.8599
```

---

## 5. Performance Finale — Seuil Optimisé

| Métrique | Score |
|---|:---:|
| 🎯 F1-score | **0.8599** |
| 🔬 Précision | **0.9508** |
| 🔍 Rappel | 0.7848 |
| 📈 PR-AUC | **0.949** |

---

## 6. Interprétation des Résultats

L'optimisation du seuil améliore considérablement la fiabilité du modèle :

| Aspect | Analyse |
|---|---|
| 🔬 Précision élevée (0.95) | Réduit les investigations de risque inutiles |
| 🔍 Rappel solide (0.78) | Maintient une détection efficace des défauts réels |
| 🏦 Applicabilité | Performances adaptées aux applications réelles de risque crédit |

---

## 7. Analyse Précision-Rappel

```
PR-AUC = 0.949
```

| Interprétation |
|---|
| ✅ Excellente capacité de discrimination sous déséquilibre sévère |
| ✅ La précision reste élevée sur la majorité des niveaux de rappel |
| ✅ Qualité de classement stable et fiable |

> La courbe PR-AUC est particulièrement pertinente ici car l'accuracy serait trompeuse avec un ratio de classes **11:1**.

---

## 8. Artefacts Sauvegardés

Les composants suivants ont été persistés avec `joblib` :

| Fichier | Contenu |
|---|---|
| `credit_default_model.joblib` | Pipeline complet (prétraitement + modèle) |
| `decision_threshold.joblib` | Seuil optimal (0.95) |

> Ces artefacts garantissent la **reproductibilité** et la **prêt-au-déploiement** du système.

---

## ✅ Conclusion

Un système de prédiction de défaut de crédit **prêt pour la production** a été développé avec succès grâce à :

| Composant | Choix |
|---|---|
| 🔒 Prétraitement | Pipeline sans fuite de données (`ColumnTransformer`) |
| ⚖️ Gestion du déséquilibre | `class_weight="balanced"` |
| 🎯 Évaluation | Seuil optimisé sur F1-score |
| ✅ Validation | Cross-validation stratifiée 5-folds |

> **Améliorations futures →** Modèles ensemblistes (Random Forest, XGBoost) et calibration des probabilités.