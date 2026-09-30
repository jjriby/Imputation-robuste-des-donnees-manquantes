# Imputation robuste des données manquantes : fiabilité par cellule et ensembles crédaux

Le projet étend missForest avec un **critère de fiabilité par cellule** : chaque valeur imputée est conservée ou rejetée selon
- un critère **SHAP** d'ordre 1 (MisShapForest, seuil `alpha`),
- un critère **SHAP-IQ / n-SII** d'ordre 2 (interactions, seuil `tau`),
- un critère **crédal** fondé sur le désaccord des arbres de la forêt (Credal-MissForest, seuil `tau_credal`),
- un critère **NCC** (Naive Credal Classifier, paramètre `s`).

Le réglage du seuil est traité comme un problème **bi-objectif** (erreur d'imputation vs taux de rejet) : front de Pareto par ε-contrainte + Optuna/TPE, puis sélection équitable par dominance de Lorenz et OWA (Rapport, §3.2).

---

## Arborescence

```
.
├── README.md
├── requirements.txt
├── Rapport.pdf                 # rapport de stage
├── resultats.ipynb             # notebook d'exécution (toutes les expériences)
├── contrib/                    # package Python (importé par le notebook)
│   ├── __init__.py             # ré-exporte les fonctions publiques
│   ├── data_evaluation.py      # chargement, encodage, injection de NaN, métriques
│   ├── imputations_classiques.py  # Mean, Median, MICE, KNN, MissForest
│   ├── shap_imputation.py      # MisShapForest (SHAP) et MisShapIQForest (SHAP-IQ)
│   ├── ncc.py                  # Naive Credal Classifier + imputation NCC
│   ├── credal.py               # ensembles crédaux, score crédal, Credal-MissForest
│   ├── credal_reg_credo.py     # variante CREDO pour la régression (credal_imputation2)
│   └── multicritere.py         # front de Pareto (ε-contrainte/Optuna), Lorenz, OWA
└── datasets/                   # jeux de données (CSV / .data)
```

Le notebook fait `from contrib import ...` : le dossier `contrib/` doit être **à côté** du notebook (ou dans le `PYTHONPATH`).

---

## Installation

Testé avec Python 3.11, numpy 2.0.2, scipy 1.17.1, scikit-learn 1.6.1, pandas 2.2, optuna 5.0.0.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook resultats.ipynb
```

Si l'import de `shapiq` échoue avec `RuntimeError: cannot cache function ... no locator available` (cache numba de la dépendance `galois`), définir un dossier de cache accessible en écriture : `export NUMBA_CACHE_DIR=/tmp/numba_cache`.

`optuna` est nécessaire pour la construction du front de Pareto (`contrib/multicritere.py`). Le code fonctionne avec Optuna ≥ 5 (`Trial.set_constraint`) comme avec Optuna 3.x–4.x (`constraints_func` du `TPESampler`).

---

## Données

Les jeux de données sont lus dans `DATA_DIR`, par défaut `./datasets` à côté du notebook. Pour un autre emplacement :

```bash
export DATA_DIR="/chemin/vers/datasets"
```

| Fichier | Utilisation principale |
|---|---|
| `heart.csv`, `parkinsons.data`, `wine.data` | jeux numériques (Chapitre 4) |
| `ecoli.data`, `credit_approval.data`, `dermatology.data` | jeux mixtes (Chapitre 4) ; Ecoli à 50 % = exemple de l'Annexe A |
| `auto_mpg.csv`, `concrete.csv`, `synth_interactions.csv` | critère enrichi n-SII (§4.6) |
| `adult.csv`, `airfoil.csv`, `boston.csv`, `california_housing.csv`, `synth_salaires.csv` | essais complémentaires |

`load_data_set` applique les règles de lecture propres à certains fichiers (séparateurs, colonnes identifiant) ; `label_encoding` encode les colonnes catégorielles en entiers et renvoie leur liste. Les valeurs manquantes sont injectées au hasard, cellule par cellule (MCAR), par `generate_missing(df, missing_rate)`, qui renvoie aussi les vraies valeurs retirées.

---

## Package `contrib`

| Module | Fonctions principales |
|---|---|
| `data_evaluation` | `load_data_set`, `label_encoding`, `generate_missing`, `Eval_NRMSE_ACC`, `EvalV2_NRSME`, `safe_eval_v2`, `compute_nrmse` |
| `imputations_classiques` | `mean_imputation`, `median_imputation`, `mice_imputation`, `knn_imputation`, `miss_forest_imputation(data, max_iterations, categorical_columns)` |
| `shap_imputation` | `shap_imputation(data, categorical_columns, max_iter, alpha)`, `shapiq_imputation(data, categorical_columns, max_iter, tau, max_order=2, ...)` |
| `ncc` | `NaiveCredalClassifier`, `ncc_imputation(data, categorical_columns, max_iter=3, s=1.0, ...)` |
| `credal` | `credal_reliability`, `CredalEnsembleClassifier`, `u65_score`, `credal_imputation(data, categorical_columns, max_iter=10, tau_credal=0.3, distance="sqe", alpha_credal=0.2)` |
| `credal_reg_credo` | `credal_imputation2(...)` (variante CREDO, score crédal en régression) |
| `multicritere` | `pareto_front_min_per_deletion`, `front_pareto_enveloppe`, `plot_front`, `pareto_mask`, `lorenz_vector`, `lorenz_nondominated`, `normaliser_front`, `lorenz_optimal`, `owa_heuristique` |

Les imputeurs avec critère de fiabilité renvoient `(df_filled, missing_idx, n, pct_del)` : tableau imputé, lignes rejetées, compteur, et pourcentage de rejet. L'erreur (NRMSE sur les colonnes numériques, PFC/accuracy sur les colonnes catégorielles) est calculée **sur les seules cellules conservées**.

---

## Notebook `resultats.ipynb`

| Section | Cellules | Contenu |
|---|---|---|
| Setup | 1–3 | imports depuis `contrib`, versions, `DATA_DIR` |
| Script d'exécution / évaluations | 5–7 | comparaison des méthodes (Mean, Median, MICE, KNN, MissForest, SHAP, SHAP-IQ, NCC, Credal) par dataset × taux × répétition → `imputation_results.xlsx` |
| Courbes (rejet, NRMSE) par hyperparamètre | 9–11 | balayage de grilles fixes de `alpha` et `tau_credal` → `imputation_results_rejection_cuvre.xlsx`, figures |
| Front de Pareto & Lorenz | 13–18 | ε-contrainte + Optuna/TPE pour SHAP et Credal, enveloppe inférieure, Pareto, Lorenz (brut et normalisé) |
| Lorenz & OWA | 20–21 | choix final par l'heuristique OWA (intervalle de stabilité le plus large) |

Les boucles d'expériences sauvegardent l'Excel après chaque (dataset, taux, répétition) : une interruption ne fait pas perdre les résultats déjà calculés.

---

## Chaîne multicritère (Rapport §3.2, Algorithme 6)

1. **ε-contrainte.** Pour chaque cible de rejet δ ∈ Δ = {0,05 ; … ; 0,95} :
   min_θ E(θ) s.c. |r(θ) − δ| ≤ ε, résolu par une étude Optuna (échantillonneur TPE, budget T essais), la contrainte étant transmise à TPE. θ ∈ [0, 1] pour `tau_credal`, θ ∈ [0, `hi`] pour `alpha`.
2. **Enveloppe inférieure.** E_env(δ) = min(E_SHAP(δ), E_Credal(δ)) sur les points faisables.
3. **Pareto.** Filtrage des points non dominés sur (rejet, NRMSE).
4. **Lorenz.** Objectifs normalisés min–max sur le front, triés par ordre décroissant puis cumulés ; on garde les vecteurs non dominés.
5. **OWA.** Balayage de w₁ ∈ [½, 1] ; on retient la solution optimale sur le plus large intervalle de w₁.

```python
from contrib import (load_data_set, label_encoding,
                     pareto_front_min_per_deletion, owa_heuristique)
from contrib.multicritere import front_pareto_enveloppe, plot_front

d = load_data_set("datasets/ecoli.data")
d = d.drop(columns=[c for c in ["name", "Sequence_Name"] if c in d.columns])  # identifiants
df, cat = label_encoding(d)
deltas = [round(0.05 * i, 2) for i in range(1, 20)]

rc = pareto_front_min_per_deletion(df, cat, 0.5, deltas, method="credal", tol=0.025, n_trials=30)
rs = pareto_front_min_per_deletion(df, cat, 0.5, deltas, method="shap",   tol=0.025, n_trials=30)

front = front_pareto_enveloppe(rs, rc)       # enveloppe + Pareto + colonne lorenz_opt
plot_front(front, rs, rc)
sol, best = owa_heuristique(front, normaliser=True)
```

### Paramètres de `pareto_front_min_per_deletion`

| Paramètre | Défaut | Rôle |
|---|---|---|
| `method` | `"credal"` | `"credal"` (θ = `tau_credal`) ou `"shap"` (θ = `alpha`) |
| `tol` | `0.05` | ε de la contrainte \|r − δ\| ≤ ε |
| `n_trials` | `30` | budget T d'essais Optuna par δ |
| `n_startup_trials` | `10` | essais aléatoires initiaux de TPE |
| `n_warm_start` | `3` | θ déjà évalués ré-injectés dans l'étude de chaque δ (`0` = études strictement indépendantes, comme l'Algorithme 6) |
| `hi` | `1.0` / `50.0` | borne supérieure de θ (credal / shap) |
| `log_alpha` | `False` | échelle logarithmique pour `alpha` |
| `max_iter` | `1` | itérations de l'imputeur à chaque évaluation |
| `seed` | `0` | graine du masque de NaN et de TPE |

Sortie : un DataFrame avec une ligne par δ (`delta, reject, NRMSE, tau_credal|alpha, feasible, ecart_rejet, n_trials, n_feasible`) ; `res.attrs["evaluations"]` contient toutes les évaluations (θ, rejet, NRMSE). Si aucun essai ne respecte la contrainte pour un δ, le réglage au rejet le plus proche est renvoyé avec `feasible=False` (jamais de NaN).

**Coût.** Au plus |Δ| × T entraînements de forêt par méthode (19 × 30 = 570 avec les valeurs par défaut) ; un cache évite de réentraîner un θ déjà évalué. Sur Ecoli à 50 % (336 lignes, `max_iter=1`), une évaluation prend environ 6 s sur un portable 12 cœurs : compter de l'ordre d'une heure par méthode avec les valeurs par défaut. Pour un premier essai : `n_trials=15, n_startup_trials=5`.

**Plancher de rejet.** Certaines cibles δ basses peuvent être inatteignables (rejet minimal non nul même au seuil le plus permissif, cf. §4.5.3) : elles ressortent avec `feasible=False` et sont écartées par `front_pareto_enveloppe`.

---

## Reproductibilité

- Le masque de valeurs manquantes est tiré avec `np.random.seed(exp)` (numéro de répétition) dans les expériences, et `seed` dans `pareto_front_min_per_deletion` : toutes les méthodes voient exactement les mêmes tableaux corrompus.
- Les forêts utilisent 100 arbres et `random_state=42`.
- Le Chapitre 4 du rapport utilise 10 répétitions par configuration et les taux de 15, 25, 50 et 75 %. Ces valeurs se règlent en tête des cellules 5 et 9 (`missing_rates`, `n_experiments`, `datasets_paths`).
- Pour reproduire l'exemple de l'Annexe A : `DATASET_CMP = "Ecoli"` et `MISSING_CMP = 0.5` dans la cellule 13.

## Correspondance rapport ↔ code

| Rapport | Code |
|---|---|
| §3.1, MisShapForest | `shap_imputation.shap_imputation` |
| §3.1.2, Credal-MissForest, Algorithmes 4–5 | `credal.credal_reliability`, `credal.credal_imputation` |
| §3.2.2, ε-contrainte, Algorithme 6 | `multicritere.pareto_front_min_per_deletion`, `front_pareto_enveloppe` |
| §3.2.3, Lorenz et OWA | `multicritere.lorenz_optimal`, `owa_heuristique` |
| §3.3, SHAP-IQ, Algorithme 7 | `shap_imputation.shapiq_imputation`, `reliability_from_interactions` |
| Chapitre 4, protocole et tableaux | `resultats.ipynb`, cellules 5–11 |
