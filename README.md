# MisShap — imputation avec détection de fiabilité

Structure du projet :

- `misshap/` — le package Python :
  - `data_evaluation.py` — données (chargement, encodage, injection de NaN)
    et évaluation (NRMSE, EvalV2_NRSME, safe_eval_v2) ;
  - `imputations_classiques.py` — moyenne, médiane, MICE, KNN, MissForest ;
  - `shap_imputation.py` — MisShapForest (SHAP) et MisShapIQForest (SHAP-IQ) ;
  - `ncc.py` — Naive Credal Classifier et ncc_imputation ;
  - `credal.py` — briques théoriques du Credal Ensembling (Nguyen et al. 2025)
    et credal_imputation ;
  - `multicritere.py` — front de Pareto (grille adaptative) et sélection
    Lorenz / OWA (paramètre `normaliser`).
- `resultats.ipynb` — le notebook, réduit aux exécutions : remplissage des
  bases et évaluations, variation d'alpha, tests multi-critère.
- `datasets/` — placer ici les CSV (ou définir la variable d'environnement
  `DATA_DIR`).

Lancement : ouvrir `resultats.ipynb` **depuis ce dossier** (les imports
`from misshap import ...` fonctionnent alors sans installation), ou faire
`pip install -r requirements.txt` au préalable.
