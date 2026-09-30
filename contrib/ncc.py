"""NCC (Naive Credal Classifier)"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import KBinsDiscretizer

from .data_evaluation import compute_nrmse


# Naive Credal Classifier
class NaiveCredalClassifier:
    """
    Retourne pour chaque instance l'ensemble des classes non dominées au sens
    des intervalles [P_inf, P_sup], dont la largeur est contrôlée par `s`.

    Paramètres
    ----------
    s : float, 
        Hyperparamètre de l'IDM, plus il est grand, plus les credal sets sont larges.
    n_bins : int,
        Nombre de bins pour la discrétisation des features continues.
    strategy : {'quantile', 'uniform', 'kmeans'},
        Stratégie de discrétisation passée à `KBinsDiscretizer`.
    """

    def __init__(self, s: float = 1.0, n_bins: int = 5,
                 strategy: str = "quantile"):
        self.s = s
        self.Nbins = n_bins
        self.strategy = strategy

    # fit
    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Discrétisation des features
        self.discretiser = KBinsDiscretizer(
            n_bins=self.Nbins, encode="ordinal", strategy=self.strategy
        )
        self.discretiser.fit(X)
        X_disc = self.discretiser.transform(X).astype(int)

        # Statistiques suffisantes IDM
        self.classes = np.unique(y)
        self.N = len(y)
        self.class_count = defaultdict(int)
        self.feature_count = defaultdict(int)
        for xi, yi in zip(X_disc, y):
            self.class_count[yi] += 1
            for j, val in enumerate(xi):
                self.feature_count[(j, val, yi)] += 1
        return self

    # joint intervals
    def joint_probability_intervals(self, x_disc):
        """Calcule les bornes [P_inf, P_sup] de la jointe pour chaque classe."""
        joint_lower, joint_upper = {}, {}
        for c in self.classes:
            n_c = self.class_count[c]
            prior_lower = n_c / (self.N + self.s)
            prior_upper = (n_c + self.s) / (self.N + self.s)
            lik_lower = lik_upper = 1.0
            for j, val in enumerate(x_disc):
                n_jvc = self.feature_count[(j, val, c)]
                lik_lower *= n_jvc / (n_c + self.s)
                lik_upper *= (n_jvc + self.s) / (n_c + self.s)
            joint_lower[c] = prior_lower * lik_lower
            joint_upper[c] = prior_upper * lik_upper
        return joint_lower, joint_upper

    # interval dominance
    def interval_dominance(self, joint_lower, joint_upper):
        """Filtre les classes dominées au sens des intervalles."""
        kept = [
            c for c in self.classes
            if not any(
                joint_lower[c2] > joint_upper[c]
                for c2 in self.classes if c2 != c
            )
        ]
        return kept or [max(self.classes, key=lambda c: joint_upper[c])]

    # predict_set
    def predict_credal_set(self, X):
        """Retourne, pour chaque instance, l'ensemble des classes non dominées."""
        X_disc = self.discretiser.transform(
            np.asarray(X, dtype=float)
        ).astype(int)
        return [
            self.interval_dominance(*self.joint_probability_intervals(xd))
            for xd in X_disc
        ]

    # predict
    def predict(self, X):
        """Prédiction ponctuelle : premier élément du credal set."""
        return np.array(
            [cs[0] for cs in self.predict_credal_set(X)],
            dtype=object,
        )

# Utilitaires
def discretize_target(y_continuous, n_bins: int = 5,
                       strategy: str = "quantile"):
    """Discrétise une cible continue. Retourne (y_disc, discretiser)."""
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    y_disc = disc.fit_transform(y_continuous.reshape(-1, 1)).astype(int).ravel()
    return y_disc, disc

# Pipeline d'imputation
def ncc_imputation(data: pd.DataFrame,
                   categorical_columns,
                   max_iter: int = 3,
                   s: float = 1.0,
                   n_bins_ncc: int = 5,
                   strategy: str = "quantile"):
    """
    Imputation itérative Random Forest + détection de fiabilité par NCC.
 
    'df_filled' est toujours complet (sert d'entraînement) ; 'df_numeric' ne
    perd un NaN que si l'imputation est jugée fiable (credal set singleton).
    Arrêt si la NRMSE moyenne augmente entre deux itérations (MissForest).
 
    Paramètres
    ----------
    data : pandas.DataFrame
        Données avec NaN (variables catégorielles déjà encodées).
    categorical_columns : list[str] | pandas.Index
        Colonnes catégorielles.
    max_iter : int
        Nombre maximal d'itérations externes.
    s : float
        Hyperparamètre IDM du NCC.
    n_bins_ncc : int
        Nombre de bins du 'KBinsDiscretizer'.
    strategy : str
        Stratégie de discrétisation.
 
    Retours
    -------
    df_filled : pandas.DataFrame
        Dataset entièrement imputé.
    missing_rows_index : list[int]
        Lignes contenant au moins une imputation non fiable.
    n_missing_rows : int
        Nombre de lignes non fiables.
    pct_missing_rows : float
        Pourcentage de lignes non fiables.
    """
    data_copy = data.copy()
    numeric_cols = data_copy.select_dtypes(include=[np.number]).columns

    # df_numeric : suit les NaN encore non fiablement imputés
    df_numeric = data_copy[numeric_cols].copy()
    # df_filled : toujours complet — sert d'entraînement aux RF/NCC
    df_filled = df_numeric.copy()
    for col in df_filled.columns:
        df_filled[col] = df_filled[col].fillna(df_filled[col].mean())

    missing_rows_index: list = []
    mean_nrmse_prec = 0
    mean_nrmse = -1

    for iteration in range(max_iter):
        df_filled_prec = df_filled.copy()
        missing_rows_index_prec = list(missing_rows_index)

        # Boucle interne : résolution des dépendances asymétriques
        # On boucle tant que le nombre de NaN dans df_numeric diminue.
        num_cols_prec = -1
        num_missing_cols = -2

        while (num_cols_prec != num_missing_cols) and (num_missing_cols != 0):
            col_missing = df_numeric.isna().sum()
            sorted_columns = col_missing.sort_values().index.tolist()

            for col in sorted_columns:
                missing_mask = df_numeric[col].isnull()
                if missing_mask.sum() == 0:
                    continue
                other_cols = df_numeric.columns.drop(col)

                # Données d'entraînement : lignes où col est observée
                X_train = df_filled.loc[~missing_mask, other_cols].values
                y_train = df_filled.loc[~missing_mask, col].values
                X_pred = df_filled.loc[missing_mask, other_cols]

                if len(X_pred) == 0:
                    continue

                # 1. Random Forest pour l'imputation
                if col in categorical_columns:
                    rf = RandomForestClassifier(
                        n_estimators=100, random_state=42
                    )
                else:
                    rf = RandomForestRegressor(
                        n_estimators=100, random_state=42
                    )
                rf.fit(X_train, y_train)
                preds = rf.predict(X_pred.values)

                # 2. NCC pour la fiabilité
                # Cible : déjà discrète si catégorielle, sinon discrétisation
                if col in categorical_columns:
                    y_train_ncc = y_train.astype(int)
                else:
                    try:
                        y_train_ncc, _ = discretize_target(
                            y_train, n_bins=n_bins_ncc, strategy=strategy
                        )
                    except Exception:
                        # Discrétisation impossible (cible quasi-constante)
                        y_train_ncc = None

                ncc_ok = False
                credal_sets = None
                if y_train_ncc is not None and len(np.unique(y_train_ncc)) >= 2:
                    try:
                        ncc = NaiveCredalClassifier(
                            s=s, n_bins=n_bins_ncc, strategy=strategy
                        )
                        ncc.fit(X_train, y_train_ncc)
                        credal_sets = ncc.predict_credal_set(X_pred.values)
                        ncc_ok = True
                    except Exception:
                        # Échec d'entraînement (ex. pas assez de données)
                        ncc_ok = False

                # 3. Imputation + décision de fiabilité
                for k, idx in enumerate(X_pred.index):
                    pred_val = preds[k]
                    df_filled.loc[idx, col] = pred_val

                    if ncc_ok:
                        is_reliable = (len(credal_sets[k]) == 1)
                    else:
                        # NCC indisponible : on est conservateur (non fiable)
                        is_reliable = False

                    if is_reliable:
                        df_numeric.loc[idx, col] = pred_val
                    # sinon : df_numeric garde NaN

                num_cols_prec = num_missing_cols
                num_missing_cols = df_numeric.isna().sum().sum()

        # Mise à jour des lignes non fiables et critère d'arrêt
        missing_rows_index = df_numeric[
            df_numeric.isna().any(axis=1)
        ].index.tolist()
        nrmse_result = compute_nrmse(df_filled_prec, df_filled)
        mean_nrmse_prec = mean_nrmse
        mean_nrmse = nrmse_result.mean()

        if iteration != 0 and mean_nrmse_prec < mean_nrmse:
            # NRMSE augmente → revenir à l'itération précédente
            df_filled = df_filled_prec.copy()
            missing_rows_index = missing_rows_index_prec
            break

    n_missing = len(missing_rows_index)
    pct_missing = n_missing / len(data_copy) * 100
    return df_filled, missing_rows_index, n_missing, pct_missing
