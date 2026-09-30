"""Credal Ensembling"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linprog, minimize
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .data_evaluation import compute_nrmse


# =====================================================================
# Briques théoriques du Credal Ensembling (Nguyen et al. 2025)
# =====================================================================
# 1. DISTANCES ENTRE DISTRIBUTIONS
_EPS = 1e-12


def d_sqe(p, q):
    """Squared Euclidean distance, Eq. (21)."""
    return float(np.sum((p - q) ** 2))


def d_l1(p, q):
    """L1 distance (Minkowski p=1), Eq. (19)."""
    return float(np.sum(np.abs(p - q)))


def d_kl(p, q):
    """KL divergence d_KL(p || q), Eq. (22)."""
    p = np.clip(p, _EPS, 1.0)
    q = np.clip(q, _EPS, 1.0)
    return float(np.sum(p * np.log(p / q)))


def d_cheb(p, q):
    """Chebyshev / L_infty distance, Eq. (20)."""
    return float(np.max(np.abs(p - q)))


def d_js(p, q):
    """Jensen-Shannon distance, Eq. (24)."""
    p = np.clip(p, _EPS, 1.0)
    q = np.clip(q, _EPS, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * (d_kl(p, m) + d_kl(q, m))


DISTANCES = {
    "sqe":  d_sqe,
    "l1":   d_l1,
    "kl":   d_kl,
    "cheb": d_cheb,
    "js":   d_js,
}

# 2. Distribution représentative p* = argmin_{p in simplexe} sum_m d(p, p^m)

def project_simplex(v):
    """Projette 'v' sur le simplexe (Wang & Carreira-Perpinan, 2013)."""
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / np.arange(1, n + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def representative_distribution(P, distance: str = "sqe"):
    """
    Calcule la distribution représentative 'p*' de '{p^1, ..., p^M}'.

    Paramètres
    ----------
    P : ndarray, shape (M, K)
        Distributions prédites par les 'M' membres de l'ensemble (chaque
        ligne somme à 1).
    distance : str
        Clé de 'DISTANCES'.

    Retours
    -------
    p_star : ndarray, shape (K,)
    """
    P = np.asarray(P, dtype=float)
    M, K = P.shape

    # Cas analytique : Squared Euclidean (Proposition 1)
    if distance == "sqe":
        return P.mean(axis=0)

    # Cas général : optimisation convexe sous contraintes
    d_fun = DISTANCES[distance]

    def objective(p):
        return sum(d_fun(p, P[m]) for m in range(M))

    constraints = [{"type": "eq", "fun": lambda p: np.sum(p) - 1.0}]
    bounds = [(0.0, 1.0)] * K
    p0 = P.mean(axis=0)

    res = minimize(
        objective, p0,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 200, "disp": False},
    )
    return project_simplex(res.x)

# 3. CREDAL SET QUANTILE-BASED  CH_alpha(x)
def quantile_credal_set(P, p_star, alpha: float = 0.2,
                        distance: str = "sqe"):
    """
    'H_alpha(x)' : les '(1-alpha)*100 %' prédictions les plus proches de 'p*'.
    'CH_alpha(x)' est leur enveloppe convexe (Eq. 31) ; on renvoie ses points
    extrêmes, ndarray (M_alpha, K).
    """
    M = P.shape[0]
    d_fun = DISTANCES[distance]
    dists = np.array([d_fun(p_star, P[m]) for m in range(M)])
    order = np.argsort(dists)               # du plus proche au plus loin
    M_alpha = max(1, int(np.ceil((1.0 - alpha) * M)))
    return P[order[:M_alpha]]


# 4. RÈGLES DE DÉCISION (perte 0/1)
def maximality_set(H_alpha):
    """
    Classes maximales (non dominées) sous Maximalité, perte 0/1 — Eq. (8)-(9).
 
    'y_bar ≻ y_bar'' ssi 'min_{p in CH} (p_{y_bar} - p_{y_bar'}) > 0' ; ce
    minimum d'une forme linéaire est atteint en un point extrême, d'où le test
    sur 'H_alpha' seul.
    """
    H = np.asarray(H_alpha)
    K = H.shape[1]
    dominated = np.zeros(K, dtype=bool)
    for y_bar in range(K):
        for y_prime in range(K):
            if y_bar == y_prime or dominated[y_prime]:
                continue
            diffs = H[:, y_bar] - H[:, y_prime]
            if np.min(diffs) > 0:
                dominated[y_prime] = True
    return np.where(~dominated)[0]


def bop_01(p):
    """Bayes-optimal prediction pour la perte 0/1 = argmax."""
    return int(np.argmax(p))


def e_admissibility_set(H_alpha, outer=None):
    """
    Classes E-admissibles : à partir d'une approximation extérieure
    (Maximalité par défaut), un LP vérifie pour chaque candidate 'y'
    l'existence de 'p in CH_alpha' dont 'y' est l'argmax — Eq. (35)-(37).
    """
    H = np.asarray(H_alpha)
    M, K = H.shape

    if outer is None:
        outer = maximality_set(H)
    outer = list(outer)

    e_admissible: set = set()

    # Heuristique : argmax sur chaque point extrême
    for m in range(M):
        e_admissible.add(bop_01(H[m]))

    # Pour les classes restantes, on résout un LP de faisabilité
    for y in outer:
        if y in e_admissible:
            continue
        c = np.zeros(M)
        A_ub = np.array([
            H[:, y_prime] - H[:, y]
            for y_prime in range(K) if y_prime != y
        ])
        b_ub = np.zeros(K - 1)
        A_eq = np.ones((1, M))
        b_eq = np.array([1.0])
        bounds = [(0.0, 1.0)] * M

        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.success:
            e_admissible.add(y)

    return np.array(sorted(e_admissible))

# 5. MESURES D'INCERTITUDE ROBUSTIFIÉES  R_S(H(x))
def smallest_margin(p):
    """SM(p) = p_(1) - p_(2). Eq. (39)."""
    s = np.sort(p)[::-1]
    return float(s[0] - s[1])


def confidence_level(p):
    """CL(p) = p_(1). Eq. (C14)."""
    return float(np.max(p))


def entropy(p):
    """Entropie de Shannon (en bits). Eq. (C15)."""
    p = np.clip(p, _EPS, 1.0)
    return float(-np.sum(p * np.log2(p)))


PROBABILISTIC_MEASURES = {
    "sm":      smallest_margin,
    "cl":      confidence_level,
    "entropy": entropy,
}


def robust_uncertainty(P, p_star, S=smallest_margin,
                       distance: str = "sqe"):
    """Robustification 'R_S(H(x))' d'une mesure 'S' — Eq. (44) / Algorithme 1,
    en O(M (K + log M))."""

    P = np.asarray(P, dtype=float)
    M = P.shape[0]
    d_fun = DISTANCES[distance]

    # compute distances {d(p*, p^m) | m in [M]}
    dists = np.array([d_fun(p_star, P[m]) for m in range(M)])

    # construct permutation p* ≻ p^(1) ≻ ... ≻ p^(M)
    order = np.argsort(dists)
    P_sorted = P[order]                          # p^(1), ..., p^(M)

    # initialize R_S = (1/(M+1))(sum_m S(p^m) + S(p*))
    R_S = (sum(S(P_sorted[m]) for m in range(M)) + S(p_star)) / (M + 1.0)

    # initialize R_S_temp = S(p*)
    R_S_temp = S(p_star)

    y_star = bop_01(p_star)

    for m in range(M):
        # p^(m) et p* ont des classes les plus probables différentes
        if bop_01(P_sorted[m]) != y_star:
            R_S = R_S_temp / (M + 1.0)
            break
        R_S_temp = R_S_temp + S(P_sorted[m])

    return R_S

# 6. CLASSIFIEUR HAUT NIVEAU (wrapper sklearn-like)
class CredalEnsembleClassifier:
    """
    Wrapper autour d'un ensemble de classifieurs probabilistes déjà entraînés
    (typiquement les arbres d'un 'RandomForestClassifier').
 
    Paramètres
    ----------
    base_ensemble : estimateur exposant 'estimators_'.
    distance : distance utilisée pour 'p*' et le tri des outliers.
    alpha : fraction d'outliers rejetée pour construire 'CH_alpha'.
    decision_rule : {"bop", "maximality", "e_admissibility"}
        'bop' = prédiction précise (argmax sur 'p*') ; les deux autres
        renvoient des prédictions set-valued.
 
    Attributs
    ---------
    classes_ : classes vues durant 'fit' (héritées de 'base_ensemble').
    """

    def __init__(self, base_ensemble, distance: str = "sqe",
                 alpha: float = 0.2, decision_rule: str = "bop"):
        if distance not in DISTANCES:
            raise ValueError(f"Distance inconnue : {distance}")
        if decision_rule not in {"bop", "maximality", "e_admissibility"}:
            raise ValueError(f"Règle inconnue : {decision_rule}")
        self.base_ensemble = base_ensemble
        self.distance = distance
        self.alpha = alpha
        self.decision_rule = decision_rule

    # Accès aux prédictions des M membres
    def ensemble_proba(self, X):
        """Tenseur '(N, M, K)' des probas des 'M' membres."""
        if not hasattr(self.base_ensemble, "estimators_"):
            raise AttributeError(
                "L'ensemble de base doit exposer `estimators_` "
                "(p. ex. un RandomForestClassifier)."
            )
        estimators = self.base_ensemble.estimators_
        global_classes = self.base_ensemble.classes_
        K = len(global_classes)
        N = X.shape[0]
        M = len(estimators)
        P = np.zeros((N, M, K))
        for m, est in enumerate(estimators):
            proba_m = est.predict_proba(X)
            # Alignement sur les classes globales
            idx = np.searchsorted(global_classes, est.classes_)
            P[:, m, idx] = proba_m
        self.classes_ = global_classes
        return P

    # API utilisateur
    def fit(self, X, y):
        """L'ensemble de base est supposé déjà entraîné ; sinon, 'fit'."""
        from sklearn.utils.validation import check_is_fitted
        try:
            check_is_fitted(self.base_ensemble)
        except Exception:
            self.base_ensemble.fit(X, y)
        self.classes_ = self.base_ensemble.classes_
        return self

    def representative_proba(self, X):
        """Renvoie 'p*(x)' pour chaque 'x'. Shape '(N, K)'."""
        P = self.ensemble_proba(X)
        return np.array([
            representative_distribution(P[i], distance=self.distance)
            for i in range(P.shape[0])
        ])

    def predict_proba(self, X):
        """Alias de 'representative_proba' ."""
        return self.representative_proba(X)

    def credal_set(self, X):
        """Liste des points extrêmes de 'CH_alpha(x)' pour chaque 'x'."""
        P = self.ensemble_proba(X)
        cred = []
        for i in range(P.shape[0]):
            p_star = representative_distribution(P[i], distance=self.distance)
            cred.append(quantile_credal_set(
                P[i], p_star, alpha=self.alpha, distance=self.distance
            ))
        return cred

    def predict(self, X):
        """
        Prédit selon 'decision_rule' : 'ndarray (N,)' de classes si "bop",
        sinon une 'list' de 'N' arrays de classes plausibles.
        """
        P = self.ensemble_proba(X)
        N = X.shape[0]
        results = []
        for i in range(N):
            p_star = representative_distribution(P[i], distance=self.distance)
            if self.decision_rule == "bop":
                results.append(self.classes_[bop_01(p_star)])
            else:
                H_alpha = quantile_credal_set(
                    P[i], p_star, alpha=self.alpha, distance=self.distance
                )
                if self.decision_rule == "maximality":
                    idx = maximality_set(H_alpha)
                else:  # e_admissibility
                    idx = e_admissibility_set(H_alpha)
                results.append(self.classes_[idx])
        if self.decision_rule == "bop":
            return np.array(results)
        return results

    def robust_score(self, X, measure: str | callable = "sm"):
        """
        Calcule 'R_S(H(x))' pour chaque 'x'.

        'measure': "sm", "cl", "entropy" ou un
        callable 'p -> float'.
        """
        S = PROBABILISTIC_MEASURES[measure] if isinstance(measure, str) else measure
        P = self.ensemble_proba(X)
        scores = np.zeros(P.shape[0])
        for i in range(P.shape[0]):
            p_star = representative_distribution(P[i], distance=self.distance)
            scores[i] = robust_uncertainty(
                P[i], p_star, S=S, distance=self.distance
            )
        return scores

# 7. ÉVALUATION : score u65 (Zaffalon et al., 2012)
def u65_score(y_true, y_pred_sets, classes=None):
    """
    Score u65 (Section 6.2.3 du papier) :
    'u_65(|Y|) = -1.2/|Y|^2 + 2.2/|Y|' si 'y in Y_pred', sinon 0
    (1 pour |Y|=1, 0.8 pour |Y|=2, ~0.6 pour |Y|=3).
 
    y_true : array '(N,)' ; y_pred_sets : liste de 'N' arrays ;
    classes : ignoré (compatibilité). Retourne le score moyen.
    """
    n = len(y_true)
    total = 0.0
    for yt, yset in zip(y_true, y_pred_sets):
        size = len(yset)
        if yt in yset:
            total += -1.2 / (size ** 2) + 2.2 / size
    return total / n


# =====================================================================
# Imputation credal (intégration dans le pipeline MissForest)
# =====================================================================
# 1. Score de fiabilite credal pour la CLASSIFICATION

def credal_reliability_classif(rf, x_row, distance="sqe"):
    """
    Score de fiabilite credal pour une instance, cas RandomForestClassifier.
 
    On empile les predict_proba des M arbres en une matrice P (M, K), puis on
    applique l'Algorithme 1 : p* representative + R_SM (Eq. 44).
    R_SM in [0, 1] : proche de 1 = consensus fort, proche de 0 = desaccord.
 
    rf : RandomForestClassifier deja fit
    x_row : pandas Series ou ndarray (1, p)
    distance : distance pour p* (cf. DISTANCES)
    """
    # Mise au format attendu par sklearn (2D)
    if isinstance(x_row, pd.Series):
        x_arr = x_row.values.reshape(1, -1)
    else:
        x_arr = np.asarray(x_row).reshape(1, -1)

 # Chaque arbre ne voit qu'un sous-ensemble des classes -> realignement
    global_classes = rf.classes_
    K = len(global_classes)
    M = len(rf.estimators_)
    P = np.zeros((M, K))
    for m, tree in enumerate(rf.estimators_):
        proba_m = tree.predict_proba(x_arr)[0]                     # (K_m,)
        idx = np.searchsorted(global_classes, tree.classes_)
        P[m, idx] = proba_m

    # Cas degenere : une seule classe -> consensus parfait
    if K < 2:
        return 1.0

    p_star = representative_distribution(P, distance=distance)
    r_sm = robust_uncertainty(P, p_star, S=smallest_margin, distance=distance)
    return float(r_sm)

# 2. Score de fiabilite credal pour la REGRESSION

# Distances entre ECDF (prérequis pour credal_reliability_regression)
def wasserstein_1_to_dirac(preds, y_ref):
    """W_1(F_H, delta_{y_ref}) = (1/M) sum_m |y^m - y_ref|."""
    return float(np.mean(np.abs(preds - y_ref)))


def cramer_von_mises_to_dirac(preds, y_ref):
    """CvM(F_H, delta_{y_ref}) = (1/M) sum_m (y^m - y_ref)^2."""
    return float(np.mean((preds - y_ref) ** 2))


def kolmogorov_smirnov_to_dirac(preds, y_ref):
    """KS(F_H, delta_{y_ref}) = max(L/M, R/M)."""
    M = len(preds)
    L = int(np.sum(preds < y_ref))
    R = int(np.sum(preds > y_ref))
    return float(max(L, R) / M)


_DISTANCES_ECDF = {
    "wasserstein": wasserstein_1_to_dirac,
    "cramer":      cramer_von_mises_to_dirac,
    "ks":          kolmogorov_smirnov_to_dirac,
}


def credal_reliability_regression(rf, x_row, y_train_std,
                                  alpha=0.2,
                                  distance="wasserstein"):
    """
    Score credal pour la régression, version ECDF.

    Pipeline :
      1) Recuperer les M predictions y^m des arbres pour l'instance x
      2) Calculer y_bar = moyenne (analogue de p* du papier credal sous Sqe)
      3) Filtrer les (1-alpha)*M arbres les plus proches de y_bar
      4) Calculer la distance entre l'ECDF des arbres consensuels et la
         masse Dirac en y_bar
      5) Normaliser et retourner un score dans [0, 1]
    """
    if distance not in _DISTANCES_ECDF:
        raise ValueError(f"distance inconnue : {distance}")

    if isinstance(x_row, pd.Series):
        x_arr = x_row.values.reshape(1, -1)
    else:
        x_arr = np.asarray(x_row).reshape(1, -1)

    preds = np.array([tree.predict(x_arr)[0] for tree in rf.estimators_])
    M = len(preds)
    y_bar = preds.mean()

    M_alpha = max(2, int(np.ceil((1.0 - alpha) * M)))
    order = np.argsort(np.abs(preds - y_bar))
    consensual = preds[order[:M_alpha]]

    d_fun = _DISTANCES_ECDF[distance]
    d_raw = d_fun(consensual, y_bar)

    if y_train_std is None or y_train_std == 0 or np.isnan(y_train_std):
        return 1.0

    if distance == "wasserstein":
        score = 1.0 - d_raw / y_train_std
    elif distance == "cramer":
        score = 1.0 - d_raw / (y_train_std ** 2)
    else:  # ks
        score = 1.0 - d_raw

    return float(max(0.0, score))

# 3. Fonction unifiee : score de fiabilite credal regression OU classification

def credal_reliability(rf, x_row, is_categorical, y_train_std=None,
                       distance="sqe", alpha_reg=0.2):
    """
    Wrapper qui appelle la version classification ou regression selon
    le type de la colonne a imputer.
    """
    if is_categorical:
        return credal_reliability_classif(rf, x_row, distance=distance)
    return credal_reliability_regression(rf, x_row, y_train_std, alpha=alpha_reg)


# 4. is_reliable_combined : test SHAP + Credal

def is_reliable_combined(shap_ok, credal_score, tau_credal):
    """
    Une imputation est jugee fiable ssi LES DEUX criteres sont satisfaits :
      - shap_ok        : booleen, sortie du critere SHAP de MisShapForest
      - credal_score   : float dans [0, 1], score du credal ensembling
      - tau_credal     : seuil minimum sur le score credal
    """
    return bool(shap_ok) and (credal_score >= tau_credal)

# 5. Pipeline d'imputation : MissForest + critere credal (sans SHAP)

def credal_imputation(data, categorical_columns, max_iter=10,
                           tau_credal=0.3, distance="sqe",
                           alpha_credal=0.2):
    """
    MissForest avec critere de fiabilite credal uniquement (sans SHAP).
 
    Pour chaque cellule (i, col) : RF entrainee sur les lignes ou col est
    observee, puis score credal R_S(H(x)) mesurant le consensus entre arbres.
    Si R_S >= tau_credal la cellule est gardee, sinon elle reste marquee
    manquante et sa ligne sera supprimee. Le critere porte donc sur la
    dispersion des sorties, pas sur l'information SHAP des features.
 
    Parameters
    ----------
    - data : pandas.DataFrame avec NaN.
    - categorical_columns : list[str] — colonnes categorielles encodees.
    - max_iter : int — nombre max d'iterations (idem MissForest).
    - tau_credal : float dans [0, 1] — seuil credal ; 0 redonne MissForest,
        proche de 1 rejette presque tout.
    - distance : str — distance pour p* (cf. DISTANCES).
    - alpha_credal : float — fraction d'arbres outliers jetee (regression).
 
    Returns
    -------
    (df_filled, missing_rows_index, n_missing_rows, pct_missing_rows)
        Memes sorties que shap_imputation.
    """
    data_copy = data.copy()
    numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
    df_numeric = data_copy[numeric_cols]

    # Initialisation par la moyenne (idem MissForest et MisShapForest)
    df_filled = df_numeric.copy()
    for col in df_filled.columns:
        #df_filled[col].fillna(df_filled[col].mean(), inplace=True)
        df_filled[col] = df_filled[col].fillna(df_filled[col].mean())

    mean_nrmse_prec = 0
    mean_nrmse = -1
    missing_rows_index = []

    for iteration in range(max_iter):
        df_numeric_iter = data_copy[numeric_cols]
        df_filled_prec = df_filled.copy()
        num_cols_prec = -1
        num_missing_cols = -2

        # Boucle interne : on impute jusqu'a stabilisation du nombre
        # de cellules encore marquees manquantes
        while (num_cols_prec != num_missing_cols) and (num_missing_cols != 0):
            col_missing_counts = df_numeric_iter[numeric_cols].isna().sum()
            sorted_columns = col_missing_counts.sort_values(
                ascending=True).index.tolist()

            for col in sorted_columns:
                missing_mask = df_numeric_iter[col].isnull()
                if missing_mask.sum() == 0:
                    continue
                other_cols = df_numeric_iter.columns.drop(col)

                # Train RF (idem MissForest)
                X_train = df_filled.loc[~missing_mask, other_cols]
                y_train = df_filled.loc[~missing_mask, col]

                is_cat = (col in categorical_columns)
                if is_cat:
                    rf = RandomForestClassifier(
                        n_estimators=100, random_state=42, oob_score=True)
                else:
                    rf = RandomForestRegressor(
                        n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)

                y_train_std = float(np.std(y_train)) if not is_cat else None

                # Prediction des cellules manquantes
                X_pred = df_filled.loc[missing_mask, other_cols]
                if len(X_pred) == 0:
                    continue
                y_pred = rf.predict(X_pred)

                # Pour chaque cellule a imputer : score credal -> decision
                for key in range(len(X_pred)):
                    idx = X_pred.index[key]
                    pred_val = y_pred[key]

                    # Score credal seul (pas de SHAP)
                    credal_score = credal_reliability(
                        rf, X_pred.iloc[key],
                        is_categorical=is_cat,
                        y_train_std=y_train_std,
                        distance=distance,
                        alpha_reg=alpha_credal,
                    )

                    reliable = (credal_score >= tau_credal)

                    df_filled.loc[idx, col] = pred_val          # toujours
                    if reliable:
                        df_numeric_iter.loc[idx, col] = pred_val
                    # sinon : ligne marquee non-fiable, sera supprimee

                num_cols_prec = num_missing_cols
                num_missing_cols = df_numeric_iter.isna().sum().sum()

        # Critere d'arret (idem MissForest et MisShapForest)
        missing_rows_index_prec = missing_rows_index
        missing_rows_index = df_numeric_iter[
            df_numeric_iter.isna().any(axis=1)].index.tolist()
        nrmse_result = compute_nrmse(df_filled_prec, df_filled)
        mean_nrmse_prec = mean_nrmse
        mean_nrmse = nrmse_result.mean()
        if iteration != 0 and mean_nrmse_prec < mean_nrmse:
            df_filled = df_filled_prec.copy()
            missing_rows_index = missing_rows_index_prec
            break

    n_missing = len(missing_rows_index)
    pct_missing = 100.0 * n_missing / len(data_copy) if len(data_copy) else 0.0
    return df_filled, missing_rows_index, n_missing, pct_missing
