"""Credal Ensembling + variante CREDO pour la regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

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

# =====================================================================
# Variante CREDO du score credal regressif (arXiv:2603.06826)
# =====================================================================
"""
Substituable a credal_reliability_regression : meme convention d'appel
(rf, x_row, y_train_std, ...), sortie dans [0, 1] croissante avec la fiabilite.

Correspondance avec le papier (annexe A) :
  - Alg. 1, ENVELOPE (l. 3-13) : intervalles par arbre [qL^m, qU^m] de niveau
    1 - alpha0, puis rognage CL = Q_{gamma/2}(qL), CU = Q_{1-gamma/2}(qU) ;
  - Alg. 2 : gamma(x) par rarete locale (kNN sur variables standardisees) ;
  - Alg. 3 : U_A = moyenne(qU - qL), U_E = (CU - CL) - U_A (non tronque).

Ecarts assumes :
  - les tirages posterieurs theta^(b) sont remplaces par les arbres d'une
    foret bootstrap, et F_theta(.|x) par la distribution empirique des
    valeurs d'entrainement de la feuille atteinte (type forets quantiles) ;
    il faut donc des feuilles assez grandes (min_samples_leaf ~ 10), sinon
    U_A est sous-estime et le bruit bascule dans U_E ;
  - pas de split ni de calibration conforme (Alg. 1, l. 1, 14-19) : tau_hat
    est constant par colonne et n'entre pas dans U_E, la decomposition est
    donc inchangee, mais il n'y a plus de garantie de couverture ;
  - le score normalise U_E par sigma_train (le papier rapporte U_E / (u - l)).

variant="range" n'est PAS CREDO : c'est l'etendue rognee des predictions
ponctuelles, sans modele aleatoire. A etiqueter separement.
"""

# Valeurs par defaut de l'annexe C.2 du papier
CREDO_DEFAULTS = dict(
    alpha0=0.10,       # alpha_0 = alpha = 0.1 dans les experiences
    gamma_min=0.10,
    gamma_max=0.75,
    m_gamma=0.0,
    tau_gamma=1.0,
    eps=1e-6,
    c_base=6.672,      # k = ceil(c_base * n^(4 / (4 + d)))
)


def _as_row(x_row):
    if isinstance(x_row, pd.Series):
        return x_row.values.reshape(1, -1).astype(float)
    return np.asarray(x_row, dtype=float).reshape(1, -1)


def _width_to_score(width, y_train_std):
    """1 - largeur / sigma, ramene dans [0, 1]. Une largeur negative
    (U_E < 0, possible hors troncature) compte comme une largeur nulle."""
    return float(np.clip(1.0 - max(width, 0.0) / y_train_std, 0.0, 1.0))


def _std_is_invalid(y_train_std):
    return y_train_std is None or y_train_std == 0 or not np.isfinite(y_train_std)


class CredoContext:
    """Quantites calculees une fois par foret et reutilisees pour toutes les
    cellules de la colonne :
      - valeurs d'entrainement par feuille et par arbre (variant="envelope") ;
      - modele de rarete locale kNN (adaptive=True), Alg. 2 lignes 1-2.

    rf peut etre None si seule la rarete est utile (variant="range").
    """

    def __init__(self, rf, X_train, y_train, k=None,
                 gamma_min=CREDO_DEFAULTS["gamma_min"],
                 gamma_max=CREDO_DEFAULTS["gamma_max"],
                 m_gamma=CREDO_DEFAULTS["m_gamma"],
                 tau_gamma=CREDO_DEFAULTS["tau_gamma"],
                 eps=CREDO_DEFAULTS["eps"],
                 c_base=CREDO_DEFAULTS["c_base"]):
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
        n, d = X.shape

        # Tables feuille -> valeurs d'entrainement, une par arbre
        self.tables = []
        if rf is not None:
            for tree in rf.estimators_:
                leaves = tree.apply(X)
                order = np.argsort(leaves, kind="stable")
                ls, ys = leaves[order], y[order]
                uniq, starts = np.unique(ls, return_index=True)
                self.tables.append((ys, uniq, starts))

        self.gamma_min, self.gamma_max = gamma_min, gamma_max
        self.m_gamma, self.tau_gamma, self.eps = m_gamma, tau_gamma, eps

        # phi = variables standardisees (annexe C.2)
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0

        if n < 2:
            # Rarete non definie : on ne rogne que le minimum (prudent)
            self.nn, self.k = None, 1
            return

        if k is None:
            k = int(np.ceil(c_base * n ** (4.0 / (4.0 + d))))
        self.k = max(1, min(k, n - 1))

        Z = (X - self.mu) / self.sd
        self.nn = NearestNeighbors(n_neighbors=self.k + 1).fit(Z)
        # Alg. 2, l. 1 : k-ieme voisin de chaque X_i, lui-meme exclu
        # (colonne 0 = le point lui-meme, donc colonne k = k-ieme voisin)
        r = self.nn.kneighbors(Z)[0][:, self.k]
        # Alg. 2, l. 2
        self.q_lo, self.q_hi = np.quantile(r, [0.50, 0.95])

    def leaf_values(self, m, leaf_id):
        ys, uniq, starts = self.tables[m]
        k = int(np.searchsorted(uniq, leaf_id))
        if k >= len(uniq) or uniq[k] != leaf_id:
            return ys[:0]
        end = starts[k + 1] if k + 1 < len(starts) else len(ys)
        return ys[starts[k]:end]

    def scarcity(self, x_row):
        """Alg. 2, l. 3-5 : score de rarete sc(x)."""
        z = (_as_row(x_row) - self.mu) / self.sd
        # Le point requete n'est pas dans X_train (lignes a imputer),
        # donc son k-ieme voisin est la colonne k-1.
        r = float(self.nn.kneighbors(z)[0][0, self.k - 1])
        return (r - self.q_lo) / (self.q_hi - self.q_lo + self.eps)

    def gamma(self, x_row):
        """Alg. 2, l. 6 : region peu peuplee -> gamma petit -> moins de
        rognage -> enveloppe plus large -> score plus bas."""
        if self.nn is None:
            return float(self.gamma_min)
        sc = self.scarcity(x_row)
        sig = 1.0 / (1.0 + np.exp(-(sc - self.m_gamma) / self.tau_gamma))
        return float(self.gamma_max - (self.gamma_max - self.gamma_min) * sig)


def credal_reliability_regression_credo(rf, x_row, y_train_std,
                                        gamma=0.2,
                                        variant="envelope",
                                        ctx=None,
                                        alpha0=CREDO_DEFAULTS["alpha0"],
                                        adaptive=False,
                                        decompose=False):
    """
    Score credal pour la regression, version CREDO.

    Parametres
    ----------
    rf       : foret de regression. Pour variant="envelope", utiliser une
               foret a feuilles assez grandes (min_samples_leaf ~ 10), celle
               qui a servi a construire ctx.
    gamma    : niveau de rognage constant (ignore si adaptive=True).
               Ne pas confondre avec le alpha conforme du papier.
    variant  : "envelope" (CREDO) ou "range" (etendue rognee, hors CREDO).
    ctx      : CredoContext, requis pour variant="envelope" ou adaptive=True.
    alpha0   : niveau nominal des intervalles par arbre.
    decompose: si True, renvoie (score, U_A, largeur_penalisee) ; la largeur
               penalisee vaut U_E pour "envelope" et hi - lo pour "range".
    """
    if variant not in ("range", "envelope"):
        raise ValueError(f"variante inconnue : {variant}")
    if (variant == "envelope" or adaptive) and ctx is None:
        raise ValueError("un CredoContext est requis pour cette configuration")

    if _std_is_invalid(y_train_std):
        return (1.0, 0.0, 0.0) if decompose else 1.0

    x_arr = _as_row(x_row)
    g = ctx.gamma(x_row) if adaptive else gamma       # Alg. 1, l. 7-9

    if variant == "range":
        preds = np.array([tree.predict(x_arr)[0] for tree in rf.estimators_])
        lo, hi = np.quantile(preds, [g / 2.0, 1.0 - g / 2.0])
        width = float(hi - lo)
        score = _width_to_score(width, y_train_std)
        return (score, 0.0, width) if decompose else score

    # Alg. 1, l. 4-6 : intervalles centraux par arbre
    qL, qU = [], []
    for m, tree in enumerate(rf.estimators_):
        vals = ctx.leaf_values(m, int(tree.apply(x_arr)[0]))
        if vals.size == 0:
            continue
        a, b = np.quantile(vals, [alpha0 / 2.0, 1.0 - alpha0 / 2.0])
        qL.append(a)
        qU.append(b)
    if not qL:
        return (1.0, 0.0, 0.0) if decompose else 1.0
    qL, qU = np.asarray(qL), np.asarray(qU)

    # Alg. 1, l. 10-11 : enveloppe rognee
    CL = float(np.quantile(qL, g / 2.0))
    CU = float(np.quantile(qU, 1.0 - g / 2.0))

    # Alg. 3, l. 2-3 : decomposition (U_E non tronque, comme le papier)
    U_A = float(np.mean(qU - qL))
    U_E = (CU - CL) - U_A

    score = _width_to_score(U_E, y_train_std)
    return (score, U_A, U_E) if decompose else score


# =====================================================================
# Score de rang (suppression du plancher de rejet)
# =====================================================================
def to_rank_score(raw, higher_is_worse=True, reference=None):
    """Transforme les mesures brutes d'une colonne en scores de rang sur [0, 1].

    Le score max(0, 1 - d / sigma) sature a 0 des que d depasse sigma : ces
    cellules sont rejetees quel que soit tau. Le rang, transformation
    monotone, ne change aucun classement mais etale les scores sur [0, 1] ;
    tau devient alors a peu pres la proportion de cellules rejetees par
    colonne (y compris quand toutes les cellules sont fiables).

    reference : mesures servant de distribution de reference. Par defaut,
    raw lui-meme (rang interne). Dans la boucle d'imputation, on passe la
    distribution du premier passage sur la colonne : sinon, en re-scorant
    seulement les quelques cellules rejetees, elles seraient re-classees
    entre elles et finiraient toutes acceptees.
    Les ex aequo recoivent le rang moyen.
    """
    x = np.asarray(raw, dtype=float)
    if x.size == 0:
        return x
    if reference is None:
        pct = rankdata(x, method="average") / (x.size + 1.0)
    else:
        ref = np.sort(np.asarray(reference, dtype=float))
        lo = np.searchsorted(ref, x, side="left")
        hi = np.searchsorted(ref, x, side="right")
        pct = ((lo + hi) / 2.0 + 0.5) / (ref.size + 1.0)
    return 1.0 - pct if higher_is_worse else pct


# =====================================================================
# Fonction unifiee : classification OU regression
# =====================================================================
def credal_reliability(rf, x_row, is_categorical, y_train_std=None,
                       distance="sqe", alpha_reg=0.2,
                       ctx=None, variant="envelope", adaptive=False,
                       alpha0=CREDO_DEFAULTS["alpha0"]):
    """Classification : R_SM (Nguyen et al.). Regression : CREDO.
    alpha_reg est le niveau de rognage gamma."""
    if is_categorical:
        return credal_reliability_classif(rf, x_row, distance=distance)
    return credal_reliability_regression_credo(
        rf, x_row, y_train_std, gamma=alpha_reg,
        variant=variant, ctx=ctx, adaptive=adaptive, alpha0=alpha0,
    )


def is_reliable_combined(shap_ok, credal_score, tau_credal):
    """Fiable ssi le critere SHAP ET le critere credal sont satisfaits."""
    return bool(shap_ok) and (credal_score >= tau_credal)


# =====================================================================
# Pipeline d'imputation : MissForest + critere credal (sans SHAP)
# =====================================================================
def credal_imputation2(data, categorical_columns, max_iter=10,
                       tau_credal=0.3, distance="sqe", alpha_credal=0.2,
                       credo_variant="envelope", credo_adaptive=False,
                       credo_alpha0=CREDO_DEFAULTS["alpha0"],
                       credo_min_samples_leaf=10, credo_k=None,
                       credal_rank=False):
    """
    MissForest avec critere de fiabilite credal uniquement (sans SHAP).

    Pour chaque cellule (i, col) : RF entrainee sur les lignes ou col est
    observee, puis score credal. Si score >= tau_credal la cellule est
    gardee, sinon elle reste marquee manquante et sa ligne sera supprimee.

    Parametres
    ----------
    data, categorical_columns, max_iter, distance : idem MissForest.
    tau_credal : seuil dans [0, 1].
    alpha_credal : niveau de rognage gamma constant (regression).
    credo_variant : "envelope" (CREDO) ou "range" (hors CREDO).
    credo_adaptive : gamma(x) par rarete locale (Alg. 2).
    credo_alpha0 : niveau nominal des intervalles par arbre.
    credo_min_samples_leaf : taille minimale des feuilles de la foret de
        SCORE (variant="envelope"). Les predictions imputees viennent
        toujours de la foret MissForest standard.
    credo_k : nombre de voisins pour la rarete (None = heuristique du papier).
    credal_rank : si True, scores remplaces par leur rang dans la colonne,
        calcule par rapport au premier passage de l'iteration
        (tau = proportion approximative de rejets par colonne).

    Retours
    -------
    (df_filled, missing_rows_index, n_missing_rows, pct_missing_rows)
    """
    data_copy = data.copy()
    numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
    df_numeric = data_copy[numeric_cols]

    # Initialisation par la moyenne (idem MissForest et MisShapForest)
    df_filled = df_numeric.copy()
    for col in df_filled.columns:
        df_filled[col] = df_filled[col].fillna(df_filled[col].mean())

    mean_nrmse_prec = 0
    mean_nrmse = -1
    missing_rows_index = []

    for iteration in range(max_iter):
        df_numeric_iter = data_copy[numeric_cols].copy()
        df_filled_prec = df_filled.copy()
        rank_ref = {}   # distributions de reference par colonne (credal_rank)
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

                # Foret d'imputation (idem MissForest)
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

                X_pred = df_filled.loc[missing_mask, other_cols]
                if len(X_pred) == 0:
                    continue
                y_pred = rf.predict(X_pred)

                # Foret de score et contexte CREDO (regression seulement)
                y_train_std = None
                rf_score, ctx = rf, None
                if not is_cat:
                    y_train_std = float(np.std(y_train))
                    if credo_variant == "envelope":
                        # Feuilles assez grandes pour que F_theta(.|x) soit
                        # une vraie distribution (sinon U_A ~ 0)
                        rf_score = RandomForestRegressor(
                            n_estimators=100, random_state=42,
                            min_samples_leaf=credo_min_samples_leaf,
                        ).fit(X_train, y_train)
                    if credo_variant == "envelope" or credo_adaptive:
                        ctx = CredoContext(
                            rf_score if credo_variant == "envelope" else None,
                            X_train, y_train, k=credo_k,
                        )

                # Scores de toutes les cellules de la colonne
                n_pred = len(X_pred)
                scores = np.empty(n_pred)
                raw = np.empty(n_pred)       # plus grand = moins fiable
                for key in range(n_pred):
                    x = X_pred.iloc[key]
                    if is_cat:
                        s = credal_reliability_classif(rf, x, distance=distance)
                        raw[key] = 1.0 - s
                    else:
                        s, _, width = credal_reliability_regression_credo(
                            rf_score, x, y_train_std,
                            gamma=alpha_credal,
                            variant=credo_variant,
                            ctx=ctx,
                            alpha0=credo_alpha0,
                            adaptive=credo_adaptive,
                            decompose=True,
                        )
                        raw[key] = (0.0 if _std_is_invalid(y_train_std)
                                    else width / y_train_std)
                    scores[key] = s

                if credal_rank:
                    if col not in rank_ref:
                        rank_ref[col] = raw.copy()
                    scores = to_rank_score(raw, higher_is_worse=True,
                                           reference=rank_ref[col])

                # Decision
                for key in range(n_pred):
                    idx = X_pred.index[key]
                    pred_val = y_pred[key]
                    df_filled.loc[idx, col] = pred_val          # toujours
                    if scores[key] >= tau_credal:
                        df_numeric_iter.loc[idx, col] = pred_val
                    # sinon : ligne marquee non fiable, sera supprimee

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
