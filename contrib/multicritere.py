"""Multi-critère : front de Pareto, Lorenz & OWA"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .credal import *
from .data_evaluation import *
from .shap_imputation import *


# =====================================================================
# Construction du front de Pareto (ε-contrainte + Optuna/TPE)
# =====================================================================
def impute_reliability_once(data, cat_cols, max_iter=2, score="credal",
                             shap_threshold=10.0, distance="sqe", credal_alpha=0.2):
    """Impute (MissForest) une fois + score de fiabilite continu par cellule."""
    data = data.copy()
    num = data.select_dtypes(include=[np.number]).columns
    dnum = data[num]; miss0 = dnum.isna()
    filled = dnum.copy()
    for c in filled: filled[c] = filled[c].fillna(filled[c].mean())
    if score == "shap":
        import shap
    cellr = {}
    for it in range(max_iter):
        last = (it == max_iter - 1)
        for col in miss0[num].sum().sort_values().index:
            im = miss0[col]
            if im.sum() == 0: continue
            other = dnum.columns.drop(col)
            Xtr = filled.loc[~im, other]; ytr = filled.loc[~im, col]
            iscat = col in list(cat_cols)
            rf = (RandomForestClassifier(n_estimators=100, random_state=42) if iscat
                  else RandomForestRegressor(n_estimators=100, random_state=42))
            rf.fit(Xtr, ytr)
            pidx = im[im].index if last else dnum[col].isnull()[dnum[col].isnull()].index
            Xp = filled.loc[pidx, other]
            if len(Xp) == 0: continue
            pr = rf.predict(Xp); filled.loc[pidx, col] = pr; dnum.loc[pidx, col] = pr
            if last:
                if score == "credal":
                    ystd = float(ytr.std()) if not iscat else None
                    for idx in Xp.index:
                        s = credal_reliability(rf, Xp.loc[idx], is_categorical=iscat,
                                               y_train_std=ystd, distance=distance,
                                               alpha_reg=credal_alpha)
                        cellr[(idx, col)] = float(np.clip(s, 0, 1))
                else:
                    expl = shap.TreeExplainer(rf); sv = expl.shap_values(Xp)
                    ev = expl.expected_value
                    base = abs(ev[0]) if np.ndim(ev) else abs(ev); base = base if base>1e-8 else 1e-8
                    svv = sv[:, :, 1] if (iscat and np.ndim(sv)==3) else sv
                    for k, idx in enumerate(Xp.index):
                        vals = np.abs(svv[k]); imp = (vals/base*100) > shap_threshold
                        tot = vals[imp].sum()
                        if tot <= 1e-12: cellr[(idx, col)] = 1.0; continue
                        obs = dnum.loc[idx, other].notna().values
                        cellr[(idx, col)] = float(np.clip(vals[imp & obs].sum()/tot, 0, 1))
    return filled, cellr

def nrmse_excluding(excluded, cat_cols, dfo, true_values, dff):
    num = [c for c in dfo.columns if c not in list(cat_cols) and dfo[c].dtype != 'object']
    err = {c: 0.0 for c in num}; cnt = {c: 0 for c in num}; ex = set(excluded)
    for (idx, col), _ in true_values.items():
        if col in num and idx not in ex:
            d = dff.at[idx, col] - dfo.at[idx, col]; err[col] += d*d; cnt[col] += 1
    nc = {c: np.sqrt(err[c]/cnt[c])/dfo[c].std()
          for c in num if cnt[c] > 0 and dfo[c].std() > 0}
    return float(np.mean(list(nc.values()))) if nc else np.nan

# Wrapper
def shap_imputation_robuste(data, categorical_columns, max_iter, alpha, min_retained=1):

    dff, mi, n, pct = shap_imputation(data, categorical_columns, max_iter, alpha)
    enough = (len(data) - len(mi)) >= max(min_retained, 1)
    return dff, mi, n, pct, enough


def safe_eval(method, dm, tv, cat_cols, v, max_iter, dfo):
    """(NRMSE, rejet) ; NRMSE=+inf (jamais NaN) si trop peu de lignes retenues."""
    if method == "credal":
        dff, mi, n, pct = credal_imputation(dm, categorical_columns=cat_cols,
                                            max_iter=max_iter, tau_credal=v, distance="sqe")
        enough = (len(dfo) - len(mi)) >= 1
    else:
        dff, mi, n, pct, enough = shap_imputation_robuste(dm, cat_cols, max_iter, v)
    if not enough:
        return np.inf, pct / 100.0
    nr = nrmse_excluding(list(mi), cat_cols, dfo, tv, dff)
    return (np.inf if np.isnan(nr) else float(nr)), pct / 100.0


def _suggest_theta(trial, method, hi, log_alpha):
    """Espace de décision Θ :
    Θ = [0, 1] pour Credal-MissForest (θ = τ_credal),
    Θ = R+ tronqué à [0, hi] pour MisShapForest (θ = α)."""
    if method == "credal":
        return trial.suggest_float("tau_credal", 0.0, 1.0)
    if log_alpha:
        return trial.suggest_float("alpha", 1e-3, hi, log=True)
    return trial.suggest_float("alpha", 0.0, hi)


def pareto_front_min_per_deletion(dfo, cat_cols, missing_rate, deltas,
                                  method="credal", tol=0.05, n_trials=30,
                                  seed=0, max_iter=1, hi=None,
                                  n_startup_trials=10, n_warm_start=3,
                                  log_alpha=False, big_penalty=1e6,
                                  verbose=True):
    """Front de Pareto (E, r) par ε-contrainte résolue avec Optuna/TPE
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pname = "tau_credal" if method == "credal" else "alpha"
    if hi is None:
        hi = 1.0 if method == "credal" else 50.0
    eps = float(tol)

    # Même tableau corrompu pour toute la procédure
    np.random.seed(seed)
    dm, tv = generate_missing(dfo.copy(), missing_rate)

    cache = {}   # θ -> (E, r) ; E = +inf si trop peu de lignes retenues
    def evalp(v):
        v = float(v)
        if v not in cache:
            cache[v] = safe_eval(method, dm.copy(), tv, cat_cols, v, max_iter, dfo)
        return cache[v]

    use_set_constraint = hasattr(optuna.trial.Trial, "set_constraint")

    out = []
    for k, d in enumerate(deltas):
        d = float(d)

        def violation(r):
            return abs(r - d) - eps          # ≤ 0  <=>  |r − δ| ≤ ε

        def objective(trial):
            theta = _suggest_theta(trial, method, hi, log_alpha)
            E, r = evalp(theta)
            c = violation(r)
            trial.set_user_attr("reject", float(r))
            trial.set_user_attr("NRMSE", float(E))
            trial.set_user_attr("constraint", float(c))
            if use_set_constraint:
                trial.set_constraint("reject_window", float(c))
            # TPE exige une valeur finie : +inf -> pénalité (la vraie valeur
            # est conservée dans user_attrs["NRMSE"])
            return float(E) if np.isfinite(E) else float(big_penalty)

        sampler_kw = dict(seed=seed + k, n_startup_trials=n_startup_trials)
        if not use_set_constraint:
            sampler_kw["constraints_func"] = lambda ft: (ft.user_attrs["constraint"],)
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(**sampler_kw))

        # Warm start (optionnel) : meilleurs θ déjà évalués pour ce δ
        if n_warm_start > 0 and cache:
            prev = sorted(cache.items(),
                          key=lambda kv: (max(violation(kv[1][1]), 0.0),
                                          kv[1][0] if np.isfinite(kv[1][0]) else np.inf))
            lo_ok = 1e-3 if (method != "credal" and log_alpha) else 0.0
            for v, _ in prev[:n_warm_start]:
                if lo_ok <= v <= hi:
                    study.enqueue_trial({pname: v})

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        feas = [t for t in done if t.user_attrs["constraint"] <= 0
                and np.isfinite(t.user_attrs["NRMSE"])]
        if feas:
            best = min(feas, key=lambda t: t.user_attrs["NRMSE"])
        else:   # repli : essai au rejet le plus proche de δ (NRMSE fini si possible)
            pool = [t for t in done if np.isfinite(t.user_attrs["NRMSE"])] or done
            best = min(pool, key=lambda t: abs(t.user_attrs["reject"] - d))

        r_b, E_b = best.user_attrs["reject"], best.user_attrs["NRMSE"]
        out.append({"delta": d, "reject": float(r_b), "NRMSE": float(E_b),
                    pname: float(best.params[pname]),
                    "feasible": bool(len(feas) > 0),
                    "ecart_rejet": float(round(abs(r_b - d), 4)),
                    "n_trials": len(done), "n_feasible": len(feas)})
        if verbose:
            print(f"  [eps-contrainte/{method}] δ={d:.2f}: "
                  f"{len(feas)}/{len(done)} essais faisables -> "
                  f"r={r_b*100:.1f}%  E={E_b:.4f}  {pname}={best.params[pname]:.4f}"
                  + ("" if feas else "  (INFAISABLE, repli)"))

    res = pd.DataFrame(out)
    res.attrs["evaluations"] = pd.DataFrame(
        [{"param": v, "reject": r, "NRMSE": E} for v, (E, r) in cache.items()]
    ).sort_values("param").reset_index(drop=True)
    res.attrs["n_forest_evaluations"] = len(cache)
    return res


def front_pareto_enveloppe(res_shap, res_credal, only_feasible=True,
                           normaliser=True):
    """Enveloppe inférieure E_env(δ) = min(E_SHAP(δ), E_Credal(δ)) puis
    filtrage de Pareto sur (reject, NRMSE) et marquage Lorenz-optimal
    """
    def _prep(res, methode, hp):
        df = res.copy()
        if only_feasible:
            df = df[df["feasible"]]
        df = df[np.isfinite(df["NRMSE"])]
        return pd.DataFrame({"delta": df["delta"], "reject": df["reject"],
                             "NRMSE": df["NRMSE"], "methode": methode,
                             "hyperparam": hp, "valeur": df[hp],
                             "feasible": df["feasible"]})

    parts = []
    if res_shap is not None:
        parts.append(_prep(res_shap, "MisShapForest", "alpha"))
    if res_credal is not None:
        parts.append(_prep(res_credal, "Credal MissForest", "tau_credal"))
    allp = pd.concat(parts, ignore_index=True)
    if allp.empty:
        raise ValueError("Aucun point faisable : augmenter tol (ε) ou n_trials.")

    # Enveloppe inférieure : pour chaque δ, la méthode d'erreur minimale
    env = (allp.sort_values(["delta", "NRMSE"])
               .groupby("delta", as_index=False).first())
    # Un même réglage peut être retenu pour plusieurs δ voisins : dédoublonnage
    env = (env.sort_values("delta")
              .drop_duplicates(subset=["methode", "valeur"])
              .reset_index(drop=True))

    # Filtrage de Pareto (objectifs à minimiser : rejet, erreur)
    P = env[["reject", "NRMSE"]].to_numpy(float)
    env["pareto"] = pareto_mask(P)
    front = env[env["pareto"]].sort_values("reject").reset_index(drop=True)
    front["lorenz_opt"] = lorenz_optimal(front[["reject", "NRMSE"]].to_numpy(float),
                                         normaliser=normaliser)
    return front


def plot_front(front, res_shap=None, res_credal=None, ax=None):
    """Front approché : évaluations brutes (gris), front Pareto, Lorenz-optimaux."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    for res, lab, m in [(res_shap, "MisShapForest (évaluations)", "x"),
                        (res_credal, "Credal (évaluations)", ".")]:
        if res is not None and "evaluations" in res.attrs:
            ev = res.attrs["evaluations"]; ev = ev[np.isfinite(ev["NRMSE"])]
            ax.scatter(ev["reject"], ev["NRMSE"], s=12, marker=m, c="0.6",
                       alpha=0.6, label=lab)
    for meth, c in [("MisShapForest", "tab:orange"), ("Credal MissForest", "tab:blue")]:
        f = front[front["methode"] == meth]
        ax.scatter(f["reject"], f["NRMSE"], s=45, c=c, edgecolor="k",
                   zorder=3, label=f"Pareto — {meth}")
    ax.step(front["reject"], front["NRMSE"], where="post", c="k", lw=1, alpha=0.5)
    L = front[front["lorenz_opt"]]
    ax.scatter(L["reject"], L["NRMSE"], s=140, facecolors="none",
               edgecolors="red", lw=1.8, zorder=4, label="Lorenz-optimal")
    ax.set_xlabel("Taux de rejet r"); ax.set_ylabel("NRMSE E")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return ax

# =====================================================================
# Sélection sur le front : Lorenz & OWA
# =====================================================================


def pareto_mask(P):
    """Masque booléen des points Pareto-optimaux (objectifs à minimiser)."""
    P = np.asarray(P, float); n = len(P); keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]: continue
        dom = np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1)
        if np.any(dom): keep[i] = False
    return keep

def lorenz_vector(v):
    """Tri décroissant (le pire critère d'abord) puis sommes cumulées."""
    return np.cumsum(np.sort(np.asarray(v, float))[::-1])

def lorenz_nondominated(L):
    """Masque booléen des vecteurs de Lorenz non dominés
    (dominance composante à composante, à minimiser)."""
    n = len(L); keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]: continue
        dom = np.all(L <= L[i], axis=1) & np.any(L < L[i], axis=1)
        if np.any(dom): keep[i] = False
    return keep

def normaliser_front(obj):
    """Min-max sur le front : 0 = idéal, 1 = nadir. Réexprime chaque critère
    en regret relatif, rendant comparables des échelles différentes."""
    obj = np.asarray(obj, float)
    ideal = obj.min(axis=0)
    nadir = obj.max(axis=0)
    span = nadir - ideal
    span[span == 0] = 1.0        # évite la division par zéro si un critère est constant
    return (obj - ideal) / span

def lorenz_optimal(obj, normaliser=True):
    """Masque des solutions Lorenz-optimales d'un front.
 
    obj : array (n, k) d'objectifs à minimiser (ex. [reject, NRMSE]).
    normaliser : applique normaliser_front avant les vecteurs de Lorenz
        (recommandé si les critères n'ont pas la même échelle).
    """

    obj = np.asarray(obj, float)
    if normaliser:
        obj = normaliser_front(obj)
    L = np.array([lorenz_vector(o) for o in obj])
    return lorenz_nondominated(L)

# ============================================================
#  Heuristique OWA
# ============================================================

def owa_heuristique(pareto, n_w=2001, plot=True, normaliser=True,col_lorenz="lorenz_opt"):
    """Choix final parmi les solutions Lorenz-optimales par balayage OWA.
 
    En 2 critères, un OWA équitable a 1 paramètre w in [0.5,1] (poids du pire
    critère) : on balaye w et on garde la solution optimale sur le plus large
    sous-intervalle.
 
    Entrée : DataFrame avec 'reject', 'NRMSE', 'methode', 'hyperparam',
    'valeur' et le booléen col_lorenz.
    normaliser : si False, OWA sur les valeurs brutes.
    Sortie : (sol, best_idx), 'sol' portant w_min/w_max/largeur par solution.
    """
    # 'sol' = solutions Lorenz-optimales (reject, NRMSE, methode, hyperparam, valeur)
    mask = pareto[col_lorenz].to_numpy(bool)
    sol  = pareto[pareto[col_lorenz]].reset_index(drop=True).copy()

    # Objectifs pour le score OWA : bruts, ou normalisés sur TOUT le front
    full = pareto[["reject", "NRMSE"]].to_numpy(float)
    obj = (normaliser_front(full) if normaliser else full)[mask]
    label = "valeurs normalisées" if normaliser else "valeurs brutes"

    p = obj.max(axis=1)   # pire critère de chaque solution
    b = obj.min(axis=1)   # meilleur critère

    # Score OWA : droite OWA_i(w) = b_i + w*(p_i - b_i)
    def owa(i, w):
        return b[i] + w * (p[i] - b[i])

    # Balayage de w in [0.5,1] : quelle solution est optimale (min) à chaque w ?
    W = np.linspace(0.5, 1.0, n_w)
    scores = np.array([owa(i, W) for i in range(len(sol))])
    winner_at = scores.argmin(axis=0)

    dw = W[1] - W[0]
    interval_width = np.array([(winner_at == i).sum() * dw for i in range(len(sol))])
    w_lo, w_hi = [], []
    for i in range(len(sol)):
        ws = W[winner_at == i]
        if len(ws): w_lo.append(ws.min()); w_hi.append(ws.max())
        else:       w_lo.append(np.nan);   w_hi.append(np.nan)

    sol["w_min"]   = np.round(w_lo, 4)
    sol["w_max"]   = np.round(w_hi, 4)
    sol["largeur"] = np.round(interval_width, 4)

    best_idx = int(np.argmax(interval_width))
    best = sol.loc[best_idx]

    print(f"=== Intervalles d'optimalité OWA ({label}, w in [0.5,1]) ===")
    print(sol[["reject", "NRMSE", "methode", "hyperparam", "valeur",
               "w_min", "w_max", "largeur"]].to_string(index=False))

    print(f"\n=== Solution choisie (intervalle le plus large : {best['largeur']:.3f}) ===")
    print(f"  reject = {best['reject']*100:.1f}%   NRMSE = {best['NRMSE']:.4f}")
    print(f"  méthode = {best['methode']}   {best['hyperparam']} = {best['valeur']:.4f}")
    print(f"  optimale pour w in [{best['w_min']:.3f}, {best['w_max']:.3f}]")

    if plot:
        # Graphe : droites OWA + enveloppe inférieure
        plt.figure(figsize=(9, 5.5))
        env_inf = scores.min(axis=0)   # enveloppe inférieure

        # Tracé des droites (on mémorise la couleur de chacune pour colorer l'aire)
        line_colors = []
        for i in range(len(sol)):
            lw = 3.5 if i == best_idx else 1.5
            (line,) = plt.plot(W, owa(i, W), lw=lw, alpha=0.9 if i == best_idx else 0.5,
                               label=f"sol {i} ({sol.loc[i,'methode']}, {sol.loc[i,'valeur']:.3f})")
            line_colors.append(line.get_color())

        # Aire sous l'enveloppe inférieure, coloriée par segment selon la droite gagnante
        # (même couleur que la droite qui forme l'enveloppe sur ce sous-intervalle de w)
        y_bas = env_inf.min() - 0.05 * (env_inf.max() - env_inf.min())
        for i in range(len(sol)):
            mask = winner_at == i
            if mask.any():
                plt.fill_between(W, y_bas, env_inf, where=mask,
                                 color=line_colors[i], alpha=0.25, linewidth=0)

        plt.plot(W, env_inf, "k-", lw=2.5, alpha=0.35, label="enveloppe inférieure")
        plt.ylim(bottom=y_bas)
        plt.xlabel("Poids W"); plt.ylabel("score OWA (à minimiser)")
        plt.title("Heuristique OWA — Solution = Plus large intervalle")
        plt.legend(fontsize=8, ncol=2); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    return sol, best_idx
