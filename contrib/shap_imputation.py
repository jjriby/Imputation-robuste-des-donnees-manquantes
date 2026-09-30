
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .data_evaluation import compute_nrmse
import shapiq


# =====================================================================
# MisShapForest (fiabilité par valeurs SHAP)
# =====================================================================
def shap_imputation(data,categorical_columns,max_iter,alpha) : 
    """
    Imputation itérative basée sur RandomForest + sélection de variables via SHAP.

    Idée générale :
    - Pour chaque colonne ayant des NaN, on entraîne un modèle (RF regressor/classifier) sur
      les lignes observées.
    - On calcule ensuite les valeurs SHAP sur les lignes à imputer afin d'identifier les
      features les plus influentes.
    - Si les features "importantes" (impact relatif > `alpha`) ne sont pas manquantes sur la
      ligne considérée, on impute la valeur et on marque la cellule comme complétée.

    La fonction renvoie également les lignes qui restent incomplètes (encore des NaN) après
    les itérations : elles peuvent être supprimées pour l'évaluation.

    Paramètres
    ----------
    data : pandas.DataFrame
        Données avec NaN (après label encoding).
    categorical_columns : list[str] | pandas.Index
        Colonnes catégorielles encodées.
    max_iter : int
        Nombre d'itérations externes.
    alpha : float
        Seuil sur l'impact relatif SHAP (en %) pour sélectionner les features à vérifier.

    Retours
    -------
    (df_filled, missing_rows_index, n_missing_rows, pct_missing_rows) : tuple
        - 'df_filled' : DataFrame imputé (numérique).
        - 'missing_rows_index' : indices des lignes encore incomplètes après imputation.
        - 'n_missing_row' : nombre de lignes incomplètes.
        - 'pct_missing_rows' : pourcentage de lignes incomplètes.
    """

    
    try:
        shap.initjs()  # utile uniquement pour l'affichage JS dans Jupyter
    except Exception:
        # Sans effet sur les calculs ; on ignore les erreurs (kernel non-notebook, etc.).
        pass
    
    
    # 0. Copie des données
    missing_rows_index = 0 
    data_copy = data.copy()
    
    # 1. Sélection des colonnes numériques
    numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
    df_numeric = data_copy[numeric_cols]
    
    
    # 3. Initialisation par la moyenne
    df_filled = df_numeric.copy()
    for col in df_filled.columns:
        #df_filled[col].fillna(df_filled[col].mean(), inplace=True)
        df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    
    
    mean_nrmse_prec = 0
    mean_nrmse = -1
    
    for iteration in range(max_iter):
    
        df_numeric = data_copy[numeric_cols]
        df_filled_prec = df_filled.copy()
        num_cols_prec = -1
        num_missing_cols = -2
       
        while ((num_cols_prec != num_missing_cols ) and (num_missing_cols != 0)) : 
            col_missing_counts = df_numeric[numeric_cols].isna().sum()
            sorted_columns = col_missing_counts.sort_values(ascending=True).index.tolist()
            
            for col in sorted_columns:
                
                
                missing_mask = df_numeric[col].isnull()
        
                if missing_mask.sum() == 0:
                    continue
                other_cols = df_numeric.columns.drop(col)
                pred_mask = missing_mask & df_filled[other_cols].notna().all(axis=1)
                
                
                X_train = df_filled.loc[~missing_mask, other_cols]
                y_train = df_filled.loc[~missing_mask, col]
        
                if(col in categorical_columns) : 
                    rf = RandomForestClassifier(n_estimators=100, random_state=42,oob_score=True)
        
                else :
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                
                
                pred_mask = missing_mask 
                t3 =  df_filled.loc[missing_mask, other_cols]
                X_pred = df_filled.loc[pred_mask, other_cols]
                                 
        
                if(len(X_pred)>0):                 
                    explainer = shap.TreeExplainer(rf)
                    shap_values = explainer.shap_values(X_pred)
                    feature_names = X_pred.columns
                    y_pred = rf.predict(X_pred)
        
                    if(col in categorical_columns):
                        shap_values_to_print = shap_values[:,:,1] # first sample's SHAP values for class 1
                    else:
                        shap_values_to_print = shap_values  # first sample's SHAP values (regression)
                    key=0
                    # print(col)
                    # print(shap_values.shape,shap_values_to_print.shape)
                    for key, values in enumerate(shap_values_to_print):
                        idx = X_pred.index[key]
                        somme = sum(abs(val) for val in values)
                        pred_val = y_pred[key] if y_pred[key] != 0 else 1e-8
                        shap_df = pd.DataFrame({
                            'feature': feature_names,
                            'shap_value': values,
                            'relative_impact_%': abs(values) / abs(explainer.expected_value[0])    * 100
                        })
                        
                        shap_df_sorted = shap_df.sort_values(by='relative_impact_%', ascending=False)
                        filtered_df = shap_df_sorted[shap_df_sorted['relative_impact_%'] > alpha]
                        
                        selected_cols = []
                        for _, row in filtered_df.iterrows():
                            selected_cols.append(row['feature'])
        
                        if(df_numeric.loc[idx, selected_cols].notna().all() ):
                            pd.DataFrame([X_pred.iloc[key].values], columns=rf.feature_names_in_)
        
                            df_filled.loc[idx, col] = rf.predict(X_pred.iloc[[key]])[0]                        
                            df_numeric.loc[idx, col] = rf.predict(X_pred.iloc[[key]])[0] 
        
                        else : 
                            df_filled.loc[idx, col] = rf.predict(X_pred.iloc[[key]])[0]   
                            
        
                        key +=1
                
                        
                num_cols_prec = num_missing_cols
                num_missing_cols = df_numeric.isna().sum().sum()
                
                    
        missing_rows_index_prec = missing_rows_index
        missing_rows_index = df_numeric[df_numeric.isna().any(axis=1)].index.tolist()
        # print(f"Indexes des lignes avec des valeurs manquantes : {missing_rows_index}")
        # print(f"Nombre de lignes à supprimer : {len(missing_rows_index)}")
        # print(f"Porcentage  de lignes à supprimer  : {len(missing_rows_index)/len(data_copy) * 100}")
        nrmse_result = compute_nrmse(df_filled_prec, df_filled)
        mean_nrmse_prec = mean_nrmse
        mean_nrmse = nrmse_result.mean()
        # EvalV2_NRSME(missing_rows_index)
        if(iteration != 0 and mean_nrmse_prec<mean_nrmse) : 
            df_filled = df_filled_prec.copy()
            missing_rows_index = missing_rows_index_prec
            break
    return(df_filled,missing_rows_index,len(missing_rows_index),len(missing_rows_index)/len(data_copy) * 100)


# =====================================================================
# MisShapIQForest (fiabilité par interactions SHAP-IQ)
# =====================================================================
def reliability_from_interactions(iv, feature_names, observed_mask,
                                  order_weight=(1.0, 1.0), eps=1e-12):
    """transforme les k-SII d'une ligne en score de fiabilite continu.

    Parametres
    ----------
    iv : objet InteractionValues renvoye par shapiq (attributs .values,
         .interaction_lookup, .baseline_value).
    feature_names : list[str]  colonnes de X_pred, dans l'ordre des indices shapiq.
    observed_mask : np.ndarray[bool]  observed_mask[k] = la feature k est-elle
         reellement observee (non imputee) 
    order_weight : (w1, w2)  ponderation optionnelle effets principaux / paires.
         (1,1) = neutre ; (0.5, 1.5) surpondere les synergies.

    Retour
    ------
    dict avec :
      reliability   : float in [0,1]  = masse observable / masse totale
      synergy_share : float in [0,1]  = part de la masse TOTALE portee par des paires
      obs_pair_mass : float           = masse des paires entierement observees
      top_pair      : (i,j) | None    = paire d'ordre 2 la plus forte
      top_pair_val  : float           = |k-SII| de cette paire
    """
    lookup = iv.interaction_lookup
    vals = np.asarray(iv.values, dtype=float)
    w1, w2 = order_weight

    total_mass = 0.0
    obs_mass = 0.0
    pair_mass = 0.0
    obs_pair_mass = 0.0
    top_pair, top_pair_val = None, 0.0

    for subset, pos in lookup.items():
        k = len(subset)
        if k == 0:                      # terme constant (baseline) : ignore
            continue
        m = abs(float(vals[pos]))
        w = w1 if k == 1 else w2
        wm = w * m
        total_mass += wm
        all_obs = all(observed_mask[j] for j in subset)
        if all_obs:
            obs_mass += wm
        if k == 2:
            pair_mass += wm
            if all_obs:
                obs_pair_mass += wm
            if m > top_pair_val:
                top_pair_val = m
                top_pair = tuple(int(j) for j in subset)

    reliability = obs_mass / (total_mass + eps)
    synergy_share = pair_mass / (total_mass + eps)
    return {
        "reliability": float(reliability),
        "synergy_share": float(synergy_share),
        "obs_pair_mass": float(obs_pair_mass),
        "top_pair": (tuple(feature_names[j] for j in top_pair) if top_pair else None),
        "top_pair_val": float(top_pair_val),
    }


def shapiq_imputation(data, categorical_columns, max_iter, tau,
                         max_order=2, index="k-SII", order_weight=(1.0, 1.0),
                         n_jobs=1, return_reliability=False):
    """MisShapIQForest : critere de fiabilite CONTINU credite par les synergies.

    Meme squelette que shapiq_imputation (foret 100 arbres random_state=42, init
    moyenne, meme boucle de convergence, memes predictions RF). SEUL le critere de
    validation change : au lieu du portail booleen "> alpha", on garde la cellule
    si son score de fiabilite continu r(idx) >= tau.

    tau dans [0,1] : 0 = on garde tout (aucun rejet), 1 = on n'accepte qu'une ligne
    dont TOUTE la masse d'interaction repose sur des features observees.

    Retour : (df_filled, missing_rows_index, n_missing_rows, pct_missing_rows)
             (+ dict reliab_log si return_reliability=True)
    """
    data_copy = data.copy()
    numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
    df_numeric = data_copy[numeric_cols]

    df_filled = df_numeric.copy()
    for col in df_filled.columns:                         
        df_filled[col] = df_filled[col].fillna(df_filled[col].mean())


    observed_init = df_numeric.notna().copy()
    missing_rows_index = []
    reliab_log = []
    mean_nrmse_prec, mean_nrmse = 0.0, -1.0

    for iteration in range(max_iter):
        df_numeric = data_copy[numeric_cols]
        df_filled_prec = df_filled.copy()
        num_cols_prec, num_missing_cols = -1, -2

        while (num_cols_prec != num_missing_cols) and (num_missing_cols != 0):
            sorted_columns = df_numeric[numeric_cols].isna().sum() \
                .sort_values(ascending=True).index.tolist()

            for col in sorted_columns:
                missing_mask = df_numeric[col].isnull()
                if missing_mask.sum() == 0:
                    continue
                other_cols = df_numeric.columns.drop(col)
                X_train = df_filled.loc[~missing_mask, other_cols]
                y_train = df_filled.loc[~missing_mask, col]

                is_cat = col in categorical_columns
                rf = (RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
                      if is_cat else
                      RandomForestRegressor(n_estimators=100, random_state=42))
                rf.fit(X_train, y_train)

                X_pred = df_filled.loc[missing_mask, other_cols]
                if len(X_pred) == 0:
                    continue
                feat_names = list(X_pred.columns)

                forest_ok = all((est.tree_.feature >= 0).any() for est in rf.estimators_)
                if not forest_ok:
                    for key in range(len(X_pred)):
                        idx = X_pred.index[key]
                        pv = rf.predict(X_pred.iloc[[key]])[0]
                        df_filled.loc[idx, col] = pv
                        df_numeric.loc[idx, col] = pv
                    continue

                explainer = (shapiq.TreeExplainer(model=rf, max_order=max_order,
                                                  index=index, class_index=1)
                             if is_cat else
                             shapiq.TreeExplainer(model=rf, max_order=max_order, index=index))

                Xp = X_pred.to_numpy(dtype=float)
                for key in range(len(Xp)):
                    idx = X_pred.index[key]
                    iv = explainer.explain(Xp[key])
                    
                    obs_mask = np.array([bool(observed_init.loc[idx, c]) for c in other_cols])
                    r = reliability_from_interactions(iv, feat_names, obs_mask,
                                                      order_weight=order_weight)
                    pv = rf.predict(X_pred.iloc[[key]])[0]
                    df_filled.loc[idx, col] = pv
                    if r["reliability"] >= tau:            # critere continu
                        df_numeric.loc[idx, col] = pv      
                    # sinon : valeur ecrite dans df_filled mais ligne reste incomplete
                    if return_reliability:
                        reliab_log.append({"iter": iteration, "col": col, "idx": idx,
                                           **r})

                num_cols_prec = num_missing_cols
                num_missing_cols = df_numeric.isna().sum().sum()

        missing_rows_index_prec = missing_rows_index
        missing_rows_index = df_numeric[df_numeric.isna().any(axis=1)].index.tolist()
        # convergence : meme critere que l'original (via compute_nrmse externe)
        try:
            nrmse_result = compute_nrmse(df_filled_prec, df_filled)  # noqa: F821
            mean_nrmse_prec, mean_nrmse = mean_nrmse, nrmse_result.mean()
            if iteration != 0 and mean_nrmse_prec < mean_nrmse:
                df_filled = df_filled_prec.copy()
                missing_rows_index = missing_rows_index_prec
                break
        except NameError:
            pass 

    out = (df_filled, missing_rows_index, len(missing_rows_index),
           len(missing_rows_index) / len(data_copy) * 100)
    if return_reliability:
        return out + (pd.DataFrame(reliab_log),)
    return out
