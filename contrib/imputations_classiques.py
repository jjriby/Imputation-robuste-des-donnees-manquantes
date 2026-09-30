"""Imputations de référence : moyenne, médiane, MICE, KNN et MissForest."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (active MICE)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import IterativeImputer, KNNImputer
from .data_evaluation import compute_nrmse


def mean_imputation(data,categorical_columns):
    """
    Imputation simple par la moyenne (colonnes numériques).

    Les colonnes catégorielles (déjà encodées en entiers) sont ensuite arrondies à l'entier
    le plus proche pour éviter des valeurs non entières après opérations numériques.

    Paramètres
    ----------
    data : pandas.DataFrame
        Données avec NaN.
    categorical_columns : list[str] | pandas.Index
        Colonnes catégorielles encodées.

    Retours
    -------
    pandas.DataFrame
        DataFrame imputé.
    """

    df_filled = data.fillna(data.mean(numeric_only=True))
    df_filled[categorical_columns] = df_filled[categorical_columns].round()  # Arrondir à l'entier le plus proche

    return df_filled

def median_imputation(data,categorical_columns):
    """
    Imputation simple par la médiane (colonnes numériques).

    Les colonnes catégorielles encodées sont arrondies à l'entier le plus proche.

    Paramètres
    ----------
    data : pandas.DataFrame
        Données avec NaN.
    categorical_columns : list[str] | pandas.Index
        Colonnes catégorielles encodées.

    Retours
    -------
    pandas.DataFrame
        DataFrame imputé.
    """

    df_filled = data.fillna(data.median(numeric_only=True))
    df_filled[categorical_columns] = df_filled[categorical_columns].round()  # Arrondir à l'entier le plus proche

    return df_filled

def mice_imputation(data,categorical_columns) : 
    """
    Imputation MICE (IterativeImputer de scikit-learn).

    Le principe : chaque variable est imputée à tour de rôle en l'expliquant par les autres
    variables, et ce de manière itérative jusqu'à convergence.

    Ici, l'imputation est appliquée uniquement sur les colonnes numériques. Les colonnes
    catégorielles encodées sont arrondies à l'entier.

    Paramètres
    ----------
    data : pandas.DataFrame
        Données avec NaN.
    categorical_columns : list[str] | pandas.Index
        Colonnes catégorielles encodées.

    Retours
    -------
    pandas.DataFrame
        DataFrame imputé.
    """

    # 1. Remplissage par MICE
    imputer = IterativeImputer(random_state=42)
    # On fit-transform seulement sur les colonnes numériques (comme dans ton remplissage par moyenne)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    df_filled = data.copy()
    # Appliquer l'imputation seulement sur les colonnes numériques
    df_filled_numeric = imputer.fit_transform(df_filled[numeric_cols])
    df_filled[numeric_cols] = df_filled_numeric
    df_filled[categorical_columns] = df_filled[categorical_columns].round()  # Arrondir à l'entier le plus proche

    return df_filled

def knn_imputation(data,categorical_columns) : 
    """
    Imputation KNN (KNNImputer de scikit-learn).

    Chaque valeur manquante est remplacée par une moyenne des 'n_neighbors' voisins les plus
    proches (distance calculée dans l'espace des variables).

    Ici, l'imputation est appliquée uniquement sur les colonnes numériques. Les colonnes
    catégorielles encodées sont arrondies à l'entier.

    Paramètres
    ----------
    data : pandas.DataFrame
        Données avec NaN.
    categorical_columns : list[str] | pandas.Index
        Colonnes catégorielles encodées.

    Retours
    -------
    pandas.DataFrame
        DataFrame imputé.
    """

    
    # 1. Remplissage par KNN
    imputer = KNNImputer(n_neighbors=20)  # tu peux ajuster n_neighbors (par défaut 5)
    data_copy = data.copy()
    
    numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
    df_filled = data_copy.copy()
    
    df_filled_numeric = imputer.fit_transform(df_filled[numeric_cols])
    df_filled[numeric_cols] = df_filled_numeric
    df_filled[categorical_columns] = df_filled[categorical_columns].round()  # Arrondir à l'entier le plus proche

    return df_filled

def miss_forest_imputation(data,max_iterations,categorical_columns) : 
    """
    Implémentation de MissForest (imputation par forêts aléatoires).

    Algorithme (simplifié) :
    1) Initialisation : remplissage des NaN par la moyenne colonne par colonne.
    2) Boucles itératives :
       - On traite les colonnes dans l'ordre croissant du nombre de NaN.
       - Pour chaque colonne cible, on entraîne un RandomForest (Regressor ou Classifier selon
         le type/nom de colonne) sur les lignes non manquantes, puis on prédit les valeurs
         manquantes.
    3) Critère d'arrêt : si la NRMSE moyenne entre itérations augmente, on revient à l'itération
       précédente et on stoppe.

    Paramètres
    ----------
    data : pandas.DataFrame
        Données avec NaN.
    max_iterations : int
        Nombre max d'itérations.
    categorical_columns : list[str] | pandas.Index
        Colonnes traitées comme catégorielles (RandomForestClassifier).

    Retours
    -------
    pandas.DataFrame
        DataFrame imputé (colonnes numériques du DataFrame d'entrée).
    """


    categorical_columns = categorical_columns

    missing_rows_index = 0 
    data_copy = data.copy()

    numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
    df_numeric = data_copy[numeric_cols]
    
    df_filled = df_numeric.copy()
    for col in df_filled.columns:
        #df_filled[col].fillna(df_filled[col].mean(), inplace=True)
        df_filled[col] = df_filled[col].fillna(df_filled[col].mean())


    max_iter = max_iterations
    mean_nrmse_prec = 0
    mean_nrmse = -1
    
    df_non_supp = data_copy.copy()
    for iteration in range(max_iter):
        # print(f"--- Iteration {iteration+1} ---")
    
        df_numeric = data_copy[numeric_cols]
        df_filled_prec = df_filled.copy()
        col_missing_counts = df_numeric[numeric_cols].isna().sum()
        sorted_columns = col_missing_counts.sort_values(ascending=True).index.tolist()
        for col in sorted_columns:
    
            missing_mask = df_numeric[col].isnull()
            if missing_mask.sum() == 0:
                continue
            other_cols = df_numeric.columns.drop(col)
            train_mask = ~missing_mask & df_numeric[other_cols].notna().all(axis=1)
            pred_mask = missing_mask & df_filled[other_cols].notna().all(axis=1)
            
            num_true = missing_mask.sum()
            X_train = df_filled.loc[~missing_mask, other_cols]
            y_train = df_filled.loc[~missing_mask, col]
    
            if(col in categorical_columns) : 
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
            else :
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            X_pred = df_filled.loc[pred_mask, other_cols]
    
            preds = rf.predict(X_pred)
            df_filled.loc[pred_mask, col] = preds  # déjà un array NumPy → bon
            df_numeric.loc[pred_mask,col] = preds
                            
                    
        missing_rows_index_prec = missing_rows_index
        nrmse_result = compute_nrmse(df_filled_prec, df_filled)
        mean_nrmse_prec = mean_nrmse
        mean_nrmse = nrmse_result.mean()
        if(iteration != 0 and mean_nrmse_prec<mean_nrmse) : 
            df_filled = df_filled_prec.copy()
            missing_rows_index = missing_rows_index_prec
            break
    return df_filled
