"""Données et évaluation"""

from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def load_data_set(url):
    """
    Charge un dataset depuis un chemin/URL CSV (ou fichier texte) et applique quelques regles
    specifiques a certains jeux de donnees utilises dans les experiences.

    Cas geres (identifies par le NOM DE FICHIER) :
    - 'ecoli.data' : fichier a colonnes separees par espaces, avec noms de colonnes fournis.
    - 'credit_approval.data' : fichier CSV sans en-tete ; des noms de colonnes A1..A16 sont fournis.
    - Autre : lecture via 'pandas.read_csv'.

    Le nom des colonnes est ensuite normalise (remplacement des espaces par des underscores).

    Parametres
    ----------
    url : str
        Chemin local ou URL vers le fichier.

    Retours
    -------
    pandas.DataFrame
        DataFrame pret a etre pre-traite.
    """

    fname = os.path.basename(url)

    if fname == "ecoli.data":
        columns = [
            "Sequence_Name",
            "mcg",
            "gvh",
            "lip",
            "chg",
            "aac",
            "alm1",
            "alm2",
            "class"
        ]
        # sep=r"\s+" est l'equivalent moderne et portable de delim_whitespace=True.
        df = pd.read_csv(url, sep=r"\s+", names=columns)

    elif fname == "credit_approval.data":
        columns = [
            "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
            "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"
        ]
        df = pd.read_csv(url, sep=",", names=columns)

    elif fname == "adult.csv":
        df = pd.read_csv(url)
        df_sample = df.sample(n=350, random_state=42)
        df = df_sample.reset_index(drop=True)

    else:
        df = pd.read_csv(url)

    # Replace spaces in column names with underscores (if any left)
    df.columns = df.columns.str.replace(" ", "_", regex=False)

    return df

def label_encoding(data) : 
    """
    Encode les colonnes catégorielles (type 'object') avec 'LabelEncoder'.

    Cette étape transforme les catégories en entiers (0..K-1) afin de pouvoir appliquer les
    méthodes d'imputation qui attendent des valeurs numériques.

    Paramètres
    ----------
    data : pandas.DataFrame
        Données d'entrée (idéalement après suppression de colonnes inutiles).

    Retours
    -------
    (data_encoded, categorical_columns) : (pandas.DataFrame, pandas.Index)
        - 'data_encoded' : DataFrame modifié (encodage effectué en place sur une copie locale).
        - 'categorical_columns' : liste/index des colonnes détectées comme catégorielles.
    """
    label_encoders = {}
    categorical_columns = data.select_dtypes(include='object').columns
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    # Afficher tous les mappings
    # for col, mapping in label_encoders.items():
    #     print(f"\n Mapping pour la colonne '{col}':")
    #     for original, encoded in mapping.items():
    #         print(f"  {original} --> {encoded}")
    return data,categorical_columns

def generate_missing(data,missing_rate) : 
    """
    Introduit artificiellement des valeurs manquantes (NaN) dans un DataFrame.

    On choisit aléatoirement 'missing_rate * (n_lignes * n_colonnes)' cellules et on remplace
    leur valeur par NaN (si la cellule n'est pas déjà NaN). Les valeurs originales sont
    mémorisées dans un dictionnaire afin d'évaluer ensuite la qualité de l'imputation.

    Paramètres
    ----------
    data : pandas.DataFrame
        Données d'entrée (sera modifié en place dans la fonction ; en pratique, on passe souvent
        une copie : 'df_original.copy()').
    missing_rate : float
        Pourcentage global de cellules à rendre manquantes (entre 0 et 1).

    Retours
    -------
    (data_missing, true_values) : (pandas.DataFrame, dict[tuple[int,str], Any])
        - 'data_missing' : DataFrame avec des NaN.
        - 'true_values' : dictionnaire {(index_ligne, nom_colonne) -> valeur_originale}.
    """
# data after preprocessing 
    df_original = data.copy()
    # Colonnes autorisées pour l'introduction de NaN
    allowed_cols = data.columns
    
    # Définir le pourcentage global de valeurs manquantes souhaité (ex. 50%)
    missing_rate = missing_rate
    
    # Calcul du nombre total de cellules modifiables
    total_cells = len(data) * len(allowed_cols)
    
    # Nombre de cellules à rendre manquantes
    n_missing_cells = int(missing_rate * total_cells)
    
    # Générer aléatoirement des positions (ligne, colonne) à rendre manquantes
    rows = np.random.choice(data.index, size=n_missing_cells, replace=True)
    cols = np.random.choice(allowed_cols, size=n_missing_cells, replace=True)
    
    # Stocker les vraies valeurs
    true_values = {}
    
    # Introduire les NaN
    for row, col in zip(rows, cols):
        if pd.notna(data.at[row, col]):
            true_values[(row, col)] = data.at[row, col]
            data.at[row, col] = np.nan
    return data,true_values


def Eval_NRMSE_ACC(categorical_columns, orignal_data, df_filled, true_values=None) : 
    """
   

    Évalue une imputation sur deux critères :
    - NRMSE (Normalized RMSE) sur les colonnes numériques
    - Accuracy sur les colonnes catégorielles (encodées en entiers)


    Paramètres
    ----------
    categorical_columns : list[str] | pandas.Index
        Colonnes catégorielles (après label encoding).
    orignal_data : pandas.DataFrame
        Données complètes (avant injection des NaN), utilisées comme vérité terrain.
    df_filled : pandas.DataFrame
        Données imputées.
    true_values : dict, optionnel
        Dictionnaire de vérité terrain renvoyé par 'generate_missing()'.

    Retours
    -------
    (nrmse_globale, global_accuracy) : (float, float)
        - NRMSE moyen sur les colonnes numériques concernées.
        - Accuracy globale sur les colonnes catégorielles concernées.
    """
# les données catégorielle encoder par lebel encoding , les données original avant génération des manquantes et avec prétraitments 

    # 1. Définir les colonnes numériques automatiquement en excluant les colonnes catégorielles
    exclude_columns = categorical_columns 
    df_original = orignal_data.copy()
    numeric_cols = [col for col in df_original.columns if col not in exclude_columns and df_original[col].dtype != 'object']
    
    
    # 3. Calculer le RMSE global sur les colonnes numériques
    errors = {}
    nombre_indv = {}
    for col in numeric_cols : 
        nombre_indv[col] = 0
        errors[col] = 0
    
    for (idx, col), true_val in true_values.items():
        if col in numeric_cols:
            imputed_val = df_filled.at[idx, col]
            true_val_scaled = df_original.at[idx, col]
            error = ((imputed_val - true_val_scaled) ** 2)
            errors[col] = errors[col]+error
            nombre_indv[col] = nombre_indv[col]+1
    nrmse_col = {}
    for col in numeric_cols : 
        if(nombre_indv[col] !=0) :
            rmse_col = np.sqrt(errors[col] / nombre_indv[col])
            nrmse_col[col] = rmse_col / df_original[col].std()
    
    nrmse_globale = sum(nrmse_col.values()) / len(nrmse_col)
    # print(f"NRMSE global : {nrmse_globale:.4f}")
    true_vals_all = []
    imputed_vals_all = []
    
    # Parcourir les colonnes catégorielles
    for col in categorical_columns:
        # Trouver les indices des valeurs pour cette colonne
        indices = [idx for (idx, c) in true_values if c == col]
        if indices:
            true_vals = df_original.loc[indices, col]  # Valeurs réelles (non manquantes)
            imputed_vals = df_filled.loc[indices, col]  # Valeurs imputées
            
            # Ajouter les valeurs réelles et imputées à la liste globale
            true_vals_all.extend(true_vals)
            imputed_vals_all.extend(imputed_vals)
    
    # Calculer l'accuracy globale
    global_accuracy = accuracy_score(true_vals_all, imputed_vals_all)
    
    # Afficher le résultat
    # print(f"\n Accuracy globale : {global_accuracy:.4f}")

    return nrmse_globale,global_accuracy

def compute_nrmse(df1, df2):
    """
    Calcule le NRMSE (RMSE normalisé par l'écart-type) colonne par colonne entre deux DataFrames.
    Cette fonction est utilisée pour évalue la converngence des méthodes MissForest et MisShapForest

    Paramètres
    ----------
    df1 : pandas.DataFrame
        Vérité terrain.
    df2 : pandas.DataFrame
        Prédictions / valeurs imputées.

    Retours
    -------
    pandas.Series
        Série indexée par nom de colonne : NRMSE de chaque colonne numérique.
    """

    # Vérification dimensions
    assert df1.shape == df2.shape, "Les deux DataFrames doivent avoir les mêmes dimensions"
    assert all(df1.columns == df2.columns), "Les colonnes doivent être identiques"

    nrmse_per_column = {}

    for col in df1.columns:
        # On ne traite que les colonnes numériques
        if pd.api.types.is_numeric_dtype(df1[col]):
            y_true = df1[col].values
            y_pred = df2[col].values

            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            range_ = np.std(y_true)

            if range_ != 0:
                nrmse = rmse / range_
            else:
                nrmse = 0  # ou np.nan si tu préfères signaler ce cas

            nrmse_per_column[col] = nrmse

    return pd.Series(nrmse_per_column)

def EvalV2_NRSME(missing_rows_index,categorical_columns,df_original,true_values,df_filled) : 
    """
    Évaluation NRMSE + Accuracy (version 2) en excluant certaines lignes (non faible selon notre méthode).

    Cette version ignore les positions (idx, col) dont l'index de ligne appartient à
    'missing_rows_index' (lignes encore incomplètes/à supprimer dans l'approche SHAP).

    Paramètres
    ----------
    missing_rows_index : list[int]
        Indices de lignes à exclure de l'évaluation.
    categorical_columns : list[str] | pandas.Index
        Colonnes catégorielles encodées.
    df_original : pandas.DataFrame
        Données complètes (vérité terrain).
    true_values : dict
        Dictionnaire {(idx, col) -> valeur_originale} produit par 'generate_missing()'.
    df_filled : pandas.DataFrame
        Données imputées.

    Retours
    -------
    (nrmse_globale, global_accuracy) : (float, float)
    """

    
    exclude_columns = categorical_columns 
    
    numeric_cols = [col for col in df_original.columns if col not in exclude_columns and df_original[col].dtype != 'object']
    
    
    # 3. Calculer le RMSE global sur les colonnes numériques
    errors = {}
    nombre_indv = {}
    for col in numeric_cols : 
        nombre_indv[col] = 0
        errors[col] = 0 
    cleaned_true_values = {
        (idx, col): val
        for (idx, col), val in true_values.items()
        if idx not in missing_rows_index
    }

    # print(cleaned_true_values)
    for (idx, col), true_val in cleaned_true_values.items():
        # print(idx, col)
        if col in numeric_cols:
            # print((idx, col), true_val)
            imputed_val = df_filled.at[idx, col]
            true_val_scaled = df_original.at[idx, col]
            # print("imputed : ",imputed_val)
            # print("true : ",true_val_scaled)

            # true_val_scaled = df_fill_vrai_val.at[idx, col]

            error = ((imputed_val - true_val_scaled) ** 2)
            errors[col] = errors[col]+error
            nombre_indv[col] = nombre_indv[col]+1
    nrmse_col = {}
    all_columns = list(errors.keys())
    # print(all_columns)
    for col in all_columns :
        if(nombre_indv[col]!=0) : 
            rmse_col = np.sqrt(errors[col] / nombre_indv[col])
            nrmse_col[col] = rmse_col / df_original[col].std()    

        # else : 
        #     print(col)
        #     print(nombre_indv[col])
        #     print(errors[col])
    # print(nrmse_col)
    nrmse_globale = sum(nrmse_col.values()) / len(nrmse_col)
    # print(f"NRMSE global : {nrmse_globale:.4f}")
    
    true_vals_all = []
    imputed_vals_all = []
    
    # Parcourir les colonnes catégorielles
    for col in categorical_columns:
        # Trouver les indices des valeurs pour cette colonne
        indices = [idx for (idx, c) in cleaned_true_values if c == col]
        if indices:
            df_org = df_original.drop(missing_rows_index).copy()
            df_fill = df_filled.drop(missing_rows_index).copy()
            true_vals = df_org.loc[indices, col]  # Valeurs réelles (non manquantes)
            imputed_vals = df_fill.loc[indices, col]  # Valeurs imputées
            
            # Ajouter les valeurs réelles et imputées à la liste globale
            true_vals_all.extend(true_vals)
            imputed_vals_all.extend(imputed_vals)
    
    # Calculer l'accuracy globale
    global_accuracy = accuracy_score(true_vals_all, imputed_vals_all)
    
    # Afficher le résultat
    # print(f"\n Accuracy globale : {global_accuracy:.4f}")

    return(nrmse_globale,global_accuracy)

def safe_eval_v2(missing_idx, categorical_columns,
                  df_original, true_values, df_filled):
    """
    Wrapper robuste autour de EvalV2_NRSME : si toutes les lignes ont ete
    rejetees (df_filled apres suppression est vide), on retourne (NaN, NaN).
    """
    if len(missing_idx) >= len(df_original):
        # toutes les lignes seraient supprimees -> non evaluable
        return float("nan"), float("nan")
    try:
        return EvalV2_NRSME(
            missing_rows_index=missing_idx,
            categorical_columns=categorical_columns,
            df_original=df_original,
            true_values=true_values,
            df_filled=df_filled,
        )
    except Exception:
        return float("nan"), float("nan")
