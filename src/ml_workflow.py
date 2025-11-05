# Description :
#     Ce fichier contiendra toutes les fonctions du pipeline :
#     - Nettoyage des données
#     - Gestion des valeurs manquantes
#     - Normalisation
#     - Split train/test
#     - Entraînement de plusieurs modèles
#     - Sélection de variables
#     - Évaluation des performances



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# 1. Chargement des données

def load_data(filepath: str) -> pd.DataFrame:
    
    """
    Charge le dataset à partir du chemin donné.
    
    Args:
        filepath (str): chemin du fichier CSV.
    Returns:
        pd.DataFrame: le dataframe chargé.
    """
    pass

# 2. Prétraitement des données

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Gère les valeurs manquantes, encode les variables si besoin, 
    et normalise les données.
    
    Args:
        df (pd.DataFrame): le dataframe brut.
    Returns:
        pd.DataFrame: le dataframe prétraité.
    """
    pass



# 3. Séparation train/test

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    
    """
    Sépare le dataset en ensembles d’entraînement et de test.
    
    Args:
        df (pd.DataFrame): le dataframe prétraité.
        target_column (str): nom de la colonne cible.
        test_size (float): proportion de test.
        random_state (int): graine aléatoire.
    Returns:
        X_train, X_test, y_train, y_test
    """
    pass



# 4. Entraînement des modèles

def train_models(X_train, y_train):
    """
    Entraîne plusieurs modèles de classification (ex: logistic regression, SVM, random forest).
    
    Args:
        X_train (array-like): données d’entraînement.
        y_train (array-like): labels d’entraînement.
    Returns:
        dict: modèles entraînés avec leur nom en clé.
    """
    pass



# 5. Évaluation des modèles

def evaluate_models(models: dict, X_test, y_test):
    """
    Évalue les modèles entraînés avec différentes métriques (accuracy, f1-score, ROC-AUC, etc.)
    
    Args:
        models (dict): dictionnaire des modèles entraînés.
        X_test (array-like): features de test.
        y_test (array-like): labels de test.
    Returns:
        pd.DataFrame: tableau récapitulatif des performances.
    """
    pass


# 6. Sélection de variables

def select_features(model, X_train, y_train, threshold: float = 0.01):
    """
    Effectue une sélection de variables (basée sur l’importance ou sur une méthode statistique).
    
    Args:
        model: modèle entraîné.
        X_train: données d’entraînement.
        y_train: labels d’entraînement.
        threshold (float): seuil minimal d’importance pour garder la variable.
    Returns:
        list: noms des variables sélectionnées.
    """
    pass


# 7. Fonction principale (pipeline complet)

def run_full_pipeline(filepath: str, target_column: str):
    """
    Fonction principale pour exécuter l’ensemble du pipeline ML sur un dataset.
    
    Args:
        filepath (str): chemin vers le dataset.
        target_column (str): nom de la variable cible.
    Returns:
        None (affiche les performances finales)
    """
    pass



