# Description :
#     Ce fichier contient toutes les fonctions du pipeline :
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
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


# 1. Chargement des données

def load_data(filepath: str) -> pd.DataFrame:
    
    """
    Charge le dataset en fonction de son type (CSV ou .data) et retourne un DataFrame pandas

    On gère ici deux cas :
    - Dataset de type CSV ( Diabetes)
    - Dataset Spambase (.data sans header, avec .names séparé)

    Args:
        filepath (str)
    Returns:
        pd.DataFrame
    """

    

    # Cas du fichier CSV Diabetes
    
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        print(f"Dataset CSV chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Cas du fichier .data Spambase
    
    elif filepath.endswith(".data"):
        
        # On reconstruit le chemin du fichier .names correspondant
        names_path = filepath.replace(".data", ".names")

        # Lecture des noms de colonnes depuis le .names
        with open(names_path, "r") as f:
            lines = f.readlines()

        # Les noms de colonnes se trouvent après une ligne vide dans .names
        col_names = []
        for line in lines:
            if ":" in line and not line.startswith("|"):
                col_names.append(line.split(":")[0].strip())

        # On ajoute la colonne cible "spam"
        col_names.append("spam")

        # Lecture du fichier principal avec les bons noms de colonnes
        df = pd.read_csv(filepath, header=None, names=col_names)
        print(f"Dataset Spambase chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    else:
        raise ValueError(" Format de fichier non reconnu ")

    print(" apercu des données :")
    print(df.head())
    return df

# 2. Prétraitement des données

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite les données :
    - Gère les valeurs manquantes
    - Normalise toutes les colonnes numériques (hors variable cible)
    - Encode les variables catégorielles si besoin (ici, aucune)
    
    Args:
        df (pd.DataFrame)
    Returns:
        pd.DataFrame
    """

    df_clean = df.copy()

    # Gestion des valeurs manquantes
    
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        print(f"{missing_count} valeurs manquantes détectées → imputation moyenne.")
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    else:
        print("Aucune valeur manquante détectée.")

    # Détection de la colonne cible
    target_col = None
    for possible_target in ["spam", "Diabetes_binary", "Outcome", "class"]:
        if possible_target in df_clean.columns:
            target_col = possible_target
            break

    if target_col is None:
        raise ValueError(" Impossible d’identifier la colonne cible dans le dataset")

    print(f" Colonne cible détectée : '{target_col}'")

    # Normalisation des variables numériques
    features = df_clean.drop(columns=[target_col])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Reconstituer le DataFrame normalisé
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled[target_col] = df_clean[target_col].values
    print(f"Normalisation terminée : {df_scaled.shape[1]} variables normalisées.")
    return df_scaled


# 3. Séparation train/test

def split_data(df: pd.DataFrame, target_column: str = None, test_size: float = 0.2, random_state: int = 42):
    """
    Sépare le dataset en ensembles d'entraînement et de test.

    Args:
        df (pd.DataFrame): le dataframe prétraité par la fct preprocess_data
        target_column (str, optional): nom de la variable cible. Si None, essaye de la détecter automatiquement
        test_size (float): proportion du jeu de test (ex: 0.2 = 20%)
        random_state (int): graine aléatoire pour la reproductibilité

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """

    df_copy = df.copy()

    # Identifier la colonne cible si non précisée
    if target_column is None:
        for possible_target in ["spam", "Diabetes_binary", "Outcome", "class"]:
            if possible_target in df_copy.columns:
                target_column = possible_target
                break

    if target_column is None:
        raise ValueError(" Impossible d'identifier la colonne cible automatiquement.")

    print(f"Colonne cible utilisée : '{target_column}'")

    # Séparer X et y
    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    print(f" Split effectué : {X_train.shape[0]} train / {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test




# 4. Entraînement des modèles

def train_models(X_train, X_test, y_train, y_test):
    """
    Entraîne plusieurs modèles de classification et compare leurs performances
    Modèles utilisés ( à changer si besoin , selon les performances qu on aura ) :
      - Random Forest
      - KNN
      - Réseau de neurones "simple" (MLP)
    
    Args:
        X_train, X_test, y_train, y_test : ensembles d'entraînement et de test

    Returns:
        dict: modèles entraînés
        dict: scores de performance (accuracy et F1 score )
    """

    # On définit les modèles à tester
    models = {
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=800, n_jobs=-1, max_leaf_nodes=24 , weight = 'balanced'),
        "KNN": KNeighborsClassifier(n_neighbors=10 , random_state=42 , n_jobs=-1),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42 ,  n_jobs=-1)
    }

    trained_models = {}
    results = {}

    # On entraine chaque modèle et on évalue ses performances
    for name, model in models.items():
        print(f"\n Entraînement du modèle : {name}")
        model.fit(X_train, y_train)  
        y_pred = model.predict(X_test)  
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        trained_models[name] = model
        results[name] = {"accuracy": acc, "f1_score": f1}

        print(f" {name} entraîné — Accuracy: {acc:.3f} | F1: {f1:.3f}")

    print("\n Résumé des performances :")
    for name, scores in results.items():
        print(f"{name:<12} → Accuracy: {scores['accuracy']:.3f} | F1: {scores['f1_score']:.3f}")

    return trained_models, results



# 5. Évaluation des modèles

def evaluate_models(models, X_test, y_test):
    """
    Évalue plusieurs modèles sur un jeu de test :
    - Affiche la matrice de confusion
    - Affiche le rapport de classification
    - Résume les scores globaux
    
    Args:
        models (dict): dictionnaire {nom: modèle_entraîné}
        X_test (DataFrame): features du test set
        y_test (Series): labels du test set
    
    Returns:
        dict: dictionnaire des rapports (texte + matrices)
    """

    evaluations = {}

    for name, model in models.items():
        print(f"\n🔍 Évaluation du modèle : {name}")
        y_pred = model.predict(X_test)

        # Rapport de la classification et matrice de confusion
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # Affichage résumé
        print("Matrice de confusion :")
        print(cm)
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))

        # Stocker les résultats
        evaluations[name] = {
            "classification_report": report,
            "confusion_matrix": cm
        }
    return evaluations


# 6. Sélection de variables

def select_features(model, X_train, top_n=10):
    """
    Sélectionne les variables les plus importantes selon un modèle de type RandomForest.

    Args:
        model: modèle entraîné (doit avoir un attribut 'feature_importances_')
        X_train (DataFrame): données d'entraînement (features)
        top_n (int): nombre de variables les plus importantes à afficher

    Returns:
        DataFrame: tableau des variables les plus importantes
    """

    # Vérification que le modèle permet l'analyse des features
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(" Le modèle choisi ne possède pas 'feature_importances_' ")

    # Extraire les importances et les trier
    importances = model.feature_importances_
    feature_names = np.array(X_train.columns)
    sorted_idx = np.argsort(importances)[::-1] 

    top_features = pd.DataFrame({
        "Feature": feature_names[sorted_idx][:top_n],
        "Importance": importances[sorted_idx][:top_n]
    })

    print(f"\n Top {top_n} variables les plus importantes :")
    print(top_features.to_string(index=False))

    return top_features

# 7. Fonction principale ( pour suivre le pipeline complet)

def run_full_pipeline(filepath: str, target_column: str = None):
    """
    Exécute l'ensemble du pipeline Machine Learning sur un dataset :
    - Chargement des données
    - Prétraitement
    - Split train/test
    - Entraînement de plusieurs modèles
    - Évaluation
    - Sélection des features importantes (si applicable)

    Args:
        filepath (str): chemin vers le dataset
        target_column (str, optional): nom de la variable cible (sinon détection automatique)

    Returns:
        None (affiche les résultats)
    """

    df = load_data(filepath)

    
    df_clean = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df_clean, target_column=target_column)

    models, results = train_models(X_train, X_test, y_train, y_test)

    
    evaluate_models(models, X_test, y_test)

    # Sélection des variables importantes (si applicable)
    
    if "RandomForest" in models:
        print("\n Analyse des variables importantes avec le modèle RandomForest :")
        select_features(models["RandomForest"], X_train, top_n=10)
    else:
        print(" Le modèle RandomForest n'a pas été entraîné : pas d'analyse des variables importantes possible ")
    
    


