"""
Ce fichier rassemble l'ensemble du pipeline utilisé pour nos deux notebooks de classification.
L’idée est d’avoir un seul fichier.py qui "centralise"  :
- le chargement des données
- la préparation (nettoyage, normalisation, PCA optionnelle)
- la split train/test
- l’entraînement des modèles
- les évaluations
- la sélection de variables importantes
- ainsi que toutes les fonctions de visualisation utilisées dans les notebooks


"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path  
from typing import Dict, Tuple, List, Union
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

# Liste des noms possibles pour la colonne cible selon nos datasets
TARGET_CANDIDATES = ["spam", "Diabetes_binary", "Outcome", "class"]



####  Chargement du fichier de données (si CSV ou .data)

def load_data(filepath: str):
    """
    On charge les données en tenant compte des deux formats utilisés dans le projet : CSV pour diabetes et .data pour Spambase
    Pour Spambase, les colonnes sont récupérées automatiquement depuis le fichier .names
    """

    # On vérifie que le fichier existe avant de tenter de le lire
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} est introuvable.")

    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        print(f"Dataset CSV chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    elif filepath.endswith(".data"):
        names_path = filepath.replace(".data", ".names")
        if not os.path.exists(names_path):
            raise FileNotFoundError(f"Fichier .names correspondant introuvable : {names_path}")

        # Logique pour extraire les noms de colonnes du fichier .names
        with open(names_path, "r") as f:
            lines = f.readlines()

        col_names: List[str] = []
        for line in lines:
            if ":" in line and not line.startswith("|"):
                col_names.append(line.split(":")[0].strip())

        col_names.append("spam") # Ajout de la cible Spambase
        
        df = pd.read_csv(filepath, header=None, names=col_names)
        print(f"Dataset Spambase chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    else:
        raise ValueError("Format de fichier non reconnu (utiliser .csv ou .data).")

    print("Aperçu des données :")
    print(df.head())
    return df


#### Prétraitement des données

def preprocess_data(df: pd.DataFrame, target_column: str = None , apply_pca: bool = False, pca_variance: float = 0.95):
    
    """
    Le prétraitement a plusieurs objectifs :
    - isoler la colonne cible si elle existe
    - gérer simplement les valeurs manquantes (moyenne) meme si dans nos 2 datasets il n'y en a pas...
    - normaliser toutes les colonnes numériques
    - appliquer éventuellement une PCA pour réduire la dimension

    La fonction renvoie le jeu de données transformé, ainsi que le modèle PCA si utilisé 
    """

    
    df_clean = df.copy()

    # Si une colonne cible est précisée, on la retire temporairement
    y = None
    if target_column is not None:
        y = df_clean[target_column]
        X = df_clean.drop(columns=[target_column])
    else:
        X = df_clean

    # Vérification des valeurs manquantes, remplacées par la moyenne ( il n'y en a pas dans nos datasets mais il vaut mieux prévoir)
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())
    
    X_processed = X.copy()

    # Rejoint la cible si elle existe
    if y is not None:
        X_processed[target_column] = y.values

    # On renvoie un None pour la PCA, car elle sera gérée après notre split
    return X_processed, None
    


####  Split train / test

def split_data(df: pd.DataFrame,target_column: str | None = None,test_size: float = 0.2,random_state: int = 42 , apply_pca: bool = False, pca_variance: float = 0.95) :
    
    """
    - Split classique : on sépare les features et la cible, puis on applique un train_test_split
    - Si la colonne cible n’est pas indiquée, on tente de la déterminer automatiquement
    
    """

    df_copy = df.copy()

    # Détection automatique de la cible si elle n’est pas fournie
    if target_column is None:
        for possible_target in TARGET_CANDIDATES:
            if possible_target in df_copy.columns:
                target_column = possible_target
                break

    if target_column is None:
        raise ValueError("Impossible d'identifier la colonne cible automatiquement.")

    print(f"Colonne cible utilisée : '{target_column}'")

    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]

    # On effectue un split stratifié ( pour garder la même proportion de classes ) 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    print(f"Split effectué : {X_train.shape[0]} train / {X_test.shape[0]} test")

    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # ---- PCA APRES le scaling ----
    pca_model = None
    if apply_pca:
        pca_model = PCA(n_components=pca_variance)
        X_train_scaled = pca_model.fit_transform(X_train_scaled)
        X_test_scaled = pca_model.transform(X_test_scaled)

        # Convertir en DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=[f"PC{i+1}" for i in range(X_train_scaled.shape[1])])
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=[f"PC{i+1}" for i in range(X_test_scaled.shape[1])])

    return X_train_scaled, X_test_scaled, y_train, y_test


#### Entraînement des modèles 

def train_models(X_train: pd.DataFrame,X_test: pd.DataFrame,y_train: pd.Series,y_test: pd.Series) :
    
    """
    On entraîne ici trois modèles simples mais efficaces :
    - Random Forest
    - KNN
    - Réseau de neurones (MLP)

    Pour chacun, on calcule des métriques de base : accuracy et F1-score.
    """

    
    models = {
        "RandomForest": RandomForestClassifier(
            random_state=42,
            n_estimators=800,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=10,
        ),
        "NeuralNet": MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=500,
            random_state=42,
        )
    }

    trained_models: Dict[str, object] = {}
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        print(f"\nEntraînement du modèle : {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        trained_models[name] = model
        results[name] = {"accuracy": acc, "f1_score": f1}

        print(f"{name} entraîné — Accuracy: {acc:.3f} | F1: {f1:.3f}")
        

    # On rajoute un récapitulatif des performances
    print("\nRésumé des performances :")
    for name, scores in results.items():
        print(f"{name:<12} → Accuracy: {scores['accuracy']:.3f} | F1: {scores['f1_score']:.3f}")

    return trained_models, results



####  Évaluation : matrices de confusion + résumé du rapport

def evaluate_models(models: Dict[str, object],X_test: pd.DataFrame,y_test: pd.Series) :
    
    """
    Pour chaque modèle, on affiche la matrice de confusion et le rapport de classification
    """

    evaluations: Dict[str, Dict[str, object]] = {}

    for name, model in models.items():
        print(f"\n {name} ")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Affichage propre de la matrice de confusion uniquement
        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Classe 0", "Classe 1"],
            yticklabels=["Classe 0", "Classe 1"]
        )
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.title(f"Matrice de confusion - {name}")
        plt.tight_layout()
        plt.show()




#### Cross-validation des modèles

def cross_validate_model(model,X,y,cv: int = 5,scoring: list | tuple = ("accuracy", "precision", "recall", "f1")) :
    
    """
    On applique une validation croisée standard pour évaluer un modèle
    et on affiche la moyenne et l’écart-type pour chaque métrique choisie...
    """

    print(f"\n Validation croisée pour : {model.__class__.__name__}")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    results = {}
    for metric in scoring:
        mean = scores[f"test_{metric}"].mean()
        std = scores[f"test_{metric}"].std()
        results[metric] = (mean, std)
        print(f"→ {metric.capitalize()} : {mean:.4f} ± {std:.4f}")

    return results



#### GridSearch pour optimiser les hyperparamètres

def grid_search_model(model,param_grid: dict,X_train: pd.DataFrame,y_train: pd.Series,scoring: str = "f1",cv: int = 5,n_jobs: int = -1) :
    
    """
    Effectue une recherche exhaustive sur une grille d’hyperparamètres
    On renvoie le meilleur modèle entraîné, les paramètres optimaux,
    et le meilleur score obtenu pendant le cross-validation.
    """

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("\n Résultats du GridSearchCV :")
    print("Meilleurs paramètres :", grid.best_params_)
    print(f"Meilleur score ({scoring}) :", grid.best_score_)

    best_model = grid.best_estimator_

    return best_model, grid.best_params_, grid.best_score_



####  Sélection des features importantes (RandomForest)

def select_features(model: object,X_train: pd.DataFrame,top_n: int = 10) :
    """
    On récupère les variables les plus importantes d’un modèle possédant l’attribut 'feature_importances_' (comme RandomForest)
    """

    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Le modèle choisi ne possède pas 'feature_importances_'.")

    importances = model.feature_importances_
    feature_names = np.array(X_train.columns)
    sorted_idx = np.argsort(importances)[::-1]

    top_features = pd.DataFrame({
        "Feature": feature_names[sorted_idx][:top_n],
        "Importance": importances[sorted_idx][:top_n],
    })

    print(f"\nTop {top_n} variables les plus importantes :")
    print(top_features.to_string(index=False))

    return top_features



#### Visualisation : distribution de la classe cible

def plot_class_distribution(y: pd.Series, title: str = "Répartition de la classe cible") :
    """
    Histogramme simple montrant l’équilibre (ou déséquilibre) des classes.
    """
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'exemples")
    plt.title(title)
    plt.tight_layout()
    plt.show()



#### Heatmap des corrélations

def plot_correlation_heatmap(df: pd.DataFrame,target_column: str | None = None,max_features: int = 30,title: str = "Matrice de corrélation"):
    """
    On visualise les relations entre variables numériques à l’aide d’une heatmap
    Quand il y a trop de colonnes, on réduit l’affichage aux plus corrélées...
    """

    data = df.copy()
    if target_column is not None and target_column in data.columns:
        data = data.drop(columns=[target_column])

    corr = data.corr(numeric_only=True)

    # Réduction si beaucoup de colonnes
    if corr.shape[0] > max_features:
        mean_corr = corr.abs().mean().sort_values(ascending=False)
        top_cols = mean_corr.index[:max_features]
        corr = corr.loc[top_cols, top_cols]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()



#### Visualisation de la PCA 2D
def plot_pca_2d(X: pd.DataFrame,y: pd.Series ) :
    
    """
    Représentation simple en deux dimensions après PCA pour visualiser la séparation des classes.
    """

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.5)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var.)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var.)")
    plt.title("Projection PCA (2D)")
    plt.tight_layout()
    plt.show()



#### Matrices de confusion pour plusieurs modèles
def plot_confusion_matrices(models: Dict[str, object],X_test: pd.DataFrame,y_test: pd.Series,class_names: List[str] | None = None):
    """
    Affiche côte à côte les matrices de confusion de plusieurs modèles pour comparer directement leurs erreurs.
    """

    n_models = len(models)
    plt.figure(figsize=(5 * n_models, 4))

    if class_names is None:
        class_names = ["0", "1"]

    for i, (name, model) in enumerate(models.items(), start=1):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.subplot(1, n_models, i)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.title(f"Matrice de confusion - {name}")

    plt.tight_layout()
    plt.show()



#### Courbes ROC
def plot_roc_curves(models: Dict[str, object],X_test: pd.DataFrame,y_test: pd.Series,title: str = "Courbes ROC"):
    
    """
    Compare graphiquement la capacité des modèles à séparer les classes.
    """

    plt.figure(figsize=(6, 5))

    for name, model in models.items():
        # Tous les modèles n’ont pas predict_proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



#### Courbes Precision–Recall
def plot_precision_recall_curves(models: Dict[str, object],X_test: pd.DataFrame,y_test: pd.Series,title: str = "Courbes Precision-Recall") :
    
    """
    la courbe PR montre mieux la qualité du modèle qu’une ROC classique..
    """

    plt.figure(figsize=(6, 5))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



#### Barplot des variables importantes

def plot_feature_importances_bar(top_features: pd.DataFrame,title: str = "Top features importantes (RandomForest)") :
    
    """
    Représente visuellement les variables les plus importantes identifiées par le RandomForest 
    """

    plt.figure(figsize=(7, 5))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=top_features.sort_values("Importance", ascending=True),
        orient="h",
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()



### Pipeline complet pour le dataset diabete

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




# 4. Entraînement des modèles pour le dataSet spam

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
        "RandomForest": RandomForestClassifier(
            random_state=42, 
            n_estimators=800, 
            n_jobs=-1, 
            max_leaf_nodes=24, 
            class_weight='balanced'  # CORRECTION ICI : 'class_weight' au lieu de 'weight'
        ),
        "KNN": KNeighborsClassifier(n_neighbors=10, n_jobs=-1),  # KNN n'a pas de random_state
        "NeuralNet": MLPClassifier(
            hidden_layer_sizes=(50,), 
            max_iter=500, 
            random_state=42
        )  # MLPClassifier n'a pas de n_jobs
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



 
### Pipeline diabete 



def load_data_and_description(filepath: str):
    
    filepath = Path(filepath)
    
    df = pd.read_csv(filepath)
    print(f"Dataset CSV chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # on examine les 5 premières lignes du dataSet via la methode head()  
    print(df.head())

    # la methode info() nous permet d'obtenir une description des données ( nb de lignes, type des variables et le nb de val non-nulles)
    print(df.info())

    # on constate qu'il n'y a que des feautres de type float représentant des proportions  
    # Histogrammes pour chacune des features 

    cols = [
        "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
        "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
    ]

    plt.figure(figsize=(20,20))

    for i,cols in enumerate(cols,1): 
        plt.subplot(6,4,i)   # 24 emplacement à l'écran
        plt.hist(df[cols],bins=20,edgecolor='black')
        plt.title(cols, fontsize=10)  
        plt.xlabel("")
        plt.ylabel("Fréquence")
    
    plt.subplots_adjust(hspace=0.8, wspace=0.6)  #  espace entre les lignes et colonnes
    plt.show()

    # Affichage de la proportion de diabétique
    
    first_col = df.iloc[:,0]
    nb_diabetiques = (1/len(df))*np.sum(first_col == 1)
    print("Le pourcentage de diabétique est de",nb_diabetiques*100,"%")


    return df

# séparation train/test

def split_dataD(df , test_size = 0.2,random_state=42):
    """
    Sépare le dataset en ensembles d'entraînement et de test.

    Args:
        df (pd.DataFrame): le dataframe prétraité par la fct preprocess_data
        test_size (float): proportion du jeu de test (ex: 0.2 = 20%)
        random_state (int): graine aléatoire pour la reproductibilité

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    # split du jeu de donneés effectué
    X_train,X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state=42,stratify =y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test) 

   
    print(f" Split effectué : {X_train.shape[0]} train / {X_test.shape[0]} test")

    return X_train,y_train, X_test,y_test




def plot_correlation_matrix(df):
    ## matrice de corrélation des variables
    corr_matrix = df.corr()

    # Correlation matrix avec heatmap
    top22_features = (
        corr_matrix["Diabetes_binary"].abs().sort_values(ascending=False).head(23).index
    )

    plt.figure(figsize=(15, 15))
    sns.heatmap(
        corr_matrix.loc[top22_features, top22_features],
        annot=True,
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )
    plt.title("Correlation matrix avec heatmap", fontsize=14)
    plt.show()



def plot_top10_correlated(df):
    ## matrice de corrélation des variables
    corr_matrix = df.corr()

    # Sélection des 10 variables les plus corrélées à la cible
    top10_features = (
        corr_matrix["Diabetes_binary"].abs().sort_values(ascending=False).head(11).index
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix.loc[top10_features, top10_features],
        annot=True,
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )
    plt.title("Top 10 features les plus corrélées à la variable cible", fontsize=14)
    plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

## Précisons qu'ici StandardScaler est utiliser sur tout le dataSet avant le split car l'on souhaite uniquement visualiser les data
## on ne récupérera pas ces donneés pour entrainer les modèles => pas de data leakage 

def scale_and_pca(df):
    ## La PCA est sensible à l’échelle des variables :
    # on veut éviter qu’une feature avec des valeurs grandes (ex : BMI) domine la variance.

    # Séparation des features et de la cible
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    # Standardisation (moyenne = 0, variance = 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(" Données standardisées :",
          "Moyenne (approx) :", X_scaled.mean().round(2),
          "et écart-type (approx) :", X_scaled.std().round(2))

    # on commence avec toutes les composantes
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Variance expliquée par chaque composante via explained_variance_ratio_
    explained_var = pca.explained_variance_ratio_

    plt.figure(figsize=(10,5))
    plt.bar(range(1, len(explained_var) + 1), explained_var * 100)
    plt.xlabel("Composante principale")
    plt.ylabel("Variance expliquée (%)")
    plt.title("Variance expliquée suivant les composantes (PCA)")
    plt.show()

    # Affiche les 2 premières composantes
    print(f"PC1 : {explained_var[0]*100:.2f}%  |  PC2 : {explained_var[1]*100:.2f}%")
    print(f"Variance cumulée (PC1+PC2) : {explained_var[:2].sum()*100:.2f}%")

    # Affiche les 3 premières composantes
    print(f"PC1 : {explained_var[0]*100:.2f}%  |  PC2 : {explained_var[1]*100:.2f}%|  PC3 : {explained_var[2]*100:.2f}%")
    print(f"Variance cumulée (PC1+PC2+PC3) : {explained_var[:3].sum()*100:.2f}%")

    return X_scaled, X_pca, explained_var, y



# fontion pour l'évaluation des modèles utiliser ensuite pour la fonction d'entrainement sur le dataSet diabetes

def evaluation_model(model, X_test, y_test, threshold=0.35):
    """
    Évalue un modèle de classification binaire avec un seuil ajustable.
    Permet de favoriser le Recall au détriment de la précision.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

    if y_prob is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_test, y_prob)

    return metrics

# fonction d'entrainement 

# classificateurs : 



def train_and_evaluate_models(X_train, y_train, X_test, y_test, threshold=0.35):
    """
    Entraîne plusieurs modèles (log reg, RF, XGBoost, Voting),
    les évalue, et retourne aussi les modèles entraînés.

    Retourne :
    - results_df : DataFrame des performances triées
    - models_trained : dictionnaire {nom: modèle entraîné}
    """

    # --- Calcul du ratio original (avant oversampling) ---
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # --- Pipeline commun : StandardScaler → OverSampling ---
    # ceci garantit un espace identique pour tous les modèles
    base_steps = [
        ("oversample", RandomOverSampler(random_state=42))
    ]

    # --- Modèles individuels avec pipelines cohérents ---
    logreg_pipeline = Pipeline(
        base_steps + [
            ("logreg", LogisticRegression(max_iter=600, class_weight="balanced"))
        ]
    )

    rf_pipeline = Pipeline(
        base_steps + [
            ("rf", RandomForestClassifier(
                n_estimators=500,
                max_depth=7,
                min_samples_split=7,
                min_samples_leaf=7,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    xgb_pipeline = Pipeline(
        base_steps + [
            ("xgb", XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=ratio,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42
            ))
        ]
    )

    # --- Liste commune des modèles ---
    models = {
        "Logistic Regression": logreg_pipeline,
        "Random Forest": rf_pipeline,
        "XGBoost": xgb_pipeline
    }

    results = {}
    models_trained = {}

    # --- Entraînement cohérent pour tous ---
    for name, model in models.items():
        model.fit(X_train, y_train)
        models_trained[name] = model
        results[name] = evaluation_model(model, X_test, y_test, threshold=threshold)

    # --- Voting: tous les modèles ont le même espace de features ---
    voting = VotingClassifier(
        estimators=[
            ("logreg", models_trained["Logistic Regression"]),
            ("rf", models_trained["Random Forest"]),
            ("xgb", models_trained["XGBoost"])
        ],
        voting="soft"
    )

    # Voting doit ré-apprendre pour être Fitted
    voting.fit(X_train, y_train)
    models_trained["Voting (soft)"] = voting
    results["Voting (soft)"] = evaluation_model(voting, X_test, y_test, threshold=threshold)

    # --- Tableau final trié ---
    results_df = pd.DataFrame(results).T.sort_values("Recall", ascending=False)
    return results_df, models_trained





# Évaluation des modèles (rapport + matrices)

def evaluate_models(models: Dict[str, object],X_test: pd.DataFrame,y_test: pd.Series) :
    
    """
    Pour chaque modèle, on affiche la matrice de confusion et le rapport de classification
    """

    evaluations: Dict[str, Dict[str, object]] = {}

    for name, model in models.items():
        print(f"\n {name} ")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Affichage propre de la matrice de confusion uniquement
        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Classe 0", "Classe 1"],
            yticklabels=["Classe 0", "Classe 1"]
        )
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.title(f"Matrice de confusion - {name}")
        plt.tight_layout()
        plt.show()


from sklearn.model_selection import cross_validate

def cross_validate_model(model,X,y,cv: int = 5,scoring: list | tuple = ("accuracy", "precision", "recall", "f1")):
    """
    Effectue une validation croisée sur un modèle et affiche plusieurs métriques.
    retourne des dictionnaire contenant les scores moyens et écarts-types
    """

    print(f"\n Validation croisée pour : {model.__class__.__name__}")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    results = {}
    for metric in scoring:
        mean = scores[f"test_{metric}"].mean()
        std = scores[f"test_{metric}"].std()
        results[metric] = (mean, std)
        print(f"→ {metric.capitalize()} : {mean:.4f} ± {std:.4f}")

    return results


def grid_search_model(model,param_grid: dict,X_train: pd.DataFrame,y_train: pd.Series,scoring: str = "f1",cv: int = 5,n_jobs: int = -1):
    """
    Effectue un GridSearchCV pour trouver les meilleurs hyperparamètres.

    output:
        best_model : modèle entraîné avec les meilleurs paramètres
        best_params : dict des meilleurs paramètres
        best_score : meilleur score cross-validation
    """

    grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=cv,n_jobs=n_jobs,verbose=1)
    grid.fit(X_train, y_train)

    print("\n Résultats du GridSearchCV :")
    print("Meilleurs paramètres :", grid.best_params_)
    print(f"Meilleur score ({scoring}) :", grid.best_score_)

    best_model = grid.best_estimator_

    return best_model, grid.best_params_, grid.best_score_

## GridSearch pour diabete pour log reg

def gridsearch_logreg(X_train, y_train):

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=600, class_weight="balanced"))
    ])

    param_grid = {
        "logreg__C": [0.1, 1, 10],
        "logreg__penalty": ["l2"]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="recall",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_res, y_res)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

# GridSearch pour le RF

def gridsearch_random_forest(X_train, y_train):

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    pipeline = Pipeline([
        ("rf", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])

    param_grid = {
        "rf__n_estimators": [200, 400],
        "rf__max_depth": [5, 10],
        "rf__min_samples_split": [5],
        "rf__min_samples_leaf": [3],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="recall",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_res, y_res)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def plot_roc_curves(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str = "Courbes ROC") -> None:
    """
    Affiche les courbes ROC pour plusieurs modèles de classification binaire.
    """
    plt.figure(figsize=(6, 5))

    for name, model in models.items():
        # Certains modèles n'ont pas predict_proba (ex: SVM sans probas)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curves(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str = "Courbes Precision-Recall") -> None:
    """
    Affiche les courbes Precision-Recall pour plusieurs modèles.
    """
    plt.figure(figsize=(6, 5))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

def plot_simple_precision_recall(model, X_test, y_test, model_name="Model"):
    """
    Courbe Precision-Recall simple pour un seul modèle
    """
    # Obtenir les probabilités
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculer la courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    # Tracer la courbe
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
    
    # Ligne de baseline
    baseline = y_test.mean()
    plt.axhline(y=baseline, color='r', linestyle='--', 
                label=f'Baseline = {baseline:.3f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbe Precision-Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.show()




# Selection des variables (feature importance)

def select_features(model: object,X_train: pd.DataFrame,top_n: int = 9):
    """
    Sélectionne les variables les plus importantes selon un modèle de type RandomForest.

    output:
        DataFrame: tableau des variables les plus importantes
    """

    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Le modèle choisi ne possède pas 'feature_importances_'.")

    importances = model.feature_importances_
    feature_names = np.array(X_train.columns)
    sorted_idx = np.argsort(importances)[::-1]

    top_features = pd.DataFrame({
        "Feature": feature_names[sorted_idx][:top_n],
        "Importance": importances[sorted_idx][:top_n],
    })

    print(f"\nTop {top_n} variables les plus importantes pour {model}:")
    print(top_features.to_string(index=False))

    return top_features