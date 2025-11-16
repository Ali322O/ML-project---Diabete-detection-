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
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_validate, StratifiedKFold


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

    # Si c’est un CSV, on le lit directement
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        print(f"Dataset CSV chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Pour Spambase : fichier .data + .names pour récupérer les noms de colonnes
    elif filepath.endswith(".data"):
        names_path = filepath.replace(".data", ".names")

        # On s’assure ici que le fichier .names existe
        if not os.path.exists(names_path):
            raise FileNotFoundError(f"Fichier .names correspondant introuvable : {names_path}")

        # On parcourt les lignes et on retient les colonnes définies dans .names
        with open(names_path, "r") as f:
            lines = f.readlines()

        col_names: List[str] = []
        for line in lines:
            if ":" in line and not line.startswith("|"):
                col_names.append(line.split(":")[0].strip())

        # La dernière colonne correspond à la cible spam
        col_names.append("spam")
        
        df = pd.read_csv(filepath, header=None, names=col_names)
        print(f"Dataset Spambase chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    else:
        # Si jamais le format n’est pas connu, on arrête
        raise ValueError("Format de fichier non reconnu (utiliser .csv ou .data).")

    # Un petit aperçu pour vérifier que tout est normal
    print("Aperçu des données :")
    print(df.head())
    return df


#### Prétraitement des données

def preprocess_data(df: pd.DataFrame, target_column: str = None, apply_pca: bool = False, pca_variance: float = 0.95):
    
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

    # Normalisation 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA optionnelle
    pca_model = None
    if apply_pca:

        pca_model = PCA(n_components=pca_variance)
        X_scaled = pca_model.fit_transform(X_scaled)

        X_scaled = pd.DataFrame(
            X_scaled,
            columns=[f"PC{i+1}" for i in range(X_scaled.shape[1])]
        )
    else:
        # Si pas de PCA, on garde les noms des colonnes originales
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    
    if y is not None:
        X_scaled[target_column] = y.values

    return X_scaled, pca_model



####  Split train / test

def split_data(df: pd.DataFrame,target_column: str | None = None,test_size: float = 0.2,random_state: int = 42) :
    
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
    return X_train, X_test, y_train, y_test



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
            n_estimators=300,
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
        print(f"\n🔍 Évaluation du modèle : {name}")
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        print("Matrice de confusion :")
        print(cm)
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))

        evaluations[name] = {
            "classification_report": report,
            "confusion_matrix": cm,
        }

    return evaluations



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
def plot_pca_2d(X: pd.DataFrame,y: pd.Series,itle: str = "Projection PCA (2D)") :
    
    """
    Représentation simple en deux dimensions après PCA pour visualiser la séparation des classes.
    """

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.5)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var.)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var.)")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Classe")
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
