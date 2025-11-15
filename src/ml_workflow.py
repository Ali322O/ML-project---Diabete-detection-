"""


Ce fichier contient toutes les fonctions du pipeline :
- Chargement des données
- Nettoyage / gestion des valeurs manquantes
- Normalisation
- Split train/test
- Entraînement de plusieurs modèles
- Évaluation des performances
- Sélection de variables importantes
- Fonctions de visualisation pour les notebooks
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Liste standard des colonnes cibles possibles dans nos datasets
TARGET_CANDIDATES = ["spam", "Diabetes_binary", "Outcome", "class"]






# On Charge d'abord les données

def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge le dataset en fonction de son type (CSV ou .data) et retourne un DataFrame pandas.

    Gère deux cas :
    - Dataset de type CSV (Diabetes)
    - Dataset Spambase (.data sans header, avec .names séparé)

    Args:
        filepath (str): chemin vers le fichier de données

    Returns:
        pd.DataFrame: dataset chargé
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} est introuvable.")

    # Cas du fichier CSV (Diabetes)
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        print(f"Dataset CSV chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Cas du fichier .data (Spambase)
    elif filepath.endswith(".data"):
        names_path = filepath.replace(".data", ".names")

        if not os.path.exists(names_path):
            raise FileNotFoundError(f"Fichier .names correspondant introuvable : {names_path}")

        with open(names_path, "r") as f:
            lines = f.readlines()

        col_names: List[str] = []
        for line in lines:
            if ":" in line and not line.startswith("|"):
                col_names.append(line.split(":")[0].strip())

        # La dernière colonne est la cible "spam"
        col_names.append("spam")
        
        df = pd.read_csv(filepath, header=None, names=col_names)
        print(f"Dataset Spambase chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    else:
        raise ValueError("Format de fichier non reconnu (utiliser .csv ou .data).")

    print("Aperçu des données :")
    print(df.head())
    return df


# Preprocessing des données

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite les données :
    - Gère les valeurs manquantes (moyenne sur les colonnes numériques)
    - Normalise toutes les colonnes numériques (hors variable cible)
    - Ne fait pas d'encodage catégoriel (nos datasets sont déjà numériques)

    Args:
        df (pd.DataFrame): dataframe brut

    Returns:
        pd.DataFrame: dataframe normalisé, avec la colonne cible intacte
    """

    df_clean = df.copy()

    # Gestion des valeurs manquantes
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        print(f"{missing_count} valeurs manquantes détectées → imputation par la moyenne.")
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    else:
        print("Aucune valeur manquante détectée.")

    # Détection de la colonne cible
    target_col = None
    for possible_target in TARGET_CANDIDATES:
        if possible_target in df_clean.columns:
            target_col = possible_target
            break

    if target_col is None:
        raise ValueError("Impossible d’identifier la colonne cible dans le dataset.")

    print(f"Colonne cible détectée : '{target_col}'")

    # Normalisation des variables numériques (sauf la cible)
    features = df_clean.drop(columns=[target_col])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled[target_col] = df_clean[target_col].values

    print(f"Normalisation terminée : {df_scaled.shape[1]} variables (features + cible).")
    return df_scaled



# Separation train / test

def split_data(
    df: pd.DataFrame,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sépare le dataset en ensembles d'entraînement et de test.

    Args:
        df (pd.DataFrame): dataframe prétraité (sortie de preprocess_data)
        target_column (str, optional): nom de la cible. Si None, détection automatique.
        test_size (float): proportion du jeu de test (ex: 0.2 = 20%)
        random_state (int): graine pour la reproductibilité

    Returns:
        X_train, X_test, y_train, y_test
    """

    df_copy = df.copy()

    # Identifier la colonne cible si non précisée
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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Split effectué : {X_train.shape[0]} train / {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test


# Entrainement des modèles

def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]]]:
    """
    Entraîne plusieurs modèles de classification et compare leurs performances.

    Modèles utilisés (simple mais stable) :
      - Random Forest
      - KNN
      - Réseau de neurones "simple" (MLP)

    Args:
        X_train, X_test, y_train, y_test : ensembles d'entraînement et de test

    Returns:
        trained_models (dict): {nom_modèle: modèle_entraîné}
        results (dict): {nom_modèle: {"accuracy": float, "f1_score": float}}
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
        ),
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

    print("\nRésumé des performances :")
    for name, scores in results.items():
        print(f"{name:<12} → Accuracy: {scores['accuracy']:.3f} | F1: {scores['f1_score']:.3f}")

    return trained_models, results


# Évaluation des modèles (rapport + matrices)

def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, object]]:
    """
    Évalue plusieurs modèles sur un jeu de test :
    - Matrice de confusion
    - Rapport de classification (precision, recall, f1)
    - Résumé des scores globaux

    Args:
        models (dict): {nom: modèle_entraîné}
        X_test (DataFrame): features du test set
        y_test (Series): labels du test set

    Returns:
        evaluations (dict): {nom: {"classification_report": dict, "confusion_matrix": np.array}}
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



# Selection des variables (feature importance)

def select_features(
    model: object,
    X_train: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Sélectionne les variables les plus importantes selon un modèle de type RandomForest.

    Args:
        model: modèle entraîné (doit avoir 'feature_importances_')
        X_train (DataFrame): données d'entraînement
        top_n (int): nombre de variables les plus importantes à retourner

    Returns:
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

    print(f"\nTop {top_n} variables les plus importantes :")
    print(top_features.to_string(index=False))

    return top_features



# Pipeline complet 

def run_full_pipeline(filepath: str, target_column: str | None = None) -> None:
    """
    Exécute l'ensemble du pipeline ML sur un dataset :
    - Chargement des données
    - Prétraitement
    - Split train/test
    - Entraînement de plusieurs modèles
    - Évaluation
    - Sélection des features importantes (RandomForest)

    Args:
        filepath (str): chemin vers le dataset
        target_column (str, optional): nom de la cible (sinon détection automatique)

    Returns:
        None (fonction surtout utilisée pour un run rapide + affichage console)
    """


    df = load_data(filepath)
    df_clean = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean, target_column=target_column)
    models, results = train_models(X_train, X_test, y_train, y_test)
    evaluations = evaluate_models(models, X_test, y_test)

    if "RandomForest" in models:
        print("\nAnalyse des variables importantes avec RandomForest :")
        select_features(models["RandomForest"], X_train, top_n=10)
    else:
        print("Le modèle 'RandomForest' n'a pas été entraîné : pas d'analyse de features.")

    print("\n Pipeline complet exécuté.\n")



# Fonctions de visualisation pour les notebooks

def plot_class_distribution(y: pd.Series, title: str = "Répartition de la classe cible") -> None:
    """
    Affiche la répartition des classes (0/1) de la cible.
    """
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'exemples")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_column: str | None = None,
    max_features: int = 30,
    title: str = "Matrice de corrélation",
) -> None:
    """
    Affiche une heatmap de corrélation des variables numériques.

    Args:
        df: DataFrame d'origine (non normalisé ou normalisé)
        target_column: nom de la cible (optionnel pour l'exclure des corrélations)
        max_features: nombre max de features à afficher (si beaucoup de colonnes)
    """
    data = df.copy()
    if target_column is not None and target_column in data.columns:
        data = data.drop(columns=[target_column])

    corr = data.corr(numeric_only=True)

    # Si beaucoup de variables, on peut limiter aux plus corrélées en moyenne
    if corr.shape[0] > max_features:
        mean_corr = corr.abs().mean().sort_values(ascending=False)
        top_cols = mean_corr.index[:max_features]
        corr = corr.loc[top_cols, top_cols]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pca_2d(
    X: pd.DataFrame,
    y: pd.Series,
    title: str = "Projection PCA (2D)",
) -> None:
    """
    Applique une PCA 2D et affiche un scatter plot coloré par la classe.

    ATTENTION : on suppose ici que X est déjà numérique.
    """
    from sklearn.decomposition import PCA

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


def plot_confusion_matrices(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: List[str] | None = None,
) -> None:
    """
    Affiche une matrice de confusion pour chaque modèle.
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


def plot_roc_curves(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str = "Courbes ROC",
) -> None:
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
    title: str = "Courbes Precision-Recall",
) -> None:
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


def plot_feature_importances_bar(
    top_features: pd.DataFrame,
    title: str = "Top features importantes (RandomForest)",
) -> None:
    """
    Affiche un barplot des features importantes (sortie de select_features).
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
