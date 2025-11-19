"""
Ce fichier rassemble l'ensemble du pipeline utilisé pour nos deux notebooks de classification.
L’idée est d’avoir un seul fichier .py qui centralise :
- le chargement des données
- la préparation (nettoyage, éventuel prétraitement simple)
- le split train/test + normalisation + PCA optionnelle
- l’entraînement des modèles (Spambase + Diabetes)
- les évaluations (metrics, matrices de confusion, courbes)
- la sélection de variables importantes
- ainsi que les fonctions spécifiques au dataset diabète
"""

from __future__ import annotations
import warnings
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_validate,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    make_scorer,
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


# ETAPE 1 : On charge les données (Spambase + Diabetes)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge les données en tenant compte :
    - du format CSV (dataset diabète)
    - du format .data + .names (dataset Spambase)

    Pour Spambase, les colonnes sont récupérées automatiquement
    depuis le fichier .names associé.
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} est introuvable.")

    # Cas général CSV ( diabetes )
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        print(f"Dataset CSV chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Cas Spambase : .data + .names
    elif filepath.endswith(".data"):
        names_path = filepath.replace(".data", ".names")
        if not os.path.exists(names_path):
            raise FileNotFoundError(
                f"Fichier .names correspondant introuvable : {names_path}"
            )

        with open(names_path, "r") as f:
            lines = f.readlines()

        col_names: List[str] = []
        for line in lines:
            if ":" in line and not line.startswith("|"):
                col_names.append(line.split(":")[0].strip())

        # Dernière colonne = cible spam
        col_names.append("spam")
        df = pd.read_csv(filepath, header=None, names=col_names)
        print(f"Dataset Spambase chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    else:
        raise ValueError("Format de fichier est non reconnu")

    print("Aperçu des données :")
    print(df.head())
    return df



# ETAPE 2 : Prétraitement léger commun aux deux notebooks

def preprocess_data(df: pd.DataFrame,target_column: str = None,apply_pca: bool = False,pca_variance: float = 0.95):
    """
    Prétraitement "léger" partagé par les deux notebooks :

    - On isole la colonne cible si elle existe
    - gère simplement les valeurs manquantes (avec la moyenne) meme si nos datasets n'en contiennent pas
    - On ne fait pas de normalisation (scaling) ici car elle sera faite après le split train/test
    - PCA optionnelle (appliquée sur les données normalisées juste après le split)
    """

    df_clean = df.copy()

    # Séparation de la cible
    y = None
    if target_column is not None and target_column in df_clean.columns:
        y = df_clean[target_column]
        X = df_clean.drop(columns=[target_column])
    else:
        X = df_clean

    # Valeurs manquantes comblées (par sécurité) , même si  on vérifié que nos datasets n'en contiennent pas
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())

    X_processed = X.copy()
    if y is not None:
        X_processed[target_column] = y.values

    return X_processed, None


# ETAPE 3 : Split train/test + normalisation + PCA optionnelle

def split_data(df: pd.DataFrame,target_column: str | None = None,test_size: float = 0.2,random_state: int = 42,apply_pca: bool = False,
            pca_variance: float = 0.95):
    """
    Split utilisé dans les deux notebooks
    - détection/choix de la colonne cible
    - séparation X / y
    - train_test_split stratifié
    - normalisation (StandardScaler) APRES le split
    - PCA optionnelle (appliquée sur les données déjà normalisées)
    """

    df_copy = df.copy()

    # Détection automatique de la cible si besoin
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

    # Split stratifié pour conserver la proportion de classes
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Split effectué : {X_train.shape[0]} train / {X_test.shape[0]} test")

    # Normalisation sur les deux jeux (fit sur train uniquement)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # PCA optionnelle
    if apply_pca:
        pca_model = PCA(n_components=pca_variance)
        X_train_pca = pca_model.fit_transform(X_train_scaled)
        X_test_pca = pca_model.transform(X_test_scaled)

        X_train_pca = pd.DataFrame(
            X_train_pca, columns=[f"PC{i+1}" for i in range(X_train_pca.shape[1])]
        )
        X_test_pca = pd.DataFrame(
            X_test_pca, columns=[f"PC{i+1}" for i in range(X_test_pca.shape[1])]
        )

        return X_train_pca, X_test_pca, y_train, y_test

    # Cas sans PCA
    return X_train_scaled, X_test_scaled, y_train, y_test


# ETAPE 4 : Entraînement des modèles (Spambase + Diabetes)
def train_models(X_train: pd.DataFrame,X_test: pd.DataFrame,y_train: pd.Series,y_test: pd.Series):
    """
    On entraîne trois modèles utilisés sur Spambase :
    - Random Forest
    - KNN
    - Réseau de neurones (MLP)
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
        print(
            f"{name:<12} → Accuracy: {scores['accuracy']:.3f} | "
            f"F1: {scores['f1_score']:.3f}"
        )

    return trained_models, results


# ETAPE 5 : Évaluation des modèles (matrices de confusion)

def evaluate_models(models: Dict[str, object],X_test: pd.DataFrame,y_test: pd.Series):
    """
    Pour chaque modèle, on affiche une matrice de confusion sous forme de heatmap et on
    on renvoie les matrices dans un dictionnaire
    """

    evaluations: Dict[str, Dict[str, object]] = {}

    for name, model in models.items():
        print(f"\n{name}")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Classe 0", "Classe 1"],
            yticklabels=["Classe 0", "Classe 1"],
        )
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.title(f"Matrice de confusion - {name}")
        plt.tight_layout()
        plt.show()

        evaluations[name] = {"confusion_matrix": cm}

    return evaluations


# ETAPE 6 : Cross-validation 

def cross_validate_model(model,X,y,cv: int = 5,scoring: list | tuple = ("accuracy", "precision", "recall", "f1")):
    """
    Validation croisée standard avec sklearn.cross_validate
    et on affiche pour chaque métrique la moyenne et l'écart-type
    et renvoie les valeurs dans un dictionnaire
    """

    print(f"\nValidation croisée pour : {model.__class__.__name__}")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    results = {}
    for metric in scoring:
        mean = scores[f"test_{metric}"].mean()
        std = scores[f"test_{metric}"].std()
        results[metric] = (mean, std)
        print(f"→ {metric.capitalize()} : {mean:.4f} ± {std:.4f}")

    return results


# ETAPE 7 : Recherche d'hyperparamètres (GridSearchCV )

def grid_search_model(model,param_grid: dict,X_train: pd.DataFrame,y_train: pd.Series,scoring: str = "f1",cv: int = 5,n_jobs: int = -1):
    """
    Effectue un GridSearchCV générique (RandomForest, KNN, etc...)
    Retourne :
    - best_model : modèle entraîné avec les meilleurs paramètres
    - best_params : dict des meilleurs paramètres
    - best_score : meilleur score CV
    """

    grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=cv,n_jobs=n_jobs,verbose=1)
    grid.fit(X_train, y_train)

    print("\nRésultats du GridSearchCV :")
    print("Meilleurs paramètres :", grid.best_params_)
    print(f"Meilleur score ({scoring}) :", grid.best_score_)

    best_model = grid.best_estimator_
    return best_model, grid.best_params_, grid.best_score_


# ETAPE 8 : Sélection des variables importantes (Feature importance)

def select_features(model: object,X_train: pd.DataFrame,top_n: int = 10):
    """
    Récupère les top_n variables les plus importantes pour un modèle
    qui expose l'attribut feature_importances_ (RandomForest, XGBoost, etc...)
    """

    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Le modèle choisi ne possède pas 'feature_importances_'.")

    importances = model.feature_importances_
    feature_names = np.array(X_train.columns)
    sorted_idx = np.argsort(importances)[::-1]

    top_features = pd.DataFrame(
        {
            "Feature": feature_names[sorted_idx][:top_n],
            "Importance": importances[sorted_idx][:top_n],
        }
    )
    print(f"\nTop {top_n} variables les plus importantes :")
    print(top_features.to_string(index=False))

    return top_features


# ETAPE 9 : Fonctions de visualisation communes aux deux notebooks

def plot_class_distribution(y: pd.Series,title: str = "Répartition de la classe cible") :
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'exemples")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame,target_column: str | None = None,max_features: int = 30,title: str = "Matrice de corrélation") :
    data = df.copy()
    if target_column is not None and target_column in data.columns:
        data = data.drop(columns=[target_column])

    corr = data.corr(numeric_only=True)

    if corr.shape[0] > max_features:
        mean_corr = corr.abs().mean().sort_values(ascending=False)
        top_cols = mean_corr.index[:max_features]
        corr = corr.loc[top_cols, top_cols]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pca_2d(X: pd.DataFrame,y: pd.Series) :
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.5)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var.)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var.)")
    plt.title("Projection PCA (2D)")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(models: Dict[str, object],X_test: pd.DataFrame,y_test: pd.Series,class_names: List[str] | None = None) :
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


def plot_roc_curves(models: Dict[str, object],X_test: pd.DataFrame,y_test: pd.Series,title: str = "Courbes ROC") :
    plt.figure(figsize=(6, 5))
    for name, model in models.items():
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


def plot_precision_recall_curves(models: Dict[str, object],X_test: pd.DataFrame,y_test: pd.Series,title: str = "Courbes Precision-Recall"):
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


def plot_feature_importances_bar(top_features: pd.DataFrame,title: str = "Top features importantes (RandomForest)"):
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


# ETAPE 10 : Fonctions spécifiques au pipeline diabète :
## Dans le dataset Spam , on n'utilise pas le training de XBoost ni de VotingClassifier

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

def analyze_diabetes_distribution(filepath: str):
    """
    Lecture du CSV du dataset diabète + quelques visualisations de base.
    """

    filepath = Path(filepath)
    df = pd.read_csv(filepath)
    print(f"Dataset CSV chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(df.head())
    print(df.info())

    cols = [
        "Diabetes_binary",
        "HighBP",
        "HighChol",
        "CholCheck",
        "BMI",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income",
    ]

    plt.figure(figsize=(20, 20))
    for i, col in enumerate(cols, 1):
        plt.subplot(6, 4, i)
        plt.hist(df[col], bins=20, edgecolor="black")
        plt.title(col, fontsize=10)
        plt.xlabel("")
        plt.ylabel("Fréquence")
    plt.subplots_adjust(hspace=0.8, wspace=0.6)
    plt.show()

    first_col = df.iloc[:, 0]
    nb_diabetiques = (1 / len(df)) * np.sum(first_col == 1)
    print("Le pourcentage de diabétique est de", nb_diabetiques * 100, "%")

    return df


def plot_correlation_matrix(df: pd.DataFrame):
    corr_matrix = df.corr()
    top22_features = (
        corr_matrix["Diabetes_binary"]
        .abs()
        .sort_values(ascending=False)
        .head(23)
        .index
    )

    plt.figure(figsize=(15, 15))
    sns.heatmap(
        corr_matrix.loc[top22_features, top22_features],
        annot=True,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
    )
    plt.title("Correlation matrix avec heatmap", fontsize=14)
    plt.show()


def plot_top10_correlated(df: pd.DataFrame) :
    corr_matrix = df.corr()
    top10_features = (
        corr_matrix["Diabetes_binary"]
        .abs()
        .sort_values(ascending=False)
        .head(11)
        .index
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix.loc[top10_features, top10_features],
        annot=True,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
    )
    plt.title("Top 10 features les plus corrélées à la variable cible", fontsize=14)
    plt.show()


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
