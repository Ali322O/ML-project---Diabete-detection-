import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from src.ml_workflow import (
    load_data,
    preprocess_data,
    split_data,
    train_models,
    cross_validate_model,
    select_features,
)


# Test de la fonction load_data

def test_load_data_csv(tmp_path):
    csv_path = tmp_path / "test.csv"

    df_expected = pd.DataFrame({
        "A": [1, 2],
        "B": [3, 4],
        "Diabetes_binary": [0, 1]
    })

    df_expected.to_csv(csv_path, index=False)

    df_loaded = load_data(str(csv_path))

    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.shape == (2, 3)
    assert list(df_loaded.columns) == ["A", "B", "Diabetes_binary"]
    

# Test de la fonction preprocess_data

def test_preprocess_data_basic():
    df = pd.DataFrame({
        "f1": [1, 2, 3],
        "f2": [4, 5, 6],
        "spam": [0, 1, 0]
    })

    df_processed, _ = preprocess_data(df, target_column="spam")

    # Rien n'est supprimé (pas de scaling ici)
    
    assert df_processed.shape == df.shape
    assert "spam" in df_processed.columns
    assert df_processed.isnull().sum().sum() == 0


# Test de la fonction split_data
def test_split_data_scaling_and_no_pca():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "f2": [5, 6, 7, 8],
        "spam": [0, 1, 0, 1]
    })

    X_train, X_test, y_train, y_test = split_data(df, target_column="spam")

    # shapes
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1

    # les données sont bien scalées ( moyenne proche de 0 )
    assert abs(X_train.mean().mean()) < 1e-6


def test_split_data_with_pca():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "f2": [5, 6, 7, 8],
        "spam": [0, 1, 0, 1]
    })

    X_train, X_test, y_train, y_test = split_data(
        df, target_column="spam", apply_pca=True, pca_variance=0.95
    )

    # PCA réduit à 1 composante (2 features très corrélés)
    assert X_train.shape[1] <= 2
    assert X_test.shape[1] == X_train.shape[1]


def test_train_models_runs_without_error():
    df = pd.DataFrame({
        "f1": np.random.randn(20),
        "f2": np.random.randn(20),
        "spam": np.random.randint(0, 2, size=20)
    })

    X_train, X_test, y_train, y_test = split_data(df, target_column="spam")

    models, results = train_models(X_train, X_test, y_train, y_test)

    # les trois modèles sont bien entraînés
    assert "RandomForest" in models
    assert "KNN" in models
    assert "NeuralNet" in models

    # présence des métriques
    assert "accuracy" in results["RandomForest"]
    assert "f1_score" in results["RandomForest"]


# Test de la fonction cross_validate_model
def test_cross_validation():
    df = pd.DataFrame({
        "f1": np.random.randn(30),
        "f2": np.random.randn(30),
        "spam": np.random.randint(0, 2, size=30)
    })

    X = df[["f1", "f2"]]
    y = df["spam"]

    model = RandomForestClassifier()

    scores = cross_validate_model(model, X, y, cv=3)

    assert "accuracy" in scores
    assert isinstance(scores["accuracy"], tuple)
    assert len(scores["accuracy"]) == 2  # (mean et std)


# Test de la fonction select_features

def test_select_features_top_n():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "f2": [4, 3, 2, 1],
        "spam": [0, 1, 0, 1]
    })

    X = df[["f1", "f2"]]
    y = df["spam"]

    model = RandomForestClassifier().fit(X, y)

    top = select_features(model, X, top_n=2)

    assert top.shape == (2, 2)
    assert "Feature" in top.columns
    assert "Importance" in top.columns


