# Projet Machine Learning UE A 

Jeux de données : **Spambase** et **Diabetes Health Indicators** 

## Structure
- `src/ml_workflow.py` : pipeline complet (prétraitement, modèles, évaluation)
- `notebooks/` : les notebooks d’expérimentation
- `tests/` : nos tests unitaires
- `data/` : jeux de données
- `requirements.txt` : dépendances

## Objectif
Comparer plusieurs classifieurs sur deux datasets aux propriétés différentes **Spambase** et **Diabetes Health Indicators** 

## Description du fichier python et des notebooks
- Pour le fichier ml_workflow.py:
C'est un fichier qui fournit un pipeline de classification permettant de préparer les données, entraîner plusieurs modèles (RandomForest, KNN, NeuralNetwork, XGboost) et visualiser les résultats.
Fonctions principales: 

Contenu du fichier ml_workflow.py

Le pipeline est entièrement modulaire et comprend :

1. Chargement & Prétraitement
    - Support .csv et .data
    - Détection de la colonne cible
    - Gestion des valeurs manquantes
    - Normalisation des variables

2. Exploration des données
    - Distribution des classes
    - Heatmap des corrélations
    - Analyse en Composantes Principales (PCA)

3. Entraînement des modèles
    - Split train/test
    - Modèles inclus : RandomForest, KNN, MLP, XGBoost
    - Première évaluation du modèle
    - Optimisation via GridSearchCV / RandomizedSearchCV
    - Validation croisée 3 ou 5 folds

4. Évaluation & Visualisations
    - Matrices de confusion
    - Courbes ROC & Precision–Recall
    - Importance des features (RandomForest)
    - Priorisation des métriques selon le dataset :
    - Spambase : F1-score & Recall
    - Diabetes : Recall (réduction des faux négatifs)

- Pour les notebooks:
Les notebooks font l'analyse du spam dataset créé par HP labs et du diabète dataset, l'entrainement et l'évaluation des modèles de classification, la comparaison des modèles utilisés et finalement le choix du modèle le plus pertinent vis-à vis des métriques notamment l'accuracy, le F1 score, le recall et la précision.
Nous n’avons pas inclus XGBoost pour le spam dataset, car les autres modèles (Random Forest, KNN) ont atteint des résultats satisfaisants, rendant l’ajout de XGBoost non nécessaire dans ce contexte. En revanche, XGBoost a été utilisé pour la classification du dataset diabète, où il a montré de bons résultats.


### Organisation de notre pipeline

Le fichier `ml_workflow.py` a été structuré de manière à pouvoir être réutilisé dans d’autres projets
Chaque étape ( loading, preprocessing, entraînement, validation croisée, visualisation) est indépendante, ce qui permet d'utiliser le pipeline sur un nouveau dataset simplement en changeant le chemin du fichier et d’exécuter les mêmes étapes dans les notebooks sans dupliquer du code.


## Validation croisée et choix des modèles


Les meilleurs modèles obtenus sont :

- **Spambase :** RandomForest (meilleurs scores globaux), suivi de KNN  
- **Diabetes :** XGBoost (meilleure stabilité et meilleur Recall)

## Notes sur la PCA

La PCA a été intégrée au pipeline pour permettre l’analyse de datasets à grande dimension.  
Cependant, la réduction de dimension n’apporte pas d’amélioration notable dans le dataset du diabetes : les premières composantes capturent la majorité de la variance, mais les modèles fonctionnent déjà très bien sans réduction....


## Installation de l’environnement virtuel

Avant de lancer le pipeline, il faut créez son propre environnement virtuel :
```python
python3 -m venv .venv
source .venv/bin/activate  # (ou .venv\Scripts\activate sous Windows)
pip install -r requirements.txt

## Lancement rapide

Une fois l’environnement installé, il est possible d’exécuter un pipeline complet en quelques lignes dans un notebook :

from src.ml_workflow import load_data, preprocess_data, split_data, train_models

df = load_data("data/spambase.data")  # (ou load_data("data/diabetes/diabetes.csv") pour le dataset diabète ) 
df_prep, _ = preprocess_data(df, target_column="spam")
X_train, X_test, y_train, y_test = split_data(df_prep, target_column="spam")
models, results = train_models(X_train, X_test, y_train, y_test)


