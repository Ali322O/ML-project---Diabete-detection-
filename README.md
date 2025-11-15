# Projet Machine Learning UE A 

Jeux de données : **Spambase** et **Diabetes Health Indicators** 

## Structure
- `src/ml_workflow.py` : pipeline complet (prétraitement, modèles, évaluation)
- `notebooks/` : les notebooks d’expérimentation
- `tests/` : nos tests unitaires
- `data/` : jeux de données
- `requirements.txt` : dépendances

## Auteurs
- Mohammed Ali Al Marjani (mohammed-ali.al-marjani@imt.atlantique.net)
- Denieul Adam ( adam.denieul@imt.atlantique.net)
- Kessentini yosr ( yosr.kessentini@imt.atlantique.net)

## Objectif
Comparer plusieurs classifieurs sur deux datasets aux propriétés différentes **Spambase** et **Diabetes Health Indicators** 

## Description du fichier python et des notebooks
- Pour le fichier ml_workflow.py:
C'est un fichier qui fournit un pipeline de classification permettant de préparer les données, entraîner plusieurs modèles (RandomForest, KNN, NeuralNetwork, XGboost) et visualiser les résultats.
Fonctions principales: 

1- Chargement des données: Support .csv et .data (Spambase), détection automatique de la colonne cible

2- Prétraitement: Gestion simple des valeurs manquantes, normalisation des features numériques

3- Split Train/Test: CrossValidation

4- Entraînement des modèles: optimisation des hyper-paramètres, renvoie des scores Accuracy et F1

5- Évaluation: Matrices de confusion, Rapport de classification

6- Sélection des features importantes pour RandomForest

7- Visualisations: Distribution des classes, Heatmap des corrélations, Courbes ROC et Precision-Recall

- Pour les notebooks:
Les notebooks font l'analyse du spam dataset créé par HP labs et du diabète dataset, l'entrainement et l'évaluation des modèles de classification, la comparaison des modèles utilisés et finalement le choix du modèle le plus pertinent vis-à vis des métriques notamment l'accuracy, le F1 score, le recall et la précision.
Nous n’avons pas inclus XGBoost pour le spam dataset, car les autres modèles (Random Forest, KNN) ont atteint des résultats satisfaisants, rendant l’ajout de XGBoost non nécessaire dans ce contexte. En revanche, XGBoost a été utilisé pour la classification du dataset diabète, où il a montré de bons résultats.

### Installation de l’environnement virtuel

Avant de lancer le pipeline, créez votre propre environnement virtuel :

```bash
python3 -m venv .venv
source .venv/bin/activate  # (ou .venv\Scripts\activate sous Windows)
pip install -r requirements.txt



Le readme est a completer au fur et a mesure du projet 