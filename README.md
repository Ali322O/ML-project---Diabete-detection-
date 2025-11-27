# Machine Learning Project – Classification Benchmark

Datasets used: **Spambase** and **Diabetes Health Indicators**

## Project Structure
- `src/ml_workflow.py` — main classification pipeline (preprocessing, modeling, evaluation)
- `notebooks/` — exploratory analysis and experiments
- `tests/` — unit tests for pipeline components
- `data/` — datasets
- `requirements.txt` — environment dependencies

## Objective
Benchmark multiple classifiers on two structurally different datasets and analyze which models perform best depending on the data characteristics.

## Pipeline Overview (`ml_workflow.py`)

The workflow is modular and reusable, designed to handle end-to-end classification experiments:

### 1. Data Loading & Preprocessing
- CSV / DATA file support  
- Automatic target-column handling  
- Missing-value treatment  
- Feature scaling & normalization  

### 2. Exploratory Analysis
- Class distribution  
- Correlation heatmaps  
- PCA for dimensionality inspection  

### 3. Model Training
- Standard train/test split  
- Models included: **RandomForest, KNN, MLP, XGBoost**  
- Baseline evaluation  
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- 5-fold cross-validation  

### 4. Evaluation & Metrics
- Confusion matrices  
- ROC & PR curves  
- Feature importance (RandomForest & XGBoost)  
- Dataset-specific metric prioritization:
  - **Spambase:** F1-score / Recall  
  - **Diabetes:** Recall 

## Notebooks
The notebooks cover:
- Exploratory analysis of both datasets  
- Model training and comparison  
- Metric-driven model selection (Accuracy, F1, Recall, Precision)

XGBoost was not included for Spambase (other models already performed strongly), but was used for Diabetes where it showed superior stability and recall.

## Pipeline Design
The `ml_workflow.py` file is structured as a reusable pipeline:
- Each step (loading, preprocessing, training, validation, visualization) is independent  
- The same pipeline can be applied to any new dataset by simply changing the input file path  
- Notebooks rely on the pipeline without duplicating code  

## Model Selection Results
- **Spambase:** RandomForest performed best overall, followed by KNN  
- **Diabetes:** XGBoost achieved the strongest performance, especially on Recall  

## Notes on PCA
PCA was included for analysis.  
Although high variance is captured by the first components, dimensionality reduction did not significantly improve model performance for the Diabetes dataset.

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
