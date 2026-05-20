# Student Performance Predictor

An end-to-end machine learning project designed to predict student academic performance using a structured, production-style pipeline.

---

## 1. Project Overview

This project implements a modular machine learning pipeline that includes:

- Data ingestion from raw sources
- Data validation and schema enforcement
- Feature engineering
- Feature preprocessing using scalable pipelines
- Train-test data splitting
- Model training and evaluation
- Hyperparameter tuning and model optimization

The project focuses on building a clean, maintainable, and production-oriented ML workflow instead of a notebook-only implementation.

---

## 2. Problem Statement

The objective is to build a machine learning system capable of predicting student performance based on demographic and academic features.

---

## 3. Objective

To estimate student scores and analyze the factors influencing academic outcomes.

---

## 4. Problem Type

- Supervised Learning
- Regression

---

## 5. Dataset

Dataset used:
- Students Performance in Exams Dataset

The dataset contains demographic and academic information related to student exam performance.

---

## 6. Input Features

### Categorical Features
- Gender
- Parental level of education
- Lunch type
- Test preparation course

### Numerical Features
- Reading score
- Writing score
- Average score (engineered feature)
- Score gap (engineered feature)

---

## 7. Target Variable

- Math score

---

## 8. Evaluation Metrics

The following regression metrics are used for evaluation:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

---

## 9. Project Structure

```text
ml_project/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │
│   ├── features/
│   │   ├── engineering.py
│   │   ├── preprocessing.py
│   │
│   ├── models/
│   │   ├── train.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │
│   ├── utils/
│
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│
├── app/
│
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 10. Pipeline Architecture

```text
Raw Data
   ↓
Data Ingestion
   ↓
Data Validation
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Preprocessing Pipeline
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Model Selection
   ↓
Model Persistence
```

---

## 11. Implementation Details

### Phase 1: Problem Definition
- Defined machine learning objective
- Identified target variable and feature groups
- Selected regression as the problem type

---

### Phase 2: Data Ingestion
- Implemented modular ingestion component
- Loaded dataset from CSV
- Standardized column names
- Saved processed dataset for downstream use

---

### Phase 3: Data Validation

Implemented validation checks for:
- Required columns
- Data types
- Missing values
- Duplicate records
- Numerical ranges

Built a fail-fast validation system to ensure data reliability before training.

---

### Phase 4: Data Preprocessing

Implemented reusable preprocessing pipelines using Scikit-learn.

#### Numerical Pipeline
- Missing value imputation using median
- Feature scaling using StandardScaler

#### Categorical Pipeline
- Missing value imputation using most frequent values
- One-hot encoding using OneHotEncoder

#### Combined Using
- ColumnTransformer

#### Output

Serialized preprocessing object:

```text
artifacts/preprocessor.pkl
```

---

### Phase 5: Feature Engineering and Data Splitting

Created engineered features:
- Average score
- Score gap between reading and writing scores

Performed:
- 80/20 train-test split
- Preprocessing fit on training data only
- Transformation applied to test data separately

This prevents data leakage during evaluation.

---

### Phase 6: Model Training and Evaluation

Implemented training and evaluation for:
- Linear Regression
- Random Forest Regressor

Evaluation metrics:
- RMSE
- MAE

Compared multiple models and selected the best-performing baseline model.

---

### Phase 7: Hyperparameter Tuning and Model Selection

Implemented hyperparameter optimization using:
- RandomizedSearchCV

Tuned Random Forest parameters:
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

Applied:
- 5-fold cross-validation

Compared tuned model performance against baseline models and selected the final best-performing model.

Serialized trained model:

```text
artifacts/model.pkl
```

---

## 12. Model Performance

### Baseline Models

| Model | RMSE | MAE |
|---|---|---|
| Linear Regression | 5.366 | 4.227 |
| Random Forest Regressor | 6.259 | 4.954 |

### Tuned Random Forest

| Model | RMSE | MAE |
|---|---|---|
| Tuned Random Forest | 6.107 | 4.678 |

### Final Selected Model

- Linear Regression

The Linear Regression model achieved the best RMSE score on the test dataset.

---

## 13. How to Run

### Step 1: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

---

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3: Execute Training Pipeline

```bash
python -m src.pipeline.training_pipeline
```

---

## 14. Key Concepts Demonstrated

- Modular ML system design
- Object-oriented programming in ML pipelines
- Data validation strategies
- Feature engineering
- Scikit-learn Pipeline and ColumnTransformer
- Train-test splitting and leakage prevention
- Multi-model evaluation
- Hyperparameter tuning using RandomizedSearchCV
- Model persistence using Joblib
- Reproducible ML workflows

---

## 15. Current Status

Development completed up to:
- Hyperparameter tuning
- Final model selection

---

## 16. Future Improvements

Planned enhancements:
- Inference pipeline
- Flask/FastAPI deployment
- REST API endpoints
- Experiment tracking using MLflow
- Docker containerization
- CI/CD integration

---

## 17. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Flask (planned)

---

## 18. Author

Gurnaaz Kaur
