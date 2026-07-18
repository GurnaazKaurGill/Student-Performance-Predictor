# Student Performance Predictor

An end-to-end machine learning project that predicts a student's math score using demographic and academic-preparation information, built as a modular, production-style pipeline with a working inference API.

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
- An inference pipeline for serving predictions on new data
- A FastAPI application exposing the model as a REST API

The project focuses on building a clean, maintainable, and production-oriented ML workflow instead of a notebook-only implementation.

---

## 2. Problem Statement

The objective is to build a machine learning system capable of predicting a student's math score using demographic and academic-preparation information that would realistically be known **before** the student takes the exam.

---

## 3. Objective

To estimate a student's math score from demographic and preparatory factors, and to identify which of these factors relate to academic outcomes.

---

## 4. Problem Type

- Supervised Learning
- Regression

---

## 5. Dataset

Dataset used:
- Students Performance in Exams Dataset (1,000 records)

The dataset contains demographic and academic information related to student exam performance.

---

## 6. Input Features

### Categorical Features
- Gender
- Race/ethnicity
- Parental level of education
- Lunch type
- Test preparation course

### Numerical Features (engineered)
- `completed_prep` — binary flag, whether the student completed the test preparation course
- `standard_lunch` — binary flag, whether the student receives standard (vs. free/reduced) lunch

**Note on excluded features:** `reading_score` and `writing_score` are deliberately **not** used as input features, even though they exist in the raw dataset. These two scores come from the same exam sitting as `math_score`, the target variable, and would not be available in any realistic use case where the goal is predicting performance ahead of the exam. An earlier version of this project used them as features, which produced an artificially low error (RMSE ≈ 5.3) by essentially leaking the answer. This was identified and corrected — see Section 12 for before/after numbers.

---

## 7. Target Variable

- Math score

---

## 8. Evaluation Metrics

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

---

## 9. Project Structure

```text
Student-Performance-Predictor/
│
├── data/
│   ├── raw/
│   │   └── students.csv
│   ├── processed/
│       └── cleaned.csv
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
│   │   ├── predict_pipeline.py
│   │
│   ├── utils/
│       ├── logger.py
│
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│
├── app/
│   ├── app.py
│
├── logs/
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
Feature Engineering (leakage-free)
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
   ↓
Inference Pipeline  →  FastAPI REST API
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

### Phase 4: Feature Engineering (corrected)

Created engineered features using only columns that would be known in advance:
- `completed_prep` — derived from test preparation course completion
- `standard_lunch` — derived from lunch type

`reading_score` and `writing_score` are excluded from feature engineering entirely, since they are outcomes of the same exam sitting as the target variable.

---

### Phase 5: Data Preprocessing

Implemented reusable preprocessing pipelines using Scikit-learn.

#### Numerical Pipeline
- Missing value imputation using median
- Feature scaling using StandardScaler

#### Categorical Pipeline
- Missing value imputation using most frequent values
- One-hot encoding using OneHotEncoder

#### Combined Using
- ColumnTransformer, with an explicit allow-list of permitted feature columns (rather than "all columns except the target"), so no future column can silently leak into the model

#### Output

Serialized preprocessing object:

```text
artifacts/preprocessor.pkl
```

---

### Phase 6: Train-Test Splitting

- 80/20 train-test split
- Preprocessing fit on training data only
- Transformation applied to test data separately

This prevents data leakage during evaluation.

---

### Phase 7: Model Training and Evaluation

Implemented training and evaluation for:
- Linear Regression
- Random Forest Regressor

Evaluation metrics:
- RMSE
- MAE

Compared multiple models and selected the best-performing baseline model.

---

### Phase 8: Hyperparameter Tuning and Model Selection

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

### Phase 9: Inference Pipeline

Implemented `PredictPipeline`, which:
- Loads the saved model and preprocessor from `artifacts/`
- Accepts a single new student's details as input
- Runs the same feature engineering and preprocessing steps used during training
- Returns a predicted math score

---

### Phase 10: REST API

Implemented a FastAPI application (`app/app.py`) exposing:
- `GET /` — health check
- `POST /predict` — accepts a student's demographic and preparation details as JSON, returns a predicted math score

Input validation is handled automatically through a Pydantic model (`StudentInput`), which rejects malformed or incomplete requests with a clear error before they ever reach the model. FastAPI also auto-generates an interactive documentation page at `/docs`, where the endpoint can be tested directly from the browser.

---

## 12. Model Performance

### Before correcting data leakage

Earlier versions of this project used `reading_score` and `writing_score` as input features. Since these come from the same exam sitting as the target (`math_score`), this produced artificially low error:

| Model | RMSE | MAE |
|---|---|---|
| Linear Regression | 5.366 | 4.227 |
| Random Forest Regressor | 6.259 | 4.954 |

### After correcting data leakage (current, honest results)

With `reading_score` and `writing_score` removed, using only demographic and preparatory features:

| Model | RMSE | MAE |
|---|---|---|
| Linear Regression | 14.160 | 11.270 |
| Random Forest Regressor (default) | 15.699 | 12.345 |
| Random Forest Regressor (tuned) | 14.580 | 11.627 |

### Final Selected Model

- **Linear Regression**

The tuned Random Forest did not outperform Linear Regression on the held-out test set, so Linear Regression was automatically selected as the final model. The higher RMSE compared to the earlier (leaked) version is expected and correct — it reflects the model being evaluated on a genuinely harder, more realistic problem.

---

## 13. How to Run

### Step 1: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Training Pipeline

```bash
python -m src.pipeline.training_pipeline
```

This runs ingestion → validation → feature engineering → preprocessing → training → tuning → model selection, and saves `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

### Step 4: Run the API Server

```bash
uvicorn app.app:app --reload
```

Then open `http://127.0.0.1:8000/docs` in a browser to test the `/predict` endpoint interactively, or send a request directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "race/ethnicity": "group B",
    "parental_level_of_education": "bachelor'\''s degree",
    "lunch": "standard",
    "test_preparation_course": "completed"
  }'
```

---

## 14. Key Concepts Demonstrated

- Modular ML system design
- Object-oriented programming in ML pipelines
- Data validation strategies
- Data leakage identification and correction
- Feature engineering with explicit, allow-list based feature selection
- Scikit-learn Pipeline and ColumnTransformer
- Train-test splitting and leakage prevention
- Multi-model evaluation
- Hyperparameter tuning using RandomizedSearchCV
- Model persistence using Joblib
- Inference pipeline design
- REST API development with FastAPI and Pydantic
- Structured logging
- Reproducible ML workflows

---

## 15. Current Status

Development completed up to:
- Hyperparameter tuning and final model selection
- Inference pipeline
- FastAPI REST API with automatic request validation

---

## 16. Future Improvements

Planned enhancements:
- Automated unit tests
- Configuration file for paths and hyperparameters
- Experiment tracking using MLflow
- Docker containerization
- CI/CD integration
- Expansion to a larger, more diverse dataset

---

## 17. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Uvicorn

---

## 18. Author

Gurnaaz Kaur
