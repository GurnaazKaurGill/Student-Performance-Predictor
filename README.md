# Student Performance Predictor

An end-to-end machine learning project designed to predict student academic performance using a structured, production-style pipeline.

---

## 1. Project Overview

This project implements a modular machine learning pipeline that includes:

* Data ingestion from raw sources
* Data validation and schema enforcement
* Feature preprocessing using scalable pipelines
* Preparation for model training and evaluation

The focus is on building a clean, maintainable system rather than a notebook-based prototype.

---

## 2. Problem Statement

The objective is to build a machine learning system capable of predicting student performance based on demographic and academic features.

---

## 3. Objective

To estimate student scores and analyze the factors that influence academic outcomes.

---

## 4. Problem Type

* Supervised Learning
* Regression

---

## 5. Input Features

### Categorical Features

* Gender
* Parental level of education
* Lunch type
* Test preparation course

### Numerical Features

* Reading score
* Writing score

---

## 6. Target Variable

* Math score

---

## 7. Evaluation Metrics (Planned)

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)

---

## 8. Project Structure

```
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
│   │   ├── preprocessing.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│
│   ├── utils/
│
├── artifacts/
│
├── app/
│
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 9. Implementation Details

### Phase 1: Problem Definition

* Defined the machine learning objective
* Identified input features and target variable
* Selected regression as the problem type

---

### Phase 2: Data Ingestion

* Implemented a modular ingestion component
* Loaded dataset from CSV
* Standardized column names
* Saved processed data for downstream use

---

### Phase 3: Data Validation

* Enforced schema constraints
* Validated:

  * Required columns
  * Data types (with flexible handling)
  * Missing values
  * Duplicate records
  * Value ranges (0–100 for scores)
* Designed fail-fast validation logic

---

### Phase 4: Data Preprocessing

Implemented a reusable preprocessing pipeline using Scikit-learn.

#### Numerical Pipeline

* Missing value imputation using median
* Feature scaling using StandardScaler

#### Categorical Pipeline

* Missing value imputation using most frequent value
* One-hot encoding with unknown category handling

#### Combined Using

* ColumnTransformer

#### Outputs

* Transformed feature matrix
* Serialized preprocessing object stored at:

  ```
  artifacts/preprocessor.pkl
  ```

---

## 10. How to Run

### Step 1: Create Virtual Environment

```
python -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```
pip install -r requirements.txt
```

### Step 3: Execute Pipeline

```
python -m src.pipeline.training_pipeline
```

---

## 11. Key Concepts Demonstrated

* Modular pipeline architecture
* Object-oriented design in ML systems
* Data validation strategies
* Scikit-learn Pipeline and ColumnTransformer
* Reproducible workflows

---

## 12. Current Status

Development completed up to Phase 4 (Data Preprocessing).

---

## 13. Future Work

* Feature engineering
* Train-test splitting
* Model training and evaluation
* Hyperparameter tuning
* Model deployment using an API

---

## 14. Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Flask (planned)

---

## 15. Author

Gurnaaz
