import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
import joblib


class DataPreprocessing:
    def __init__(self):
        self.num_features = ["reading_score", "writing_score"]
        self.cat_features = [
            "gender",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course"
        ]

        self.preprocessor = None

    def create_pipeline(self):
        num_pipeline = Pipeline(steps= [
            ("imputer", SimpleImputer(strategy= "median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline(steps= [
            ("imputer", SimpleImputer(strategy= "most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown= "ignore"))
        ])

        self.preprocessor = ColumnTransformer(transformers= [
            ("num", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

    def fit_transform(self, df: pd.DataFrame):
        X = df.drop("math_score", axis=1)
        y = df["math_score"]

        X_transformed = self.preprocessor.fit_transform(X)

        return X_transformed, y

    def transform(self, df: pd.DataFrame):
        X = df.drop("math_score", axis=1)
        return self.preprocessor.transform(X)

    def save(self, path="artifacts/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.preprocessor, path)