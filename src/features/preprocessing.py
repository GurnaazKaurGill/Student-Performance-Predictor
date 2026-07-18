import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib


class DataPreprocessing:
    """
    Builds and applies the preprocessing pipeline.

    Numeric features here are the engineered 0/1 flags, NOT reading_score
    or writing_score. Those two columns never reach this class as inputs,
    because they come from the same exam sitting as math_score, our
    target, and would not be known ahead of time in a real use case.
    """

    def __init__(self):
        self.num_features = [
            "completed_prep",
            "standard_lunch",
        ]

        self.cat_features = [
            "gender",
            "race/ethnicity",
            "parental_level_of_education",
        ]

        self.preprocessor = None

    def create_pipeline(self):
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Explicitly SELECTS allowed columns instead of just dropping the
        target. Even if the raw data later gains a new column, it can't
        silently leak in, because only columns named here are ever used.
        """
        allowed = self.num_features + self.cat_features
        return df[allowed]

    def fit_transform(self, df: pd.DataFrame):
        X = self._select_features(df)
        y = df["math_score"]

        X_transformed = self.preprocessor.fit_transform(X)
        return X_transformed, y

    def transform(self, df: pd.DataFrame):
        X = self._select_features(df)
        return self.preprocessor.transform(X)

    def save(self, path="artifacts/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.preprocessor, path)