import os
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

class ModelTrainer:
    def __init__(self):
        self.models= {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(random_state=42)
        }
        self.results = {}

    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        return rmse, mae

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        best_model = None
        best_score = float("inf")

        for name, model in self.models.items():
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            rmse, mae = self.evaluate(y_test, preds)

            self.results[name] = {
                "rmse": rmse,
                "mae": mae
            }

            print(f"{name} -> RMSE: {rmse:.3f}, MAE: {mae:.3f}")

            if rmse < best_score:
                best_score = rmse
                best_model = model

        return best_model

    def tune_random_forest(self, X_train, y_train):
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }

        rf = RandomForestRegressor(random_state=42)

        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        search.fit(X_train, y_train)

        print("Best RF Params:", search.best_params_)
        print("Best CV RMSE:", -search.best_score_)

        return search.best_estimator_

    def save_model(self, model, path="artifacts/model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)