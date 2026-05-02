import os
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , mean_absolute_error

class ModelTrainer:
    def __init__(self):
        self.models= {
            "linear_regression": LinearRegression(),
            "randm_forest": RandomForestRegressor(random_state=42)
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

    def save_model(self, model, path="artifacts/model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)