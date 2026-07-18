import pandas as pd
import joblib

from src.features.engineering import FeatureEngineering
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictPipeline:
    """
    Loads the saved preprocessor and model, and turns a single new
    student's details into a predicted math_score.
    """

    def __init__(
        self,
        model_path="artifacts/model.pkl",
        preprocessor_path="artifacts/preprocessor.pkl",
    ):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        logger.info("Model and preprocessor loaded for inference")

    def predict(self, input_dict: dict) -> float:
        df = pd.DataFrame([input_dict])

        # FeatureEngineering expects a math_score column to exist because
        # it was built for training data; this placeholder has no effect
        # on the output since it never feeds into completed_prep or
        # standard_lunch.
        df["math_score"] = 0

        fe = FeatureEngineering(df)
        df = fe.create_features()

        X = self.preprocessor.transform(df)
        prediction = self.model.predict(X)[0]

        logger.info(f"Prediction made: {prediction:.2f}")
        return round(float(prediction), 2)