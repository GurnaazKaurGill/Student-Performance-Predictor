import joblib
import pandas as pd
import shap

from src.features.engineering import FeatureEngineering
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Maps raw preprocessed column names to short, human-readable labels.
# Kept explicit rather than auto-derived, so the display stays correct
# even if internal naming conventions change later.
FEATURE_DISPLAY_NAMES = {
    "completed_prep": "Completed Test Prep",
    "standard_lunch": "Standard Lunch",
    "gender": "Gender",
    "race/ethnicity": "Race/Ethnicity",
    "parental_level_of_education": "Parental Education",
}


class ExplainPipeline:
    """
    Computes SHAP-based feature contributions for a single prediction.

    SHAP explains a prediction by comparing it against a "background"
    reference (a real sample of training data) and measuring how much
    each feature pushes the prediction away from that baseline, in the
    same units as the target (math score points). Because the final
    model is Linear Regression, shap.LinearExplainer computes these
    values exactly, with no approximation needed.
    """

    def __init__(
        self,
        model_path="artifacts/model.pkl",
        preprocessor_path="artifacts/preprocessor.pkl",
        background_path="artifacts/background_data.pkl",
    ):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        background_data = joblib.load(background_path)

        self.explainer = shap.LinearExplainer(self.model, background_data)
        self.raw_feature_names = self.preprocessor.get_feature_names_out()
        self.base_value = round(float(self.explainer.expected_value), 2)

        logger.info("Explainability pipeline initialized")

    def _title_case(self, text: str) -> str:
        """
        Capitalizes each word's first letter without mangling
        apostrophes the way Python's built-in str.title() does
        (e.g. "bachelor's degree" -> "Bachelor's Degree", not
        "Bachelor'S Degree").
        """
        return " ".join(word[:1].upper() + word[1:] for word in text.split(" "))

    def _display_name(self, raw_name: str) -> str:
        """
        Converts a raw column name like
        'cat__parental_level_of_education_bachelor's degree' into
        'Parental Education: Bachelor's Degree'.
        """
        raw_name = raw_name.split("__", 1)[1]  # drop 'num__' / 'cat__' prefix

        for col, label in FEATURE_DISPLAY_NAMES.items():
            if raw_name == col:
                return label
            if raw_name.startswith(col + "_"):
                value = raw_name[len(col) + 1:]
                return f"{label}: {self._title_case(value)}"

        return raw_name  # fallback; should not normally be reached

    def explain(self, input_dict: dict) -> dict:
        """
        Returns the base value (average prediction over the background
        sample) and a list of each relevant feature's contribution for
        this specific input, in math-score points.

        For one-hot encoded categorical features, only the category the
        student actually has is included (e.g. only "Gender: Female",
        never also a near-zero "Gender: Male" entry), since showing an
        unselected category's contribution would be misleading.
        """
        df = pd.DataFrame([input_dict])
        df["math_score"] = 0  # placeholder; unused by feature engineering

        fe = FeatureEngineering(df)
        df = fe.create_features()

        X = self.preprocessor.transform(df)
        shap_values = self.explainer.shap_values(X)[0]
        x_row = X[0]

        contributions = []
        for name, shap_val, x_val in zip(self.raw_feature_names, shap_values, x_row):
            prefix = name.split("__", 1)[0]

            # For categorical (one-hot) features, only include the
            # category actually present for this input (value == 1).
            if prefix == "cat" and x_val < 0.5:
                continue

            contributions.append({
                "feature": self._display_name(name),
                "contribution": round(float(shap_val), 2),
            })

        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

        logger.info(f"Explanation computed: {len(contributions)} features")

        return {
            "base_value": self.base_value,
            "contributions": contributions,
        }