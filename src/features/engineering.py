import pandas as pd


class FeatureEngineering:
    """
    Creates new features from the raw dataset.

    IMPORTANT: math_score is our prediction target. reading_score and
    writing_score are scores from the SAME exam sitting as math_score,
    so they are not real-world "inputs" you'd have before a student
    takes the math exam. We do not turn them into features here.

    Every feature this class creates is built ONLY from columns that
    would realistically be known in advance: demographics and
    preparation history.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_features(self) -> pd.DataFrame:
        df = self.df

        df["completed_prep"] = (
            df["test_preparation_course"].str.lower() == "completed"
        ).astype(int)

        df["standard_lunch"] = (
            df["lunch"].str.lower() == "standard"
        ).astype(int)

        return df