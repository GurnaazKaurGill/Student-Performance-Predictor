import pandas as pd 

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_features(self) -> pd.DataFrame:
        self.df["avg_score"] = (
            self.df["reading_score"] + self.df["writing_score"]
        ) / 2

        self.df["score_gap"] = abs(
            self.df["reading_score"] - self.df["writing_score"]
        )

        return self.df
