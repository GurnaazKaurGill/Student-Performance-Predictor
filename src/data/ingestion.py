import os
import pandas as pd

class DataIngestion:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"File not found at {self.input_path}")

        df = pd.read_csv(self.input_path)

        if df.empty:
            raise ValueError("Dataset is empty")

        return df

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        return df

    def save_processed_data(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)

    def run(self) -> pd.DataFrame:
        df = self.load_data()
        df = self.standardize_columns(df)
        self.save_processed_data(df)
        return df