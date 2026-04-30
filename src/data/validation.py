import pandas as pd

REQUIRED_COLUMNS = {
    "gender": "object",
    "parental_level_of_education": "object",
    "lunch": "object",
    "test_preparation_course": "object",
    "math_score": "int64",
    "reading_score": "int64",
    "writing_score": "int64",
}


class DataValidation:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.errors = []

    def check_columns(self):
        missing = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            self.errors.append(f"Missing columns: {missing}")

    def check_dtypes(self):
        for col, dtype in REQUIRED_COLUMNS.items():
            if col in self.df.columns:
                if dtype == "object":
                    if not pd.api.types.is_string_dtype(self.df[col]):
                        self.errors.append(
                            f"Column {col} expected string, got {self.df[col].dtype}"
                        )
                else:
                    if self.df[col].dtype != dtype:
                        self.errors.append(
                            f"Column {col} expected {dtype}, got {self.df[col].dtype}"
                        )

    def check_missing_values(self):
        null_counts = self.df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                self.errors.append(f"{col} has {count} missing values")

    def check_duplicates(self):
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            self.errors.append(f"{dup_count} duplicate rows found")

    def check_ranges(self):
        for col in ["math_score", "reading_score", "writing_score"]:
            if col in self.df.columns:
                if not self.df[col].between(0, 100).all():
                    self.errors.append(f"{col} has values outside 0–100")

    def validate(self):
        self.check_columns()
        self.check_dtypes()
        self.check_missing_values()
        self.check_duplicates()
        self.check_ranges()

        if self.errors:
            raise ValueError("Validation failed:\n" + "\n".join(self.errors))

        return True