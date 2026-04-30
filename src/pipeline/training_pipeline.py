from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation

if __name__ == "__main__":
    ingestion = DataIngestion(
        input_path= "data/raw/students.csv",
        output_path= "data/processed/cleaned.csv"
    )

    df = ingestion.run()
    print(df.head())
    print(df.info())

    validator = DataValidation(df)
    validator.validate()

    print("Data validation passed")