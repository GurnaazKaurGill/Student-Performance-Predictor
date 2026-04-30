from src.data.ingestion import DataIngestion

if __name__ == "__main__":
    ingestion = DataIngestion(
        input_path= "data/raw/students.csv",
        output_path= "data/processed/cleaned.csv"
    )

    df = ingestion.run()
    print(df.head())