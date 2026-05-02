from sklearn.model_selection import train_test_split

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.features.engineering import FeatureEngineering
from src.features.preprocessing import DataPreprocessing
from src.models.train import ModelTrainer

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


    fe = FeatureEngineering(df)
    df = fe.create_features()

    train_df, test_df = train_test_split(
        df, 
        test_size = 0.2, 
        random_state= 42
    )
    print("Train shape: ", train_df.shape)
    print("Test shape: ", test_df.shape)


    preprocessor = DataPreprocessing()
    preprocessor.create_pipeline()

    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_test = test_df["math_score"]
    preprocessor.save()

    print("Preprocessing Completed")
    print("Train features Shape: ", X_train.shape)


    trainer = ModelTrainer()
    best_model = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

    trainer.save_model(best_model)
    print("Best model saved")