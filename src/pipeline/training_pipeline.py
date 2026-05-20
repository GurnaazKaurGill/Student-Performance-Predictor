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
    print("Test features shape:", X_test.shape)


    trainer = ModelTrainer()
    base_model = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

    # Hyperparameter tuning
    print("\nTuning Random Forest...")

    tuned_rf = trainer.tune_random_forest(
        X_train,
        y_train
    )


    # Evaluate Tuned Model
    preds = tuned_rf.predict(X_test)
    rmse, mae = trainer.evaluate(
        y_test,
        preds
    )
    print(f"Tuned RF -> RMSE: {rmse:.3f}, MAE: {mae:.3f}")


    # Compare Models
    best_existing_rmse = min(
        result["rmse"]
        for result in trainer.results.values()
    )

    if rmse < best_existing_rmse:
        final_model = tuned_rf
        print("Tuned Random Forest selected as final model")
    else:
        final_model = base_model
        print("Best base model selected as final model")

    # Final Model
    trainer.save_model(final_model)
    print("Final model saved")