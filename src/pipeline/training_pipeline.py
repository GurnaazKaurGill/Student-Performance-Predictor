from sklearn.model_selection import train_test_split

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.features.engineering import FeatureEngineering
from src.features.preprocessing import DataPreprocessing
from src.models.train import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    ingestion = DataIngestion(
        input_path="data/raw/students.csv",
        output_path="data/processed/cleaned.csv"
    )

    df = ingestion.run()
    logger.info(f"Data loaded: {df.shape}")

    validator = DataValidation(df)
    validator.validate()
    logger.info("Data validation passed")

    fe = FeatureEngineering(df)
    df = fe.create_features()
    logger.info("Feature engineering complete")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    preprocessor = DataPreprocessing()
    preprocessor.create_pipeline()

    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_test = test_df["math_score"]
    preprocessor.save()
    logger.info("Preprocessing completed and preprocessor saved")

    trainer = ModelTrainer()
    base_model = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

    logger.info("Tuning Random Forest...")
    tuned_rf = trainer.tune_random_forest(X_train, y_train)

    preds = tuned_rf.predict(X_test)
    rmse, mae = trainer.evaluate(y_test, preds)
    logger.info(f"Tuned RF -> RMSE: {rmse:.3f}, MAE: {mae:.3f}")

    best_existing_rmse = min(
        result["rmse"] for result in trainer.results.values()
    )

    if rmse < best_existing_rmse:
        final_model = tuned_rf
        logger.info("Tuned Random Forest selected as final model")
    else:
        final_model = base_model
        logger.info("Best base model selected as final model")

    trainer.save_model(final_model)
    logger.info("Final model saved")