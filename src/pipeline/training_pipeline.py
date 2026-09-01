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

    # Save a sample of the transformed training data as a "background"
    # reference for SHAP. SHAP explains a prediction by comparing it
    # against what a typical input looks like; this sample is that
    # baseline. 100 rows is plenty for a dataset this size and keeps
    # explanation computation fast.
    import joblib
    import numpy as np

    background_sample = X_train[np.random.RandomState(42).choice(
        X_train.shape[0], size=min(100, X_train.shape[0]), replace=False
    )]
    joblib.dump(background_sample, "artifacts/background_data.pkl")
    logger.info("Background data sample saved for explainability")

    # Persist a small metrics summary so the dashboard's Model Performance
    # page can show real numbers from THIS training run, rather than
    # numbers hardcoded into the frontend that would silently go stale
    # the next time the model is retrained.
    #
    # The "before leakage" figures are a historical record, not something
    # this pipeline can regenerate: the current feature engineering
    # deliberately no longer builds the leaking features at all, so
    # there is nothing left to re-measure. They are recorded here once,
    # by hand, from the last run of the superseded code, and labeled
    # accordingly for the frontend to display honestly.
    import json

    final_model_name = "Random Forest (tuned)" if final_model is tuned_rf else \
        [name for name, m in trainer.models.items() if m is final_model][0].replace("_", " ").title()

    metrics = {
        "before_leakage_note": "Historical record from a superseded version that used reading_score and writing_score as features (later identified as data leakage and removed). Not reproducible from the current pipeline.",
        "before_leakage": {
            "linear_regression": {"rmse": 5.366, "mae": 4.227},
            "random_forest": {"rmse": 6.259, "mae": 4.954},
        },
        "after_leakage_fix": {
            name.replace("_", " ").title(): {
                "rmse": round(result["rmse"], 3),
                "mae": round(result["mae"], 3),
            }
            for name, result in trainer.results.items()
        },
        "tuned_random_forest": {"rmse": round(rmse, 3), "mae": round(mae, 3)},
        "final_model": final_model_name,
        "test_set_size": int(X_test.shape[0]),
        "train_set_size": int(X_train.shape[0]),
        "feature_count": int(X_train.shape[1]),
    }

    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Model metrics saved to artifacts/metrics.json")