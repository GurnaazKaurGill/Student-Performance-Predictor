"""
Subgroup Fairness Audit
------------------------
Runs the trained model and explainability pipeline across the entire
held-out test set (not just one prediction at a time), then aggregates
the results by demographic group to check two separate questions:

  1. Does the model's reasoning (SHAP contribution) systematically
     favor or penalize any group, on average?
  2. Is the model's accuracy (RMSE) meaningfully different for any
     group?

These are genuinely different questions. A model can be equally
*accurate* across groups while still *reasoning* about them
differently, and vice versa. Reporting both avoids drawing a
conclusion that only one metric would support.

This does not run automatically as part of training; it is a
standalone analysis script, run on demand:

    python -m src.pipeline.fairness_audit
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed, just saving files
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.features.engineering import FeatureEngineering
from src.features.preprocessing import DataPreprocessing
from src.pipeline.explain_pipeline import ExplainPipeline, FEATURE_DISPLAY_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def rebuild_test_set():
    """
    Reproduces the exact same train/test split used during training
    (same random_state), so the audit runs on the same held-out data
    the model was originally evaluated on, not a different sample.
    """
    ingestion = DataIngestion(
        input_path="data/raw/students.csv",
        output_path="data/processed/cleaned.csv",
    )
    df = ingestion.run()
    DataValidation(df).validate()

    fe = FeatureEngineering(df)
    df = fe.create_features()

    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return test_df


def plot_report(report: pd.DataFrame):
    """
    Saves one horizontal bar chart per demographic/behavioral group,
    showing average SHAP contribution by category. Colored green for
    positive (favored) and red for negative (penalized), matching the
    same color convention used in the frontend's per-prediction chart.
    """
    for group_name in report["group"].unique():
        subset = report[report["group"] == group_name].sort_values("avg_shap_contribution")

        colors = ["#ef4444" if v < 0 else "#10b981" for v in subset["avg_shap_contribution"]]

        fig, ax = plt.subplots(figsize=(6, max(2, 0.5 * len(subset))))
        ax.barh(subset["category"], subset["avg_shap_contribution"], color=colors)
        ax.axvline(0, color="#999", linewidth=0.8)
        ax.set_xlabel("Average SHAP contribution (math score points)")
        ax.set_title(f"{group_name}: average model contribution by category")
        plt.tight_layout()

        filename = f"reports/fairness_{group_name.lower().replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved chart: {filename}")


def run_audit():
    logger.info("Rebuilding held-out test set for audit")
    test_df = rebuild_test_set()
    y_test = test_df["math_score"].values

    preprocessor = joblib.load("artifacts/preprocessor.pkl")
    model = joblib.load("artifacts/model.pkl")
    X_test = preprocessor.transform(test_df)

    explain_pipeline = ExplainPipeline()
    explainer = explain_pipeline.explainer
    raw_feature_names = explain_pipeline.raw_feature_names

    logger.info(f"Computing SHAP values for {X_test.shape[0]} test rows")
    shap_values = explainer.shap_values(X_test)

    predictions = model.predict(X_test)
    squared_errors = (predictions - y_test) ** 2

    # Group raw feature names by which real-world column they belong to,
    # e.g. all the "cat__gender_*" columns belong to "gender".
    groups = {}
    for idx, raw_name in enumerate(raw_feature_names):
        prefix, name = raw_name.split("__", 1)
        for col in FEATURE_DISPLAY_NAMES:
            if name == col or name.startswith(col + "_"):
                groups.setdefault(col, []).append((idx, name))
                break

    rows = []
    for col, entries in groups.items():
        for idx, raw_name in entries:
            col_values = X_test[:, idx]

            if col_values.max() <= 1.0 and col_values.min() >= 0.0 and len(entries) > 1:
                # One-hot categorical column: only look at rows where
                # this specific category was actually selected.
                mask = col_values >= 0.5
                category = raw_name[len(col) + 1:] if raw_name != col else raw_name
            else:
                # Binary numeric feature (completed_prep, standard_lunch):
                # split into "yes" (1) and "no" (0) as two groups.
                mask = col_values >= 0.5
                category = "Yes"

            if mask.sum() == 0:
                continue

            rows.append({
                "group": FEATURE_DISPLAY_NAMES[col],
                "category": explain_pipeline._title_case(category) if col in
                    ("gender", "race/ethnicity", "parental_level_of_education")
                    else category,
                "count": int(mask.sum()),
                "avg_shap_contribution": round(float(shap_values[mask, idx].mean()), 2),
                "rmse": round(float(np.sqrt(squared_errors[mask].mean())), 2),
            })

            # For binary numeric features, also record the "No" group.
            if category == "Yes":
                mask_no = ~mask
                if mask_no.sum() > 0:
                    rows.append({
                        "group": FEATURE_DISPLAY_NAMES[col],
                        "category": "No",
                        "count": int(mask_no.sum()),
                        "avg_shap_contribution": round(float(shap_values[mask_no, idx].mean()), 2),
                        "rmse": round(float(np.sqrt(squared_errors[mask_no].mean())), 2),
                    })

    report = pd.DataFrame(rows).sort_values(["group", "avg_shap_contribution"], ascending=[True, False])

    overall_rmse = round(float(np.sqrt(squared_errors.mean())), 2)

    os.makedirs("reports", exist_ok=True)
    report.to_csv("reports/fairness_audit.csv", index=False)
    plot_report(report)

    logger.info(f"Overall test-set RMSE: {overall_rmse}")
    logger.info("Fairness audit report saved to reports/fairness_audit.csv")

    print(f"\nOverall test-set RMSE: {overall_rmse}\n")
    print(report.to_string(index=False))
    print(f"\nSaved to reports/fairness_audit.csv\n")

    return report


if __name__ == "__main__":
    run_audit()