import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.explain_pipeline import ExplainPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Student Performance Predictor",
    description="Predicts a student's math score from demographic and preparation details.",
    version="1.0.0",
)

# Loaded once at startup, same reasoning as before: loading a model from
# disk on every request would be slow.
pipeline = PredictPipeline()
explainer = ExplainPipeline()


@app.on_event("startup")
def announce_urls():
    print("\n" + "=" * 60)
    print("Student Performance Predictor is ready:")
    print("  Frontend:  http://127.0.0.1:8000/ui/")
    print("  API docs:  http://127.0.0.1:8000/docs")
    print("=" * 60 + "\n")


class StudentInput(BaseModel):
    """
    Defines exactly what a valid request looks like. FastAPI reads this
    class and automatically:
      - rejects requests missing any of these fields
      - rejects requests with the wrong type (e.g. a number where a
        string is expected)
      - generates the interactive docs at /docs from this same class
    This replaces the manual `missing = [...]` check we wrote by hand
    in the Flask version.
    """
    gender: str = Field(examples=["female"])
    race_ethnicity: str = Field(examples=["group B"], alias="race/ethnicity")
    parental_level_of_education: str = Field(examples=["bachelor's degree"])
    lunch: str = Field(examples=["standard"])
    test_preparation_course: str = Field(examples=["completed"])

    class Config:
        populate_by_name = True


@app.get("/")
def home():
    return {"status": "Student Performance Predictor API is running"}


@app.post("/predict")
def predict(student: StudentInput):
    try:
        input_dict = student.model_dump(by_alias=True)
        prediction = pipeline.predict(input_dict)
        explanation = explainer.explain(input_dict)

        return {
            "predicted_math_score": prediction,
            "base_value": explanation["base_value"],
            "contributions": explanation["contributions"],
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info():
    """
    Real metrics from the last actual training run, read from
    artifacts/metrics.json (written by training_pipeline.py). If the
    model hasn't been trained yet on this machine, this returns a
    clear error rather than fabricated numbers.
    """
    try:
        with open("artifacts/metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Run the training pipeline first: "
                   "python -m src.pipeline.training_pipeline",
        )


@app.get("/fairness-audit")
def fairness_audit():
    """
    Serves the subgroup fairness audit (reports/fairness_audit.csv,
    produced by src/pipeline/fairness_audit.py) as JSON. Returns a
    clear error, not fabricated data, if the audit hasn't been run yet.
    """
    try:
        df = pd.read_csv("reports/fairness_audit.csv")
        return {"rows": df.to_dict(orient="records")}
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No fairness audit found. Run it first: "
                   "python -m src.pipeline.fairness_audit",
        )


@app.get("/dataset-info")
def dataset_info():
    """
    Live summary statistics computed directly from the raw training
    data on every request. The dataset is small (1,000 rows), so this
    is cheap enough to compute on demand rather than caching.
    """
    try:
        df = pd.read_csv("data/raw/students.csv")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Raw dataset not found.")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    def value_counts(col):
        return df[col].value_counts().sort_index().to_dict()

    return {
        "total_students": int(len(df)),
        "average_math_score": round(float(df["math_score"].mean()), 2),
        "average_reading_score": round(float(df["reading_score"].mean()), 2),
        "average_writing_score": round(float(df["writing_score"].mean()), 2),
        "gender": value_counts("gender"),
        "race_ethnicity": value_counts("race/ethnicity"),
        "parental_level_of_education": value_counts("parental_level_of_education"),
        "lunch": value_counts("lunch"),
        "test_preparation_course": value_counts("test_preparation_course"),
    }


# Serves everything in app/static/ as static files. html=True means
# requesting the root path serves index.html automatically.
# This must be added AFTER the /predict and / routes above, since
# FastAPI checks routes in the order they're defined, and a route
# mounted at "/" would otherwise intercept every request before it
# reaches /predict.
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")