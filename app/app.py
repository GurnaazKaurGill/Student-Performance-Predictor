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


# Serves everything in app/static/ as static files. html=True means
# requesting the root path serves index.html automatically.
# This must be added AFTER the /predict and / routes above, since
# FastAPI checks routes in the order they're defined, and a route
# mounted at "/" would otherwise intercept every request before it
# reaches /predict.
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")