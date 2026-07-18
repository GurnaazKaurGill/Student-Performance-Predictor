from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.predict_pipeline import PredictPipeline
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
        return {"predicted_math_score": prediction}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))