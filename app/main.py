from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "model" / "artifacts" / "cars_pipe.pkl"

loaded = joblib.load(MODEL_PATH)
model = loaded["model"]
label_encoder = loaded["label_encoder"]
metadata = loaded["metadata"]

app = FastAPI(
    title="Car Price Category Prediction API",
    description="Predicts a car price category from listing attributes.",
    version=str(metadata.get("version", "1")),
)


class CarFeatures(BaseModel):
    region: str = Field(..., examples=["baltimore"])
    year: int = Field(..., examples=[2013])
    manufacturer: str = Field(..., examples=["ford"])
    model: str = Field(..., examples=["mustang"])
    fuel: str = Field(..., examples=["gas"])
    odometer: float = Field(..., examples=[85000])
    title_status: str = Field(..., examples=["clean"])
    transmission: str = Field(..., examples=["manual"])
    state: str = Field(..., examples=["md"])
    lat: float = Field(..., examples=[39.1618])
    long: float = Field(..., examples=[-76.6297])

    class Config:
        extra = "ignore"


class Prediction(BaseModel):
    price_category: str


@app.get("/status")
def status() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, Any]:
    return metadata


@app.post("/predict", response_model=Prediction)
def predict(form: CarFeatures) -> Prediction:
    df = pd.DataFrame([form.model_dump()])
    prediction_encoded = model.predict(df)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)
    return Prediction(price_category=prediction_label[0])

