from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from models.wine_model import model_state

router = APIRouter()


class WineFeatures(BaseModel):
    fixed_acidity: Optional[float] = Field(None, example=7.4)
    volatile_acidity: Optional[float] = Field(None, example=0.70)
    citric_acid: Optional[float] = Field(None, example=0.0)
    residual_sugar: Optional[float] = Field(None, example=1.9)
    chlorides: Optional[float] = Field(None, example=0.076)
    free_sulfur_dioxide: Optional[float] = Field(None, example=11.0)
    total_sulfur_dioxide: Optional[float] = Field(None, example=34.0)
    density: Optional[float] = Field(None, example=0.9978)
    pH: Optional[float] = Field(None, example=3.51)
    sulphates: Optional[float] = Field(None, example=0.56)
    alcohol: Optional[float] = Field(None, example=9.4)


FEATURE_MAP = {
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "chlorides": "chlorides",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide",
    "density": "density",
    "pH": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}


@router.post("/predict", summary="Predecir calidad del vino")
def predict(wine: WineFeatures):
    if not model_state["trained"]:
        raise HTTPException(
            status_code=400,
            detail="El modelo no ha sido entrenado. Llama primero a POST /train"
        )

    model = model_state["model"]
    scaler = model_state["scaler"]
    features = model_state["features"]

    wine_dict = {
        "fixed acidity": wine.fixed_acidity,
        "volatile acidity": wine.volatile_acidity,
        "citric acid": wine.citric_acid,
        "residual sugar": wine.residual_sugar,
        "chlorides": wine.chlorides,
        "free sulfur dioxide": wine.free_sulfur_dioxide,
        "total sulfur dioxide": wine.total_sulfur_dioxide,
        "density": wine.density,
        "pH": wine.pH,
        "sulphates": wine.sulphates,
        "alcohol": wine.alcohol,
    }

    missing = [f for f in features if wine_dict.get(f) is None]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Faltan los siguientes parámetros requeridos: {missing}"
        )

    import pandas as pd
    row = pd.DataFrame([[wine_dict[f] for f in features]], columns=features)

    if scaler:
        row = scaler.transform(row)

    prediction = model.predict(row)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = round(float(model.predict_proba(row)[0][int(prediction)]), 4)

    label = "bueno" if prediction == 1 else "malo"

    return {
        "prediction": label,
        "prediction_code": int(prediction),
        "confidence": proba,
        "model_used": model_state["model_name"],
    }
