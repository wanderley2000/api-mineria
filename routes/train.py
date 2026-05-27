from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from models.wine_model import train_model, ALL_FEATURES

router = APIRouter()


class TrainRequest(BaseModel):
    model_name: str = Field(
        "random_forest",
        description="Modelo a usar: 'random_forest', 'decision_tree', 'knn'",
        example="random_forest",
    )
    features: Optional[List[str]] = Field(
        None,
        description="Lista de features a usar. Si no se indica, se usan todas.",
        example=["alcohol", "sulphates", "volatile acidity"],
    )
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Proporción de test (0.1 – 0.5)")
    scale: bool = Field(True, description="Aplicar StandardScaler")
    sample_size: float = Field(1.0, ge=0.1, le=1.0, description="Fracción del dataset a usar")

    # Params por modelo
    n_estimators: int = Field(100, ge=10, le=500, description="[RandomForest] Número de árboles")
    max_depth: Optional[int] = Field(None, ge=1, le=50, description="[RF/Tree] Profundidad máxima")
    n_neighbors: int = Field(5, ge=1, le=50, description="[KNN] Cantidad de vecinos")


@router.post("/train", summary="Entrenar / reentrenar el modelo")
def train(body: TrainRequest):
    valid_models = {"random_forest", "decision_tree", "knn"}
    if body.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"model_name inválido. Opciones: {sorted(valid_models)}"
        )

    features = body.features or ALL_FEATURES
    invalid = [f for f in features if f not in ALL_FEATURES]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Features inválidas: {invalid}. Disponibles: {ALL_FEATURES}"
        )

    try:
        result = train_model(
            model_name=body.model_name,
            features=features,
            test_size=body.test_size,
            scale=body.scale,
            sample_size=body.sample_size,
            n_estimators=body.n_estimators,
            max_depth=body.max_depth,
            n_neighbors=body.n_neighbors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al entrenar: {str(e)}")

    return {"message": "Modelo entrenado exitosamente", **result}
