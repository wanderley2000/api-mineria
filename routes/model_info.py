from fastapi import APIRouter, HTTPException
from models.wine_model import model_state, ALL_FEATURES

router = APIRouter()


@router.get("/model/info", summary="Información del modelo actual")
def model_info():
    if not model_state["trained"]:
        raise HTTPException(
            status_code=400,
            detail="No hay modelo entrenado todavía. Llama a POST /train primero."
        )
    return {
        "model": model_state["model_name"],
        "features_used": model_state["features"],
        "total_features": len(model_state["features"]),
        "precision": model_state["precision"],
        "accuracy": model_state["accuracy"],
        "params": model_state["params"],
        "available_features": ALL_FEATURES,
    }
