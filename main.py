from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.predict import router as predict_router
from routes.train import router as train_router
from routes.model_info import router as info_router

app = FastAPI(
    title="Wine Quality API",
    description="API de predicción de calidad de vino tinto usando Machine Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(train_router)
app.include_router(info_router)

@app.get("/")
def root():
    return {
        "message": "Wine Quality Prediction API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/train", "/model/info", "/docs"]
    }
