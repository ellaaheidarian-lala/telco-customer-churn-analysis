import os
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from app.schemas import CustomerData, PredictionResponse
from app.model import ModelLoader

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready MLOps microservice for predicting customer churn probability.",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.pkl")
model_service = None


@app.on_event("startup")
def load_artifacts():
    global model_service
    try:
        model_service = ModelLoader(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model on startup: {e}")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for Docker container & orchestrator status monitoring."""
    if model_service is None or model_service.model is None:
        raise HTTPException(status_code=500, detail="Model service uninitialized")
    return {"status": "healthy", "service": "churn-prediction-api"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerData):
    """Predict customer churn based on telemetry input features."""
    if model_service is None or model_service.model is None:
        raise HTTPException(status_code=500, detail="Model is not currently loaded.")

    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction, probability = model_service.predict(input_df)

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            status="success",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
