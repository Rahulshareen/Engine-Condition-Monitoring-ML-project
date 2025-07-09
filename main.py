from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("engine_failure_model.pkl")      
scaler = joblib.load("scaler.pkl")   

# Initialize app
app = FastAPI()

# Health check
@app.get("/health")
def health_check():
    return {"status": "API is healthy"}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Engine Condition Monitoring API Check"}

# Define input format
class EngineData(BaseModel):
    engine_rpm: float
    lub_oil_pressure: float
    fuel_pressure: float
    coolant_pressure: float
    lub_oil_temp: float
    coolant_temp: float

# Prediction endpoint
@app.post("/predict")
def predict_engine_failure(data: EngineData):
    # Convert input to array
    input_data = np.array([[data.engine_rpm,
                            data.lub_oil_pressure,
                            data.fuel_pressure,
                            data.coolant_pressure,
                            data.lub_oil_temp,
                            data.coolant_temp]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of class '1'

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }

