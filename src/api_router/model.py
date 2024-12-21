from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import pickle
import xgboost as xgb
import sys
import os
from src.api_router.preprocess import Preprocessor 
sys.modules["__main__"].Preprocessor = Preprocessor
# Router
router = APIRouter()

# Load Preprocessor
PREPROCESSOR_PATH = os.path.join('src',"checkpoint","preprocess.pkl")
with open(PREPROCESSOR_PATH, "rb") as f:
    prep = pickle.load(f)
print(f"Preprocessor loaded from {PREPROCESSOR_PATH}.")

# Load XGBoost Model
MODEL_PATH =  os.path.join('src',"checkpoint","xgboost_heart_disease_model.pth")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Input Schema
class InputData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

# Preprocess Function
def preprocess_input(data: InputData):
    input_dict = {
        "Age": [data.Age],
        "Sex": [data.Sex],
        "ChestPainType": [data.ChestPainType],
        "RestingBP": [data.RestingBP],
        "Cholesterol": [data.Cholesterol],
        "FastingBS": [data.FastingBS],
        "RestingECG": [data.RestingECG],
        "MaxHR": [data.MaxHR],
        "ExerciseAngina": [data.ExerciseAngina],
        "Oldpeak": [data.Oldpeak],
        "ST_Slope": [data.ST_Slope],
    }
    input_df = pd.DataFrame(input_dict)
    transformed_input = prep.transform(input_df)
    return transformed_input

# API Endpoint
@router.post("/predict")
async def predict_heart_disease(data: InputData):
    input_data = preprocess_input(data)
    prediction = model.predict(input_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "Normal"
    return {"prediction": result}
