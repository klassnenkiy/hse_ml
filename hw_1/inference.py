from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import io

app = FastAPI()

model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

class CarFeatures(BaseModel):
    name: str
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    seats: float
    car_class: str
    owner_and_seller: str
    mileage: float
    km_driven: float
    max_power: float
    year: int
    engine: float
    max_torque_rpm: float
    torque: float
    year_squared: float
    power_per_liter: float
    first_second_owner_and_dealer: float
    log_mileage: float

class CarFeaturesBatch(BaseModel):
    cars: List[CarFeatures]

def preprocess_input_data(car: CarFeatures) -> pd.DataFrame:
    return pd.DataFrame([car.dict()])

@app.get("/")
def root():
    return {
        "Name": "Tyulyagin Stanislav AI_HW1_Regression_with_inference_pro",
        "description": "Homework 1",
    }

@app.post("/predict_item")
def predict_item(car: CarFeatures):
    try:
        data = preprocess_input_data(car)
        processed_data = preprocessor.transform(data)
        prediction = model.predict(processed_data)
        return {"predicted_price": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_items")
def predict_bulk(file: UploadFile = File(...)):
    try:
        input_data = pd.read_csv(file.file)
        processed_data = preprocessor.transform(input_data)
        predictions = model.predict(processed_data)
        input_data["predicted_price"] = predictions

        output = io.StringIO()
        input_data.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))