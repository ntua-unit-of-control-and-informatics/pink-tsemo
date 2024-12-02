from typing import Union

from fastapi import FastAPI
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_response import PredictionResponse

from src.tsemo import handle_tsemo

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/infer")
def predict(req: PredictionRequest) -> PredictionResponse:
    return handle_tsemo(req.dataset.input)

