import uvicorn
from fastapi import FastAPI
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_response import PredictionResponse

from src.loggers.log_middleware import LogMiddleware
from src.tsemo import handle_tsemo

app = FastAPI()
app.add_middleware(LogMiddleware)

@app.get("/")
def read_root():
    return {"Status": "OK"}

@app.post("/infer")
def predict(req: PredictionRequest) -> PredictionResponse:
    return handle_tsemo(req.model, req.dataset.input)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005, log_config=None)
