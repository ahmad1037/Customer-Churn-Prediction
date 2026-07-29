from fastapi import FastAPI

from app.schemas import Customer
from app.predictor import predict_customer

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API"
    }

@app.post("/predict")
def predict(customer: Customer):

    result = predict_customer(
        customer.model_dump()
    )

    return result