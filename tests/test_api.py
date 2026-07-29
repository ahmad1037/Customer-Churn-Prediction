from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_home():

    response = client.get("/")

    assert response.status_code == 200

def test_prediction():

    payload = {
        "Gender": "Male",
        "Senior_Citizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure_Months": 36,
        "Internet_Service": "Fiber optic",
        "Contract": "Month-to-month",
        "Monthly_Charges": 89.5,
        "Total_Charges": 3222.6,
        "Paperless_Billing": "Yes",
        "Payment_Method": "Electronic check",
        "CLTV": 4200
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()