import requests

API_URL = "http://127.0.0.1:8000/predict"

def predict(customer):

    response = requests.post(
        API_URL,
        json=customer,
        timeout=10
    )

    response.raise_for_status()

    return response.json()