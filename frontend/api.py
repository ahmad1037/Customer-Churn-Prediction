import os
import requests

API_URL = os.getenv("API_URL", "http://api:8000")


def predict(customer):
    response = requests.post(
        f"{API_URL}/predict",
        json=customer,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()