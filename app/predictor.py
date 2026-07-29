
import pandas as pd

from app.model_loader import (
    model,
    preprocessor,
)


# Default values for fields that are not provided by the API.
# IMPORTANT:
# These column names must EXACTLY match the columns used
# when the preprocessor was trained.
DEFAULT_VALUES = {
    "Country": "United States",
    "State": "California",
    "City": "Los Angeles",
    "Zip Code": 90001,
    "Lat Long": "33.973951,-118.248405",
    "Latitude": 33.973951,
    "Longitude": -118.248405,
    "Gender": "Male",
    "Senior Citizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "DSL",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 50.0,
    "Total Charges": 600.0,
    "CLTV": 4000.0,
}


def predict_customer(customer: dict):

    # Start with default values
    row = DEFAULT_VALUES.copy()

    # Update the default values with the values
    # received from the API request.
    #
    # The LEFT side must match the original training
    # dataset column names.
    #
    # The RIGHT side must match the Pydantic model field names.
    row.update({
        "Gender": customer["Gender"],
        "Senior Citizen": customer["Senior_Citizen"],
        "Partner": customer["Partner"],
        "Dependents": customer["Dependents"],
        "Tenure Months": customer["Tenure_Months"],
        "Internet Service": customer["Internet_Service"],
        "Contract": customer["Contract"],
        "Monthly Charges": customer["Monthly_Charges"],
        "Total Charges": customer["Total_Charges"],
        "Paperless Billing": customer["Paperless_Billing"],
        "Payment Method": customer["Payment_Method"],
        "CLTV": customer["CLTV"],
    })

    # Convert to DataFrame
    df = pd.DataFrame([row])

    # Apply the same preprocessing used during training
    transformed = preprocessor.transform(df)

    probability = model.predict_proba(transformed)[0][1]

    prediction = int(probability >= 0.50)

    return {
    "prediction": prediction,
    "churn_probability": round(float(probability), 4),
    "risk_level": (
        "High"
        if probability >= 0.70
        else "Medium"
        if probability >= 0.40
        else "Low"
    )
}