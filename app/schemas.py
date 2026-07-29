from pydantic import BaseModel

class Customer(BaseModel):
    Gender: str
    Senior_Citizen: int
    Partner: str
    Dependents: str
    Tenure_Months: int
    Internet_Service: str
    Contract: str
    Monthly_Charges: float
    Total_Charges: float
    Paperless_Billing: str
    Payment_Method: str
    CLTV: float