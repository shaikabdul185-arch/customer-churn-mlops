"""Production FastAPI inference & drift monitoring service for Customer Churn MLOps."""
from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Customer Churn Prediction & MLOps API", version="1.0.0")


class CustomerRecord(BaseModel):
    customer_id: str
    tenure_months: float
    contract_type: str
    payment_method: str
    internet_service: str
    tech_support: str
    monthly_charges: float
    total_charges: float = 0.0
    support_tickets: float = 0.0
    usage_change_pct: float = 0.0
    late_payments_last_year: int = 0


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "version": "1.0.0"}
