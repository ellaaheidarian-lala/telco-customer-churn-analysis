from pydantic import BaseModel, Field

class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, description="Number of months the customer has stayed")
    monthly_charges: float = Field(..., ge=0.0, description="Monthly amount charged")
    total_charges: float = Field(..., ge=0.0, description="Total amount charged")
    contract_type: int = Field(..., ge=0, le=2, description="Contract type (0: Month-to-month, 1: One year, 2: Two year)")
    paperless_billing: int = Field(..., ge=0, le=1, description="Paperless billing (1: Yes, 0: No)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 12,
                    "monthly_charges": 70.35,
                    "total_charges": 844.20,
                    "contract_type": 0,
                    "paperless_billing": 1
                }
            ]
        }
    }

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    status: str
