from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# Find project root
current_file = Path(__file__).resolve()

for path in current_file.parents:
    if (path / "pyproject.toml").exists():
        PROJECT_ROOT = path
        break
else:
    raise FileNotFoundError(
        "Project root could not be found."
    )


# Model path
MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "campaign_response_model.joblib"
)


# Load trained model
model = joblib.load(
    MODEL_FILE
)


# Create FastAPI application
app = FastAPI(
    title="Retail Campaign Response API",
    version="1.0.0",
)


# Configure static files
app.mount(
    "/static",
    StaticFiles(
        directory=PROJECT_ROOT / "app" / "static"
    ),
    name="static",
)


# Configure HTML templates
templates = Jinja2Templates(
    directory=PROJECT_ROOT / "app" / "templates"
)


# Define the data expected from the user
class CustomerInput(BaseModel):

    total_sales: float = Field(ge=0)
    unique_products: int = Field(ge=0)
    number_of_invoices: int = Field(ge=0)

    nps: float | None = Field(
        default=None,
        ge=0,
        le=10
    )

    n_comp: int = Field(ge=0)
    n_communications: int = Field(ge=0)

    loyalty: int = Field(
        ge=0,
        le=1
    )


@app.get("/")
def root(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
def predict(
    customer: CustomerInput
):

    customer_data = customer.model_dump()

    if customer_data["nps"] is None:
        customer_data["nps"] = np.nan

    input_df = pd.DataFrame(
        [customer_data]
    )

    probability = model.predict_proba(
        input_df
    )[0, 1]

    prediction = model.predict(
        input_df
    )[0]

    prediction_label = (
        "Likely to respond"
        if prediction == 1
        else "Unlikely to respond"
    )

    return {
        "response_probability": round(
            float(probability),
            4
        ),
        "response_probability_percent": round(
            float(probability) * 100,
            2
        ),
        "predicted_response": int(
            prediction
        ),
        "prediction_label": prediction_label
    }