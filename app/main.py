from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# --------------------------------------------------
# Find project root
# --------------------------------------------------

current_file = Path(__file__).resolve()

for path in current_file.parents:

    if (path / "pyproject.toml").exists():

        PROJECT_ROOT = path
        break

else:

    raise FileNotFoundError(
        "Project root could not be found."
    )


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "campaign_response_model.joblib"
)


model = joblib.load(
    MODEL_FILE
)


FEATURES = [
    "total_sales",
    "unique_products",
    "number_of_invoices",
    "nps",
    "n_comp",
    "n_communications",
    "loyalty",
]


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Retail Campaign Response API",
    version="1.0.0",
)


# --------------------------------------------------
# Static files
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(
        directory=PROJECT_ROOT
        / "app"
        / "static"
    ),
    name="static",
)


# --------------------------------------------------
# HTML templates
# --------------------------------------------------

templates = Jinja2Templates(
    directory=PROJECT_ROOT
    / "app"
    / "templates"
)


# --------------------------------------------------
# Single prediction input model
# --------------------------------------------------

class CustomerInput(BaseModel):

    total_sales: float = Field(
        ge=0
    )

    unique_products: int = Field(
        ge=0
    )

    number_of_invoices: int = Field(
        ge=0
    )

    nps: float | None = Field(
        default=None,
        ge=0,
        le=10
    )

    n_comp: int = Field(
        ge=0
    )

    n_communications: int = Field(
        ge=0
    )

    loyalty: int = Field(
        ge=0,
        le=1
    )


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.get("/")
def root(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# --------------------------------------------------
# Single customer prediction
# --------------------------------------------------

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

        "prediction_label":
            prediction_label
    }


# --------------------------------------------------
# Batch prediction
# --------------------------------------------------

@app.post("/predict-batch")
def predict_batch(
    file: UploadFile = File(...)
):

    # ----------------------------------------------
    # Validate file extension
    # ----------------------------------------------

    if not file.filename.lower().endswith(
        ".csv"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only CSV files are supported."
            )
        )


    # ----------------------------------------------
    # Read CSV
    # ----------------------------------------------

    try:

        batch_data = pd.read_csv(
            file.file
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "The CSV file could not be read."
            )
        ) from error


    # ----------------------------------------------
    # Validate required columns
    # ----------------------------------------------

    missing_columns = [

        column

        for column in FEATURES

        if column not in batch_data.columns

    ]


    if missing_columns:

        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Required columns are missing.",

                "missing_columns":
                    missing_columns,
            }
        )


    # ----------------------------------------------
    # Select model input columns
    # ----------------------------------------------

    batch_input = batch_data[
        FEATURES
    ].copy()


    # ----------------------------------------------
    # Convert model columns to numeric
    # ----------------------------------------------

    try:

        for column in FEATURES:

            batch_input[column] = pd.to_numeric(
                batch_input[column],
                errors="raise"
            )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Model input columns must "
                "contain numeric values."
            )
        ) from error


    # ----------------------------------------------
    # Validate missing values
    # ----------------------------------------------

    required_non_null = [

        column

        for column in FEATURES

        if column != "nps"

    ]


    if batch_input[
        required_non_null
    ].isna().any().any():

        raise HTTPException(
            status_code=400,
            detail=(
                "Missing values are only "
                "allowed in the nps column."
            )
        )


    # ----------------------------------------------
    # Validate negative values
    # ----------------------------------------------

    if (
        (batch_input["total_sales"] < 0).any()
        or
        (
            batch_input[
                "unique_products"
            ] < 0
        ).any()
        or
        (
            batch_input[
                "number_of_invoices"
            ] < 0
        ).any()
        or
        (
            batch_input[
                "n_comp"
            ] < 0
        ).any()
        or
        (
            batch_input[
                "n_communications"
            ] < 0
        ).any()
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Negative values are not allowed."
            )
        )


    # ----------------------------------------------
    # Validate NPS
    # ----------------------------------------------

    valid_nps = (

        batch_input[
            "nps"
        ].isna()

        |

        batch_input[
            "nps"
        ].between(
            0,
            10
        )

    )


    if not valid_nps.all():

        raise HTTPException(
            status_code=400,
            detail=(
                "NPS must be between 0 and 10."
            )
        )


    # ----------------------------------------------
    # Validate loyalty
    # ----------------------------------------------

    if not batch_input[
        "loyalty"
    ].isin(
        [0, 1]
    ).all():

        raise HTTPException(
            status_code=400,
            detail=(
                "Loyalty must contain "
                "only 0 or 1."
            )
        )


    # ----------------------------------------------
    # Generate predictions
    # ----------------------------------------------

    probabilities = model.predict_proba(
        batch_input
    )[:, 1]


    predictions = model.predict(
        batch_input
    )


    # ----------------------------------------------
    # Build response
    # ----------------------------------------------

    results = []


    for index, (
        probability,
        prediction
    ) in enumerate(
        zip(
            probabilities,
            predictions
        )
    ):


        customer = batch_input.iloc[
            index
        ]


        prediction_label = (

            "Likely to respond"

            if prediction == 1

            else "Unlikely to respond"

        )


        nps_value = (

            None

            if pd.isna(
                customer["nps"]
            )

            else float(
                customer["nps"]
            )

        )


        results.append({

            "row":
                index + 1,

            "total_sales":
                float(
                    customer[
                        "total_sales"
                    ]
                ),

            "unique_products":
                int(
                    customer[
                        "unique_products"
                    ]
                ),

            "number_of_invoices":
                int(
                    customer[
                        "number_of_invoices"
                    ]
                ),

            "nps":
                nps_value,

            "n_comp":
                int(
                    customer[
                        "n_comp"
                    ]
                ),

            "n_communications":
                int(
                    customer[
                        "n_communications"
                    ]
                ),

            "loyalty":
                int(
                    customer[
                        "loyalty"
                    ]
                ),

            "response_probability":
                round(
                    float(
                        probability
                    ),
                    4
                ),

            "response_probability_percent":
                round(
                    float(
                        probability
                    ) * 100,
                    2
                ),

            "predicted_response":
                int(
                    prediction
                ),

            "prediction_label":
                prediction_label,
        })


    return {

        "rows_processed":
            len(
                batch_input
            ),

        "predictions":
            results,

    }