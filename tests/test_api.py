from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(
    app
)


def test_home_page():

    response = client.get(
        "/"
    )

    assert response.status_code == 200

    assert (
        "Retail Campaign Analytics"
        in response.text
    )


def test_single_prediction():

    customer_data = {
        "total_sales": 1200,
        "unique_products": 35,
        "number_of_invoices": 4,
        "nps": 8,
        "n_comp": 1,
        "n_communications": 6,
        "loyalty": 1,
    }

    response = client.post(
        "/predict",
        json=customer_data
    )

    assert response.status_code == 200

    data = response.json()

    assert "response_probability" in data
    assert "response_probability_percent" in data
    assert "predicted_response" in data
    assert "prediction_label" in data

    assert (
        0
        <= data["response_probability"]
        <= 1
    )

    assert (
        0
        <= data["response_probability_percent"]
        <= 100
    )

    assert data["predicted_response"] in [
        0,
        1
    ]

    assert data["prediction_label"] in [
        "Likely to respond",
        "Unlikely to respond",
    ]


def test_invalid_single_prediction_nps():

    customer_data = {
        "total_sales": 1200,
        "unique_products": 35,
        "number_of_invoices": 4,
        "nps": 15,
        "n_comp": 1,
        "n_communications": 6,
        "loyalty": 1,
    }

    response = client.post(
        "/predict",
        json=customer_data
    )

    assert response.status_code == 422


def test_batch_prediction():

    csv_content = (
        "total_sales,unique_products,"
        "number_of_invoices,nps,n_comp,"
        "n_communications,loyalty\n"
        "1200,35,4,8,1,6,1\n"
        "1200,35,4,10,0,8,1\n"
        "700,20,2,,1,4,0\n"
    )

    csv_file = BytesIO(
        csv_content.encode("utf-8")
    )

    response = client.post(
        "/predict-batch",
        files={
            "file": (
                "customers.csv",
                csv_file,
                "text/csv"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["rows_processed"] == 3

    assert len(
        data["predictions"]
    ) == 3

    for prediction in data["predictions"]:

        assert (
            0
            <= prediction[
                "response_probability"
            ]
            <= 1
        )

        assert (
            prediction[
                "predicted_response"
            ]
            in [0, 1]
        )

        assert (
            prediction[
                "prediction_label"
            ]
            in [
                "Likely to respond",
                "Unlikely to respond",
            ]
        )


def test_batch_missing_required_column():

    csv_content = (
        "total_sales,unique_products,"
        "number_of_invoices,nps,n_comp,"
        "n_communications\n"
        "1200,35,4,8,1,6\n"
    )

    csv_file = BytesIO(
        csv_content.encode("utf-8")
    )

    response = client.post(
        "/predict-batch",
        files={
            "file": (
                "customers.csv",
                csv_file,
                "text/csv"
            )
        }
    )

    assert response.status_code == 400

    data = response.json()

    assert (
        "loyalty"
        in data["detail"]["missing_columns"]
    )