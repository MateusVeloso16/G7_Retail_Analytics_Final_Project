// --------------------------------------------------
// Single Customer Prediction
// --------------------------------------------------

const form = document.getElementById(
    "prediction-form"
);

const resultDiv = document.getElementById(
    "result"
);


form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();


        const npsValue = document.getElementById(
            "nps"
        ).value;


        const customerData = {

            total_sales: parseFloat(
                document.getElementById(
                    "total_sales"
                ).value
            ),

            unique_products: parseInt(
                document.getElementById(
                    "unique_products"
                ).value
            ),

            number_of_invoices: parseInt(
                document.getElementById(
                    "number_of_invoices"
                ).value
            ),

            nps: npsValue === ""
                ? null
                : parseFloat(npsValue),

            n_comp: parseInt(
                document.getElementById(
                    "n_comp"
                ).value
            ),

            n_communications: parseInt(
                document.getElementById(
                    "n_communications"
                ).value
            ),

            loyalty: parseInt(
                document.getElementById(
                    "loyalty"
                ).value
            )
        };


        try {

            const response = await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify(
                        customerData
                    )
                }
            );


            const data = await response.json();


            if (!response.ok) {

                throw new Error(
                    "Prediction request failed."
                );
            }


            const isPositive =
                data.predicted_response === 1;


            const labelClass = isPositive
                ? "prediction-positive"
                : "prediction-negative";


            const barClass = isPositive
                ? "probability-positive"
                : "probability-negative";


            resultDiv.innerHTML = `
                <h2>Prediction Result</h2>

                <div
                    class="prediction-label ${labelClass}"
                >
                    ${data.prediction_label}
                </div>

                <div class="probability-text">
                    Response Probability:
                    <strong>
                        ${data.response_probability_percent}%
                    </strong>
                </div>

                <div class="probability-bar">

                    <div
                        class="probability-fill ${barClass}"
                        style="
                            width:
                            ${data.response_probability_percent}%;
                        "
                    >
                    </div>

                </div>
            `;

        }

        catch (error) {

            resultDiv.innerHTML = `
                <h2>Prediction Result</h2>

                <p>
                    An error occurred while
                    generating the prediction.
                </p>
            `;
        }
    }
);


// --------------------------------------------------
// Batch Prediction
// --------------------------------------------------

const batchForm = document.getElementById(
    "batch-form"
);

const batchFile = document.getElementById(
    "batch-file"
);

const batchResult = document.getElementById(
    "batch-result"
);


let latestBatchPredictions = [];


batchForm.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();


        if (batchFile.files.length === 0) {

            batchResult.innerHTML = `
                <p>
                    Please select a CSV file.
                </p>
            `;

            return;
        }


        const file = batchFile.files[0];

        const formData = new FormData();

        formData.append(
            "file",
            file
        );


        batchResult.innerHTML = `
            <p>
                Processing CSV file...
            </p>
        `;


        try {

            const response = await fetch(
                "/predict-batch",
                {
                    method: "POST",
                    body: formData
                }
            );


            const data = await response.json();


            if (!response.ok) {

                let errorMessage =
                    "Batch prediction failed.";

                if (
                    typeof data.detail === "string"
                ) {

                    errorMessage = data.detail;

                } else if (
                    data.detail
                    &&
                    data.detail.message
                ) {

                    errorMessage =
                        data.detail.message;

                    if (
                        data.detail.missing_columns
                    ) {

                        errorMessage +=
                            " Missing columns: "
                            +
                            data.detail
                                .missing_columns
                                .join(", ");
                    }
                }


                throw new Error(
                    errorMessage
                );
            }


            latestBatchPredictions =
                data.predictions;


            let tableRows = "";


            data.predictions.forEach(
                function(prediction) {

                    const labelClass =
                        prediction.predicted_response === 1
                            ? "prediction-positive"
                            : "prediction-negative";


                    tableRows += `
                        <tr>

                            <td>
                                ${prediction.row}
                            </td>

                            <td>
                                ${prediction.response_probability_percent}%
                            </td>

                            <td class="${labelClass}">
                                ${prediction.prediction_label}
                            </td>

                        </tr>
                    `;
                }
            );


            batchResult.innerHTML = `
                <h3>
                    Batch Prediction Results
                </h3>

                <p>
                    Rows processed:
                    <strong>
                        ${data.rows_processed}
                    </strong>
                </p>

                <table>

                    <thead>

                        <tr>
                            <th>Row</th>
                            <th>Probability</th>
                            <th>Prediction</th>
                        </tr>

                    </thead>

                    <tbody>
                        ${tableRows}
                    </tbody>

                </table>

                <button
                    type="button"
                    id="download-predictions"
                >
                    Download Predictions
                </button>
            `;


            const downloadButton =
                document.getElementById(
                    "download-predictions"
                );


            downloadButton.addEventListener(
                "click",
                downloadPredictions
            );

        }

        catch (error) {

            batchResult.innerHTML = `
                <p class="prediction-negative">
                    ${error.message}
                </p>
            `;
        }
    }
);


// --------------------------------------------------
// Download Batch Predictions
// --------------------------------------------------

function downloadPredictions() {

    if (
        latestBatchPredictions.length === 0
    ) {
        return;
    }


    const csvRows = [

        [
            "row",
            "total_sales",
            "unique_products",
            "number_of_invoices",
            "nps",
            "n_comp",
            "n_communications",
            "loyalty",
            "response_probability",
            "response_probability_percent",
            "predicted_response",
            "prediction_label"
        ].join(",")

    ];


    latestBatchPredictions.forEach(
        function(prediction) {

            const npsValue =
                prediction.nps === null
                    ? ""
                    : prediction.nps;


            csvRows.push(

                [
                    prediction.row,
                    prediction.total_sales,
                    prediction.unique_products,
                    prediction.number_of_invoices,
                    npsValue,
                    prediction.n_comp,
                    prediction.n_communications,
                    prediction.loyalty,
                    prediction.response_probability,
                    prediction.response_probability_percent,
                    prediction.predicted_response,
                    `"${prediction.prediction_label}"`
                ].join(",")

            );
        }
    );


    const csvContent =
        csvRows.join("\n");


    const blob = new Blob(
        [csvContent],
        {
            type: "text/csv;charset=utf-8;"
        }
    );


    const downloadUrl =
        URL.createObjectURL(
            blob
        );


    const link =
        document.createElement(
            "a"
        );


    link.href = downloadUrl;

    link.download =
        "batch_predictions.csv";


    document.body.appendChild(
        link
    );


    link.click();


    document.body.removeChild(
        link
    );


    URL.revokeObjectURL(
        downloadUrl
    );
}