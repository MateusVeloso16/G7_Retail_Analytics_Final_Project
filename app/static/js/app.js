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