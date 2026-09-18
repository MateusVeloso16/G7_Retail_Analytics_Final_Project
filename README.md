# Retail Analytics – Final Project

![API Tests](https://github.com/MateusVeloso16/G7_Retail_Analytics_Final_Project/actions/workflows/ci.yml/badge.svg)

End-to-end retail analytics project combining data analysis, customer segmentation, machine learning, sentiment analysis and model deployment.

The project analyses retail customer behaviour, campaign response and customer feedback, then deploys the final campaign response model through a FastAPI web application.

---

## Project Objectives

The main objectives of the project are:

- Analyse overall retail sales performance.
- Understand customer purchasing behaviour.
- Segment customers using clustering techniques.
- Analyse factors associated with marketing campaign response.
- Compare traditional statistical modelling with machine learning models.
- Analyse customer feedback using NLP and sentiment analysis.
- Deploy the final predictive model through a web application.
- Support both individual and batch customer predictions.

---

## Project Workflow

The project was developed in the following stages:

1. Data Management
2. Exploratory Data Analysis
3. Customer Segmentation
4. Logistic Regression
5. Machine Learning Model Comparison
6. Sentiment Analysis
7. Model Deployment
8. API Testing
9. Docker Containerisation
10. Continuous Integration
11. MLflow Experiment Tracking

---

## Data Management

Three datasets were used:

- Online Retail Sales Data
- Campaign Response Data
- Online Retail Customer Feedback

The datasets were cleaned and combined to create a customer-level analytical dataset.

Main customer features include:

- Total sales
- Unique products purchased
- Number of invoices
- Purchase activity
- Cancellation activity
- Net Promoter Score (NPS)
- Number of complaints
- Number of communications
- Loyalty status
- Campaign response

The final customer dataset contains **3,834 customers**.

The overall campaign response rate was approximately **40.56%**.

---

## Exploratory Data Analysis

EDA was used to investigate relationships between customer characteristics and campaign response.

Some of the main findings were:

- NPS showed the strongest relationship with campaign response.
- Loyalty customers had higher campaign response rates.
- Customers receiving more communications generally showed higher response rates.
- Purchase behaviour variables showed relatively weak relationships with campaign response.
- Sales increased strongly during the later months of the year, particularly September to November.

---

## Customer Segmentation

Customer segmentation was performed using **K-Means clustering** based on RFM-style customer behaviour.

Before clustering:

- Monetary variables were transformed using `log1p`.
- Features were standardised using `StandardScaler`.
- Multiple values of K were evaluated.

Although K=2 produced the highest silhouette score, **K=4** was selected because it provided more commercially useful customer segments.

The final segments were:

| Segment | Customers | Percentage |
|---|---:|---:|
| Dormant / Low-Value | 1,407 | 36.91% |
| Regular Customers | 1,070 | 28.07% |
| Recent / Developing | 719 | 18.86% |
| High-Value Customers | 616 | 16.16% |

---

## Logistic Regression

Binary Logistic Regression was used as the traditional statistical model.

Features used:

- Total Sales
- Unique Products
- Number of Invoices
- NPS
- Number of Complaints
- Number of Communications
- Loyalty

The model showed that:

- NPS
- Complaints
- Communications
- Loyalty

were statistically significant predictors of campaign response.

Purchase-related variables were considerably weaker predictors.

The Logistic Regression test ROC-AUC was approximately:

**0.64**

---

## Machine Learning Models

Four machine learning algorithms were compared:

| Model | Test ROC-AUC |
|---|---:|
| Gaussian Naive Bayes | 0.6551 |
| Random Forest | **0.7249** |
| Support Vector Machine | 0.6620 |
| XGBoost | 0.6867 |

The **Random Forest** model produced the strongest test ROC-AUC and was selected as the final campaign response model.

The most influential features were mainly:

- NPS
- Number of Communications
- Number of Complaints
- Loyalty

These findings were consistent with the Logistic Regression analysis.

---

## Sentiment Analysis

Customer feedback was analysed using a pretrained Transformer model:

`cardiffnlp/twitter-roberta-base-sentiment-latest`

After cleaning, **964 customer feedback comments** were analysed.

Overall sentiment distribution:

| Sentiment | Comments |
|---|---:|
| Positive | 456 |
| Negative | 455 |
| Neutral | 53 |

Because many comments contained mixed opinions, the feedback was also split into smaller clauses.

A total of **2,160 clauses** were analysed.

Main themes included:

- Delivery
- Customer Support
- App
- Checkout / Ordering
- Billing / Payment
- Coupon / Promotion

The analysis identified particularly strong negative sentiment around:

- App performance
- Checkout problems
- Delivery delays
- Coupon application issues

The feedback dataset was also highly repetitive, so duplicated phrases were considered when interpreting the results.

---

## Final Predictive Model

The final production model is a Random Forest pipeline containing:

- Median imputation for missing NPS values
- Random Forest classifier with 300 trees

The model was trained using all available customer records after model selection and validation.

The trained model is stored in:

```text
models/campaign_response_model.joblib
```

---

## Web Application

The model is deployed using **FastAPI** with a custom HTML, CSS and JavaScript interface.

The application supports two prediction modes.

### Single Prediction

A user can manually enter customer information and receive:

- Campaign response probability
- Predicted response class
- Human-readable prediction label

### Batch Prediction

A CSV file containing multiple customers can be uploaded.

The application:

- Validates the input file
- Generates predictions for all customers
- Displays the results in a table
- Allows the prediction results to be downloaded as CSV

---

## API Endpoints

### Home

```text
GET /
```

Loads the web application.

### Single Prediction

```text
POST /predict
```

Generates a prediction for one customer.

### Batch Prediction

```text
POST /predict-batch
```

Generates predictions for multiple customers from a CSV file.

---

## Technologies Used

### Data Analysis

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Statistics and Machine Learning

- Scikit-learn
- Statsmodels
- XGBoost
- K-Means
- Logistic Regression
- Random Forest
- Support Vector Machine
- Naive Bayes

### NLP

- PyTorch
- Hugging Face Transformers
- RoBERTa

### Deployment

- FastAPI
- Uvicorn
- Jinja2
- HTML
- CSS
- JavaScript

### Engineering

- Git
- GitHub
- Docker
- GitHub Actions
- Pytest
- uv
- MLflow

---

## Project Structure

```text
G7_Retail_Analytics_Final_Project/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   └── campaign_response_model.joblib
│
├── notebooks/
│   ├── 01_data_management.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_customer_segmentation.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_machine_learning_models.ipynb
│   └── 06_sentiment_analysis.ipynb
│
├── src/
│   └── g7_retail_analytics_final_project/
│       └── models/
│           ├── train_model.py
│           └── mlflow_experiments.py
│
├── tests/
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/MateusVeloso16/G7_Retail_Analytics_Final_Project.git
```

Enter the project directory:

```bash
cd G7_Retail_Analytics_Final_Project
```

Install the dependencies:

```bash
uv sync
```

Run the FastAPI application:

```bash
uv run uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

## Running the Tests

Run the automated API tests with:

```bash
uv run python -m pytest tests/test_api.py -v
```

The test suite validates:

- Home page
- Single prediction
- Input validation
- Batch prediction
- Missing CSV columns

---

## Docker

Build the Docker image:

```bash
docker build -t retail-campaign-analytics .
```

Run the container:

```bash
docker run --rm -p 8000:8000 retail-campaign-analytics
```

Then open:

```text
http://127.0.0.1:8000
```

---

## Continuous Integration

GitHub Actions is configured to automatically run the API test suite whenever:

- Code is pushed to the `main` branch.
- A pull request is opened against `main`.

This helps ensure that changes do not break the prediction application.

The workflow is defined in:

```text
.github/workflows/ci.yml
```

---

## Experiment Tracking

**MLflow** is used to track and compare the machine learning experiments.

The following models are registered as separate MLflow runs:

- Gaussian Naive Bayes
- Random Forest
- Support Vector Machine
- XGBoost

For each model, MLflow records:

- Model parameters
- Training ROC-AUC
- Test ROC-AUC
- Accuracy
- Sensitivity
- Specificity
- Balanced Accuracy

The experiment reproduced the results obtained during the machine learning analysis, with Random Forest achieving the highest test ROC-AUC of **0.7249**.

Run the experiment tracking script with:

```bash
uv run python src/g7_retail_analytics_final_project/models/mlflow_experiments.py
```

Start the MLflow interface with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

The local MLflow database is excluded from Git version control through `.gitignore`.

---

## Author

**Mateus Veloso dos Santos**

Retail Analytics Final Project