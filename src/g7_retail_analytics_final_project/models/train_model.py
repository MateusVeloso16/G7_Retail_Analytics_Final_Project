from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score




current_file = Path(__file__).resolve()

for path in current_file.parents:
    if (path / "pyproject.toml").exists():
        PROJECT_ROOT = path
        break
else:
    raise FileNotFoundError(
        "Project root could not be found."
    )



DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)



df = pd.read_csv(DATA_FILE)


print("Project root:", PROJECT_ROOT)
print("Dataset shape:", df.shape)
print()
print("Dataset columns:")
print(df.columns.tolist())




FEATURES = [
    "total_sales",
    "unique_products",
    "number_of_invoices",
    "nps",
    "n_comp",
    "n_communications",
    "loyalty",
]

TARGET = "response"


X = df[FEATURES].copy()
y = df[TARGET].copy()


print()
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

print()
print("Missing values by feature:")
print(X.isna().sum())

print()
print("Target distribution:")
print(y.value_counts())




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)



model_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)



model_pipeline.fit(
    X_train,
    y_train,
)



test_probabilities = model_pipeline.predict_proba(
    X_test
)[:, 1]

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)


print()
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

print(
    "Training response rate:",
    round(y_train.mean() * 100, 2),
    "%"
)

print(
    "Test response rate:",
    round(y_test.mean() * 100, 2),
    "%"
)

print()
print(
    "Test ROC-AUC:",
    round(test_auc, 4)
)




final_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


final_pipeline.fit(
    X,
    y,
)



MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



MODEL_FILE = (
    MODEL_DIR
    / "campaign_response_model.joblib"
)

joblib.dump(
    final_pipeline,
    MODEL_FILE,
    compress=3,
)


print()
print(
    "Final model trained using:",
    len(X),
    "customers"
)

print(
    "Model saved to:",
    MODEL_FILE
)



loaded_model = joblib.load(
    MODEL_FILE
)


sample_customer = pd.DataFrame(
    [
        {
            "total_sales": 1200.0,
            "unique_products": 35,
            "number_of_invoices": 4,
            "nps": 8,
            "n_comp": 1,
            "n_communications": 6,
            "loyalty": 1,
        }
    ]
)


sample_probability = loaded_model.predict_proba(
    sample_customer
)[0, 1]

sample_prediction = loaded_model.predict(
    sample_customer
)[0]


print()
print(
    "Sample response probability:",
    round(sample_probability, 4)
)

print(
    "Sample predicted class:",
    int(sample_prediction)
)