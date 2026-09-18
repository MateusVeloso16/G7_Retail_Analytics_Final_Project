from pathlib import Path

import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier


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
# Paths
# --------------------------------------------------

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset.csv"
)

MLFLOW_DB = (
    PROJECT_ROOT
    / "mlflow.db"
)


# --------------------------------------------------
# MLflow configuration
# --------------------------------------------------

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment(
    "Campaign Response Model Comparison"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

master_df = pd.read_csv(
    DATA_FILE
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


X = master_df[
    FEATURES
].copy()

y = master_df[
    "response"
].copy()


# --------------------------------------------------
# Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)


# --------------------------------------------------
# Median imputation
# --------------------------------------------------

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(
    X_train
)

X_test_imputed = imputer.transform(
    X_test
)


X_train_imputed = pd.DataFrame(
    X_train_imputed,
    columns=FEATURES,
    index=X_train.index,
)

X_test_imputed = pd.DataFrame(
    X_test_imputed,
    columns=FEATURES,
    index=X_test.index,
)


# --------------------------------------------------
# Standardised version for SVM
# --------------------------------------------------

svm_scaler = StandardScaler()

X_train_svm = svm_scaler.fit_transform(
    X_train_imputed
)

X_test_svm = svm_scaler.transform(
    X_test_imputed
)


X_train_svm = pd.DataFrame(
    X_train_svm,
    columns=FEATURES,
    index=X_train.index,
)

X_test_svm = pd.DataFrame(
    X_test_svm,
    columns=FEATURES,
    index=X_test.index,
)


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_model(
    model,
    X_train_data,
    X_test_data,
):

    train_probabilities = (
        model.predict_proba(
            X_train_data
        )[:, 1]
    )

    test_probabilities = (
        model.predict_proba(
            X_test_data
        )[:, 1]
    )


    train_auc = roc_auc_score(
        y_train,
        train_probabilities,
    )

    test_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )


    test_predictions = (
        test_probabilities >= 0.50
    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
    ).ravel()


    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn)
    )

    sensitivity = (
        tp
        /
        (tp + fn)
    )

    specificity = (
        tn
        /
        (tn + fp)
    )

    balanced_accuracy = (
        sensitivity
        +
        specificity
    ) / 2


    return {
        "train_roc_auc": train_auc,
        "test_roc_auc": test_auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


# --------------------------------------------------
# Model configurations
# --------------------------------------------------

models = {

    "Gaussian Naive Bayes": {

        "model": GaussianNB(),

        "train_data":
            X_train_imputed,

        "test_data":
            X_test_imputed,

        "params": {
            "model_type":
                "GaussianNB",
        },
    },


    "Random Forest": {

        "model": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),

        "train_data":
            X_train_imputed,

        "test_data":
            X_test_imputed,

        "params": {
            "model_type":
                "RandomForestClassifier",

            "n_estimators":
                300,

            "random_state":
                42,
        },
    },


    "Support Vector Machine": {

        "model": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
        ),

        "train_data":
            X_train_svm,

        "test_data":
            X_test_svm,

        "params": {
            "model_type":
                "SVC",

            "kernel":
                "rbf",

            "probability":
                True,

            "random_state":
                42,
        },
    },


    "XGBoost": {

        "model": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),

        "train_data":
            X_train_imputed,

        "test_data":
            X_test_imputed,

        "params": {
            "model_type":
                "XGBClassifier",

            "n_estimators":
                300,

            "max_depth":
                3,

            "learning_rate":
                0.05,

            "subsample":
                0.8,

            "colsample_bytree":
                0.8,

            "eval_metric":
                "logloss",

            "random_state":
                42,
        },
    },
}


# --------------------------------------------------
# Train models and record experiments
# --------------------------------------------------

results = []


for model_name, configuration in models.items():

    print(
        f"\nTraining {model_name}..."
    )


    model = configuration[
        "model"
    ]

    train_data = configuration[
        "train_data"
    ]

    test_data = configuration[
        "test_data"
    ]


    model.fit(
        train_data,
        y_train,
    )


    metrics = evaluate_model(
        model,
        train_data,
        test_data,
    )


    with mlflow.start_run(
        run_name=model_name
    ):

        mlflow.log_params(
            configuration[
                "params"
            ]
        )


        mlflow.log_params({
            "test_size":
                0.20,

            "split_random_state":
                42,

            "training_rows":
                len(X_train),

            "test_rows":
                len(X_test),

            "number_of_features":
                len(FEATURES),

            "threshold":
                0.50,
        })


        mlflow.log_metrics(
            metrics
        )


        mlflow.set_tag(
            "project",
            "G7 Retail Analytics Final Project",
        )

        mlflow.set_tag(
            "task",
            "Campaign Response Classification",
        )

        mlflow.set_tag(
            "selected_for_deployment",
            str(
                model_name
                == "Random Forest"
            ),
        )


    results.append({

        "model":
            model_name,

        **metrics,
    })


# --------------------------------------------------
# Show comparison
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="test_roc_auc",
    ascending=False,
)


print(
    "\nModel comparison:"
)

print(
    results_df.round(4).to_string(
        index=False
    )
)


print(
    "\nMLflow tracking database:"
)

print(
    MLFLOW_DB
)