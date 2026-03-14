import logging
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
np.random.seed(42)


# STEP 1: Load Data

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Data loaded successfully")
print(f"  Train samples : {len(X_train)}")
print(f"  Test  samples : {len(X_test)}")
print(f"  Features      : {X.shape[1]}")



# STEP 2: Helper Functions

def eval_metrics(actual, pred, pred_proba=None):
    acc = accuracy_score(actual, pred)
    f1  = f1_score(actual, pred)
    auc = roc_auc_score(actual, pred_proba) if pred_proba is not None else None
    return acc, f1, auc


def log_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title(f"Top 10 Feature Importances — {model_name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()

        mlflow.log_figure(fig, f"feature_importance_{model_name}.png")
        plt.close(fig)


# STEP 3: Set Experiment

mlflow.set_experiment("breast_cancer_classification")



# STEP 4: Train & Log — XGBoost

print("\n Training XGBoost ")

xgb_params = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "random_state": 42,
}

with mlflow.start_run(run_name="XGBoost"):
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train)

    preds = model_xgb.predict(X_test)
    proba = model_xgb.predict_proba(X_test)[:, 1]
    acc, f1, auc = eval_metrics(y_test, preds, proba)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}")

    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", xgb_params["n_estimators"])
    mlflow.log_param("max_depth", xgb_params["max_depth"])
    mlflow.log_param("learning_rate", xgb_params["learning_rate"])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    log_feature_importance(model_xgb, list(data.feature_names), "XGBoost")

    signature = infer_signature(X_train, model_xgb.predict(X_train))
    tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
    if tracking_url_type != "file":
        mlflow.xgboost.log_model(model_xgb, "model", registered_model_name="XGBoost_BreastCancer", signature=signature)
    else:
        mlflow.xgboost.log_model(model_xgb, "model", signature=signature)

    print("  Logged to MLflow ")



# STEP 5: Train & Log — Random Forest

print("\n Training Random Forest ")

rf_params = {
    "n_estimators": 100,
    "max_depth": 6,
    "max_features": "sqrt",
    "random_state": 42,
}

with mlflow.start_run(run_name="RandomForest"):
    model_rf = RandomForestClassifier(**rf_params)
    model_rf.fit(X_train, y_train)

    preds = model_rf.predict(X_test)
    proba = model_rf.predict_proba(X_test)[:, 1]
    acc, f1, auc = eval_metrics(y_test, preds, proba)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}")

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", rf_params["n_estimators"])
    mlflow.log_param("max_depth", rf_params["max_depth"])
    mlflow.log_param("max_features", rf_params["max_features"])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    log_feature_importance(model_rf, list(data.feature_names), "RandomForest")

    signature = infer_signature(X_train, model_rf.predict(X_train))
    tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
    if tracking_url_type != "file":
        mlflow.sklearn.log_model(model_rf, "model", registered_model_name="RF_BreastCancer", signature=signature)
    else:
        mlflow.sklearn.log_model(model_rf, "model", signature=signature)

    print("  Logged to MLflow ✓")



# STEP 6: Train & Log — Gradient Boosting

print("\n Training Gradient Boosting")

gb_params = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
}

with mlflow.start_run(run_name="GradientBoosting"):
    model_gb = GradientBoostingClassifier(**gb_params)
    model_gb.fit(X_train, y_train)

    preds = model_gb.predict(X_test)
    proba = model_gb.predict_proba(X_test)[:, 1]
    acc, f1, auc = eval_metrics(y_test, preds, proba)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}")

    mlflow.log_param("model", "GradientBoosting")
    mlflow.log_param("n_estimators", gb_params["n_estimators"])
    mlflow.log_param("max_depth", gb_params["max_depth"])
    mlflow.log_param("learning_rate", gb_params["learning_rate"])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    log_feature_importance(model_gb, list(data.feature_names), "GradientBoosting")

    signature = infer_signature(X_train, model_gb.predict(X_train))
    tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
    if tracking_url_type != "file":
        mlflow.sklearn.log_model(model_gb, "model", registered_model_name="GB_BreastCancer", signature=signature)
    else:
        mlflow.sklearn.log_model(model_gb, "model", signature=signature)

    print("  Logged to MLflow ✓")

# STEP 7: Compare All Models

results = []
for name, model in [("XGBoost", model_xgb), ("RandomForest", model_rf), ("GradientBoosting", model_gb)]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc, f1, auc = eval_metrics(y_test, preds, proba)
    results.append({"Model": name, "Accuracy": round(acc, 4), "F1": round(f1, 4), "AUC": round(auc, 4)})

df_results = pd.DataFrame(results).sort_values("AUC", ascending=False)
print(df_results.to_string(index=False))
print(f"\nBest model by AUC: {df_results.iloc[0]['Model']}")
print("\nDone! Now run: mlflow ui --port=5001")