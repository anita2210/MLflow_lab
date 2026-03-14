import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────
# STEP 1: Find the Best Run from experiment.py
# Looks inside breast_cancer_classification experiment
# and picks the run with the highest roc_auc
# ─────────────────────────────────────────
print("=== STEP 1: Finding Best Run ===")

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("breast_cancer_classification")

if experiment is None:
    print("ERROR: Experiment not found!")
    print("Make sure you ran experiment.py first.")
    exit()

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"],
    max_results=5
)

print("Top runs by AUC:")
for r in runs:
    print(f"  {r.data.params.get('model', 'unknown'):20s} | AUC: {r.data.metrics.get('roc_auc', 0):.4f} | run_id: {r.info.run_id}")

best_run = runs[0]
best_run_id = best_run.info.run_id
best_model_name = best_run.data.params.get("model", "unknown")
print(f"\nBest model : {best_model_name}")
print(f"Run ID     : {best_run_id}")


# ─────────────────────────────────────────
# STEP 2: Load the Best Model Directly
# No server needed — load and predict in Python
# ─────────────────────────────────────────
print("\n=== STEP 2: Load & Test Model Directly ===")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_uri = f"runs:/{best_run_id}/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

predictions = loaded_model.predict(X_test)
print(f"Model     : {best_model_name}")
print(f"Accuracy  : {accuracy_score(y_test, predictions):.4f}")
print(f"Predictions (first 10): {list(predictions[:10])}")
print(f"Actual     (first 10): {list(y_test[:10])}")
print("Model loaded and tested successfully ✓")


# ─────────────────────────────────────────
# STEP 3: Print the Serve Command
# Copy the command below and run it in a
# SEPARATE terminal to start the model server
# ─────────────────────────────────────────
print("\n=== STEP 3: Serve the Model ===")
print("Run this command in a SEPARATE terminal:")
print(f"\n  mlflow models serve --env-manager=local -m runs:/{best_run_id}/model -h 127.0.0.1 -p 5001\n")
print("Then run serving_request.py to send predictions via HTTP")