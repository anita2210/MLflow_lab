import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ─────────────────────────────────────────
# STEP 1: Autolog — simplest way to use MLflow
# Just call mlflow.autolog() before training
# MLflow will automatically log params, metrics, and model
# ─────────────────────────────────────────
print("=== PART 1: Autologging ===")

mlflow.autolog()

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("Autolog done — check MLflow UI, everything was logged automatically!")


# ─────────────────────────────────────────
# STEP 2: Manual logging — log exactly what you want
# ─────────────────────────────────────────
print("\n=== PART 2: Manual Logging ===")

mlflow.autolog(disable=True)  # turn off autolog for manual control
mlflow.set_experiment("starter_manual")

with mlflow.start_run() as run:
    rf2 = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    rf2.fit(X_train, y_train)
    preds = rf2.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)

    # Log params
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 4)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model
    signature = infer_signature(X_test, preds)
    mlflow.sklearn.log_model(rf2, "model", signature=signature)

    print(f"Run ID  : {run.info.run_id}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Manual logging done - OK")


# ─────────────────────────────────────────
# STEP 3: Reload a saved model
# ─────────────────────────────────────────
print("\n=== PART 3: Reload Saved Model ===")

run_id = run.info.run_id
loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
reloaded_preds = loaded_model.predict(X_test)

print(f"Reloaded model accuracy: {accuracy_score(y_test, reloaded_preds):.4f}")
print("Model reloaded successfully - OK")

print("\nDone! Now run: mlflow ui --port=5001")