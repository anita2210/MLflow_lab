import requests
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────
# Only run this AFTER the model server is running
# in a separate terminal via the command from serving.py
# ─────────────────────────────────────────

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

url = "http://127.0.0.1:5002/invocations"

payload = {
    "dataframe_records": X_test.head(10).to_dict(orient="records")
}

print("Sending request to model server...")
response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

print(f"Status Code : {response.status_code}")
print(f"Predictions : {response.json()}")
print(f"Actual      : {list(y_test[:10])}")