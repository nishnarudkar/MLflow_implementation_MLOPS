import mlflow
import mlflow.sklearn
import dagshub

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import matplotlib.pyplot as plt

dagshub.init(
    repo_owner="nishnarudkar",
    repo_name="MLflow_implementation_MLOPS",
    mlflow=True
)

mlflow.set_experiment("Breast_Cancer_Model_Comparison")

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

results = []

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        
        model.fit(X_train, y_train)

        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        
        mlflow.log_param("model", model_name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        
        mlflow.sklearn.log_model(model, artifact_path=model_name)

        
        results.append({
            "model": model_name,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
        })


results_df = pd.DataFrame(results)

plt.figure(figsize=(8,5))
plt.bar(results_df["model"], results_df["test_accuracy"])

plt.title("Model Comparison (Test Accuracy)")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("model_comparison.png")

with mlflow.start_run(run_name="Model_Comparison_Chart"):
    mlflow.log_artifact("model_comparison.png")

print("All models logged successfully!")