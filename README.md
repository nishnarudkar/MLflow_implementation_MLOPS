# Breast Cancer Classification with MLflow

A machine learning project that compares multiple classification models on the breast cancer dataset using MLflow for experiment tracking.

## Overview

This project trains and evaluates four different classification models on the Wisconsin Breast Cancer dataset:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

All experiments are tracked using MLflow, allowing easy comparison of model performance.

## Requirements

- Python 3.7+
- MLflow
- scikit-learn
- pandas
- matplotlib

## Installation

1. Clone this repository or download the files

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the MLflow tracking server (optional, for UI access):
```bash
mlflow server --host 127.0.0.1 --port 5000
```

2. Run the training script:
```bash
python main.py
```

3. View results in the MLflow UI:
```bash
mlflow ui
```
Then navigate to `http://127.0.0.1:5000` in your browser.

## Project Structure

```
.
├── main.py              # Main training script
├── requirements.txt     # Project dependencies
├── mlflow.db           # MLflow tracking database
└── mlruns/             # MLflow experiment artifacts and logs
```

## Features

- Automated training of multiple models
- Experiment tracking with MLflow
- Metrics logging (train and test accuracy)
- Model versioning and artifact storage
- Easy model comparison through MLflow UI

## Results

The script logs the following for each model:
- Model name and parameters
- Training accuracy
- Test accuracy
- Serialized model artifacts

Access the MLflow UI to compare model performance and select the best model for deployment.
