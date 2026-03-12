# MLflow Implementation - MLOps with DagsHub

A comprehensive machine learning project demonstrating MLflow and DagsHub integration for experiment tracking, model comparison, and MLOps best practices using the Wisconsin Breast Cancer dataset.

## Overview

This project showcases modern MLOps practices by training and comparing multiple classification models with full experiment tracking capabilities. It demonstrates how to use MLflow with DagsHub for managing the complete ML lifecycle with cloud-based experiment tracking, visualization, and collaboration.

### Models Implemented
- **Logistic Regression** - Baseline linear model
- **Random Forest** - Ensemble tree-based classifier
- **Support Vector Machine (SVM)** - Kernel-based classifier
- **K-Nearest Neighbors (KNN)** - Instance-based learning

## Features

✨ **MLflow + DagsHub Integration**
- Cloud-based experiment tracking with DagsHub
- Centralized model registry
- Metrics and parameter logging
- Artifact storage and management
- Team collaboration and sharing

📊 **Model Comparison & Visualization**
- Side-by-side performance metrics
- Training vs test accuracy tracking
- Sample size tracking
- Automated comparison chart generation
- Visual performance analysis

🔧 **MLOps Best Practices**
- Reproducible experiments
- Version control ready
- Modular code structure
- Automated model logging
- Remote experiment tracking
- Artifact versioning

## Requirements

- Python 3.7+
- MLflow
- DagsHub
- scikit-learn
- pandas
- matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nishnarudkar/MLflow_implementation_MLOPS.git
cd MLflow_implementation_MLOPS
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up DagsHub (Optional - for remote tracking):
   - Create a free account at [DagsHub](https://dagshub.com/)
   - Fork or create a repository
   - Update the `dagshub.init()` parameters in `main.py` with your credentials

## Usage

### Quick Start

1. **Configure DagsHub credentials (first time only):**

Set environment variables:
```bash
# On Linux/Mac
export MLFLOW_TRACKING_USERNAME=your_dagshub_username
export MLFLOW_TRACKING_PASSWORD=your_dagshub_token

# On Windows (PowerShell)
$env:MLFLOW_TRACKING_USERNAME="your_dagshub_username"
$env:MLFLOW_TRACKING_PASSWORD="your_dagshub_token"
```

Or the script will prompt you for credentials on first run.

2. **Run the training script:**
```bash
python main.py
```

The script will:
- Train all four models
- Log parameters and metrics to MLflow/DagsHub
- Save model artifacts
- Generate a comparison chart
- Display results in console

3. **View results:**
   - **DagsHub UI**: Visit your DagsHub repository to see experiments
   - **Local MLflow UI**: Run `mlflow ui` and open `http://127.0.0.1:5000`
   - **Comparison Chart**: Check `model_comparison.png` in your directory

## Project Structure

```
MLflow_implementation_MLOPS/
├── main.py                  # Main training and tracking script
├── requirements.txt         # Project dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
├── model_comparison.png    # Generated comparison chart
├── mlflow.db               # MLflow tracking database (generated)
└── mlruns/                 # Experiment artifacts and logs (generated)
```

## What Gets Logged

For each model, MLflow tracks:
- **Parameters**: 
  - Model name
  - Training samples count (455)
  - Test samples count (114)
- **Metrics**: 
  - Training accuracy
  - Test accuracy
- **Artifacts**: 
  - Serialized model files
  - Model comparison visualization chart
- **Metadata**: Run timestamps and execution details

## Visualization

The script automatically generates a bar chart comparing test accuracy across all models. This chart is:
- Saved locally as `model_comparison.png`
- Logged as an artifact in MLflow
- Accessible through the DagsHub/MLflow UI

## DagsHub Integration

DagsHub provides:
- **Remote Tracking**: All experiments synced to the cloud
- **Collaboration**: Share experiments with team members
- **Visualization**: Enhanced UI for comparing runs
- **Data Versioning**: Track datasets alongside models
- **Git Integration**: Seamless version control

Access your experiments at: `https://dagshub.com/nishnarudkar/MLflow_implementation_MLOPS`

## Results & Analysis

Access the DagsHub or MLflow UI to:
- Compare model performance across runs
- Visualize metrics and parameters
- Download trained models
- View comparison charts
- Track experiment history
- Select the best model for deployment
- Share results with collaborators

## Dataset

**Wisconsin Breast Cancer Dataset**
- 569 samples (455 train / 114 test)
- 30 features
- Binary classification (malignant/benign)
- Loaded via scikit-learn
- 80/20 train-test split with random_state=42

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is open source and available for educational purposes.

## Author

[Nishant Narudkar](https://github.com/nishnarudkar)

## Acknowledgments

- MLflow for experiment tracking capabilities
- DagsHub for cloud-based MLOps platform
- scikit-learn for ML algorithms and dataset
- Wisconsin Breast Cancer dataset contributors
