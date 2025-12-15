# Heart Disease Prediction - Machine Learning Components

This directory contains the Python machine learning components for the Heart Guardian Guide project.

## Files Overview

### 1. `train_model.py`
- Trains Random Forest and Logistic Regression models
- Performs data preprocessing and feature scaling
- Saves trained models to `models/` directory
- Generates model performance metrics

### 2. `predict_api.py`
- Flask API server for real-time predictions
- Loads trained models and serves predictions
- Converts frontend data to ML features
- Returns risk percentage, level, and feature importance

### 3. `data_analysis.py`
- Exploratory data analysis (EDA)
- Generates correlation matrices and feature distributions
- Creates visualizations for data insights

### 4. `requirements.txt`
- Python dependencies for ML components

## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   cd ml
   pip install -r requirements.txt
   ```

2. **Run data analysis:**
   ```bash
   python data_analysis.py
   ```

3. **Train models:**
   ```bash
   python train_model.py
   ```

4. **Start prediction API:**
   ```bash
   python predict_api.py
   ```

## Integration with Frontend

The Flask API runs on `http://localhost:5000` and provides:
- `POST /predict` - Heart disease risk prediction

To integrate with the React frontend, update the `dataProcessor.ts` to call the Python API instead of using the JavaScript risk calculation.

## Model Performance

After training, check `models/model_results.json` for:
- Model accuracy scores
- Classification reports
- Feature importance rankings

## Generated Files

- `models/random_forest_model.pkl` - Trained Random Forest model
- `models/logistic_regression_model.pkl` - Trained Logistic Regression model
- `models/scaler.pkl` - Feature scaler
- `models/model_results.json` - Model performance metrics
- `correlation_matrix.png` - Feature correlation heatmap
- `feature_distributions.png` - Feature distribution plots
- `target_analysis.png` - Target variable analysis