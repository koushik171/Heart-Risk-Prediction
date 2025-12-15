import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import json

def load_and_preprocess_data():
    """Load and preprocess the heart disease dataset"""
    df = pd.read_excel('../public/data/heart_risk_dataset.xlsx')
    df = df.dropna()
    
    X = df.drop('target', axis=1) if 'target' in df.columns else df.iloc[:, :-1]
    y = df['target'] if 'target' in df.columns else df.iloc[:, -1]
    
    return X, y, df

def train_models(X, y):
    """Train multiple ML models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'classification_report': classification_report(y_test, rf_pred, output_dict=True)
    }
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    models['logistic_regression'] = lr_model
    results['logistic_regression'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'classification_report': classification_report(y_test, lr_pred, output_dict=True)
    }
    
    # Save models
    joblib.dump(models['random_forest'], 'models/random_forest_model.pkl')
    joblib.dump(models['logistic_regression'], 'models/logistic_regression_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    with open('models/model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return models, results, scaler

if __name__ == "__main__":
    print("Loading data...")
    X, y, df = load_and_preprocess_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    print("Training models...")
    models, results, scaler = train_models(X, y)
    
    for model_name, result in results.items():
        print(f"{model_name}: Accuracy = {result['accuracy']:.4f}")