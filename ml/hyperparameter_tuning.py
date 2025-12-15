import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

def tune_random_forest(X_train, y_train):
    """Hyperparameter tuning for Random Forest"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Random Forest - Best parameters:")
    print(grid_search.best_params_)
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_logistic_regression(X_train, y_train):
    """Hyperparameter tuning for Logistic Regression"""
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Logistic Regression - Best parameters:")
    print(grid_search.best_params_)
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_models():
    """Tune all models and save the best ones"""
    # Load data
    df = pd.read_excel('../public/data/heart_risk_dataset.xlsx')
    df = df.dropna()
    
    X = df.drop('target', axis=1) if 'target' in df.columns else df.iloc[:, :-1]
    y = df['target'] if 'target' in df.columns else df.iloc[:, -1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Tuning Random Forest...")
    best_rf = tune_random_forest(X_train_scaled, y_train)
    
    print("\nTuning Logistic Regression...")
    best_lr = tune_logistic_regression(X_train_scaled, y_train)
    
    # Evaluate tuned models
    print("\nEvaluating tuned models:")
    
    rf_pred = best_rf.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Tuned Random Forest accuracy: {rf_accuracy:.4f}")
    
    lr_pred = best_lr.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"Tuned Logistic Regression accuracy: {lr_accuracy:.4f}")
    
    # Save best models
    joblib.dump(best_rf, 'models/tuned_random_forest_model.pkl')
    joblib.dump(best_lr, 'models/tuned_logistic_regression_model.pkl')
    joblib.dump(scaler, 'models/tuned_scaler.pkl')
    
    print("\nTuned models saved to models/ directory")

if __name__ == "__main__":
    tune_models()