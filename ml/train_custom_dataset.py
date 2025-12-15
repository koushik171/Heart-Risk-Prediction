import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json

def load_and_preprocess_data(file_path):
    """Load and preprocess the heart disease dataset"""
    df = pd.read_excel(file_path)
    print(f"Dataset loaded: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle missing values
    df = df.dropna()
    print(f"After removing NaN: {df.shape}")
    
    # Encode categorical variables if any
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    
    # Assume last column is target or find target column
    target_col = df.columns[-1]  # Default to last column
    if 'target' in df.columns:
        target_col = 'target'
    elif 'diagnosis' in df.columns:
        target_col = 'diagnosis'
    elif 'heart_disease' in df.columns:
        target_col = 'heart_disease'
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, label_encoders

def train_models(X, y):
    """Train multiple ML models with hyperparameter tuning"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # Random Forest with GridSearch
    rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy')
    rf_grid.fit(X_train_scaled, y_train)
    
    rf_pred = rf_grid.predict(X_test_scaled)
    models['random_forest'] = rf_grid.best_estimator_
    results['random_forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'best_params': rf_grid.best_params_,
        'classification_report': classification_report(y_test, rf_pred, output_dict=True)
    }
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train_scaled, y_train)
    gb_pred = gb.predict(X_test_scaled)
    
    models['gradient_boosting'] = gb
    results['gradient_boosting'] = {
        'accuracy': accuracy_score(y_test, gb_pred),
        'classification_report': classification_report(y_test, gb_pred, output_dict=True)
    }
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    
    models['logistic_regression'] = lr
    results['logistic_regression'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'classification_report': classification_report(y_test, lr_pred, output_dict=True)
    }
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = models[best_model_name]
    
    # Save models and results
    joblib.dump(best_model, 'models/best_heart_model.pkl')
    joblib.dump(scaler, 'models/heart_scaler.pkl')
    
    with open('models/heart_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save feature names
    with open('models/heart_features.txt', 'w') as f:
        f.write('\n'.join(X.columns.tolist()))
    
    return models, results, scaler, best_model_name

if __name__ == "__main__":
    # Path to your dataset
    dataset_path = r"C:\Project\Heart Prediction\HeartPredict_Training_2000.xlsx"
    
    print("Loading and preprocessing data...")
    X, y, label_encoders = load_and_preprocess_data(dataset_path)
    
    print("\nTraining models...")
    models, results, scaler, best_model_name = train_models(X, y)
    
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        if 'best_params' in result:
            print(f"  Best params: {result['best_params']}")
    
    print(f"\nBest Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
    print("\nModels saved to 'models/' directory")