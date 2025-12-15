"""Utility functions for ML pipeline"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

def load_dataset(file_path):
    """Load dataset from various formats"""
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format")

def save_model(model, filepath):
    """Save model with metadata"""
    model_data = {
        'model': model,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    joblib.dump(model_data, filepath)

def load_model(filepath):
    """Load model with metadata"""
    if os.path.exists(filepath):
        model_data = joblib.load(filepath)
        if isinstance(model_data, dict) and 'model' in model_data:
            return model_data['model']
        return model_data
    return None

def calculate_risk_level(probability, thresholds={'low': 0.4, 'high': 0.7}):
    """Calculate risk level from probability"""
    if probability < thresholds['low']:
        return 'Low'
    elif probability < thresholds['high']:
        return 'Medium'
    else:
        return 'High'

def log_prediction(user_data, prediction, log_file='prediction_log.json'):
    """Log predictions for monitoring"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_data': user_data,
        'prediction': prediction
    }
    
    # Load existing logs
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    
    # Add new log
    logs.append(log_entry)
    
    # Save logs
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

def validate_input_data(data, required_fields=['age']):
    """Validate input data"""
    errors = []
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if 'age' in data:
        if not isinstance(data['age'], (int, float)) or data['age'] < 0 or data['age'] > 120:
            errors.append("Age must be between 0 and 120")
    
    if 'weight' in data and data['weight']:
        if not isinstance(data['weight'], (int, float)) or data['weight'] <= 0:
            errors.append("Weight must be positive")
    
    if 'height' in data and data['height']:
        if not isinstance(data['height'], (int, float)) or data['height'] <= 0:
            errors.append("Height must be positive")
    
    return errors

def get_model_info(model_path):
    """Get information about saved model"""
    if not os.path.exists(model_path):
        return None
    
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict):
        return {
            'timestamp': model_data.get('timestamp'),
            'version': model_data.get('version'),
            'file_size': os.path.getsize(model_path)
        }
    
    return {
        'file_size': os.path.getsize(model_path),
        'modified': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
    }

def create_feature_vector(user_input, feature_names):
    """Create feature vector from user input"""
    feature_vector = np.zeros(len(feature_names))
    
    # Map user input to features
    feature_mapping = {
        'age': 'age',
        'sex': 'sex',
        'chest_pain': 'cp',
        'blood_pressure': 'trestbps',
        'cholesterol': 'chol'
    }
    
    for user_field, feature_name in feature_mapping.items():
        if user_field in user_input and feature_name in feature_names:
            idx = feature_names.index(feature_name)
            feature_vector[idx] = user_input[user_field]
    
    return feature_vector