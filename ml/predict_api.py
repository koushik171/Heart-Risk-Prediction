from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load trained models
try:
    rf_model = joblib.load('models/random_forest_model.pkl')
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("Models loaded successfully")
except:
    rf_model = lr_model = scaler = None
    print("Models not found. Run train_model.py first.")

@app.route('/predict', methods=['POST'])
def predict():
    """Predict heart disease risk"""
    try:
        data = request.json
        
        # Convert symptoms to features
        features = convert_symptoms_to_features(data)
        
        if rf_model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Get predictions
        rf_prob = rf_model.predict_proba(features_scaled)[0][1]
        lr_prob = lr_model.predict_proba(features_scaled)[0][1]
        
        # Average predictions
        avg_prob = (rf_prob + lr_prob) / 2
        risk_percentage = avg_prob * 100
        
        # Determine risk level
        if risk_percentage > 70:
            risk_level = 'High'
        elif risk_percentage > 40:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Feature importance (simplified)
        feature_importance = get_feature_importance(features)
        
        return jsonify({
            'riskPercentage': round(risk_percentage, 2),
            'riskLevel': risk_level,
            'confidence': round(max(rf_prob, 1-rf_prob) * 100, 2),
            'featureImportance': feature_importance
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_symptoms_to_features(data):
    """Convert frontend data to ML features"""
    features = [0] * 13  # Assuming 13 features in dataset
    
    # Age
    features[0] = data.get('age', 50) / 100  # Normalize
    
    # Gender (assume male=1, female=0)
    features[1] = 1  # Default
    
    # Chest pain type
    symptoms = data.get('symptoms', [])
    features[2] = 1 if 'chest_pain' in symptoms else 0
    
    # Blood pressure
    features[3] = 1 if 'high_bp' in symptoms else 0
    
    # Other features based on symptoms
    features[4] = 1 if 'shortness_breath' in symptoms else 0
    features[5] = 1 if data.get('familyHistory', False) else 0
    features[6] = 1 if data.get('smokingHistory', False) else 0
    
    # BMI calculation
    if data.get('weight') and data.get('height'):
        bmi = data['weight'] / ((data['height']/100) ** 2)
        features[7] = min(bmi / 40, 1)  # Normalize BMI
    
    return features

def get_feature_importance(features):
    """Get simplified feature importance"""
    feature_names = ['Age', 'Gender', 'Chest Pain', 'Blood Pressure', 
                    'Shortness of Breath', 'Family History', 'Smoking', 'BMI']
    
    importance = []
    for i, (name, value) in enumerate(zip(feature_names[:len(features)], features)):
        if value > 0:
            importance.append({
                'feature': name,
                'importance': round(value * 0.3, 2),
                'value': 'Present' if value == 1 else f'{value:.2f}'
            })
    
    return sorted(importance, key=lambda x: x['importance'], reverse=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)