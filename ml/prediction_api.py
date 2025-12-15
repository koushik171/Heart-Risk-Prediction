import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained models
try:
    heart_model = joblib.load('models/best_heart_model.pkl')
    heart_scaler = joblib.load('models/heart_scaler.pkl')
    
    with open('models/heart_features.txt', 'r') as f:
        heart_features = [line.strip() for line in f.readlines()]
    
    print("Heart disease model loaded successfully")
except Exception as e:
    print(f"Error loading heart model: {e}")
    heart_model = None

try:
    disease_model = joblib.load('models/health_model.pkl')
    print("Disease classification model loaded successfully")
except Exception as e:
    print(f"Error loading disease model: {e}")
    disease_model = None

try:
    suggestion_models = joblib.load('models/suggestion_models.pkl')
    suggestion_encoders = joblib.load('models/suggestion_encoders.pkl')
    suggestion_features = joblib.load('models/suggestion_features.pkl')
    print("Suggestion models loaded successfully")
except Exception as e:
    print(f"Error loading suggestion models: {e}")
    suggestion_models = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Map symptoms to model features
        input_data = []
        for feature in heart_features:
            if feature == 'Age':
                input_data.append(data.get('age', 40))
            elif feature == 'WeightKg':
                input_data.append(data.get('weight', 70))
            elif feature == 'HeightCm':
                input_data.append(data.get('height', 170))
            elif feature == 'FamilyHistoryHeartDisease':
                input_data.append(1 if data.get('familyHistory', False) else 0)
            elif feature == 'SmokingHistory':
                input_data.append(1 if data.get('smokingHistory', False) else 0)
            elif feature == 'ChestPain':
                input_data.append(1 if 'chest_pain' in data.get('symptoms', []) else 0)
            elif feature == 'UnusualFatigue':
                input_data.append(1 if 'fatigue' in data.get('symptoms', []) else 0)
            elif feature == 'IrregularHeartbeat':
                input_data.append(1 if 'irregular_heartbeat' in data.get('symptoms', []) else 0)
            elif feature == 'HeartPalpitations':
                input_data.append(1 if 'palpitations' in data.get('symptoms', []) else 0)
            elif feature == 'ShortnessOfBreath':
                input_data.append(1 if 'shortness_breath' in data.get('symptoms', []) else 0)
            elif feature == 'SwellingLegsAnkles':
                input_data.append(1 if 'swelling' in data.get('symptoms', []) else 0)
            elif feature == 'DizzinessLightheaded':
                input_data.append(1 if 'dizziness' in data.get('symptoms', []) else 0)
            elif feature == 'HighBloodPressure':
                input_data.append(1 if 'high_bp' in data.get('symptoms', []) else 0)
            elif feature == 'NauseaVomiting':
                input_data.append(1 if 'nausea' in data.get('symptoms', []) else 0)
            elif feature == 'BlueLipsFingernails':
                input_data.append(1 if 'cyanosis' in data.get('symptoms', []) else 0)
            elif feature == 'JawOrBackPain':
                input_data.append(1 if 'jaw_back_pain' in data.get('symptoms', []) else 0)
            elif feature == 'FaintingEpisodes':
                input_data.append(1 if 'fainting' in data.get('symptoms', []) else 0)
            elif feature == 'DifficultyDuringExercise':
                input_data.append(1 if 'exercise_difficulty' in data.get('symptoms', []) else 0)
            else:
                input_data.append(0)
        
        # Scale input
        input_scaled = heart_scaler.transform([input_data])
        
        # Predict
        prediction = heart_model.predict(input_scaled)[0]
        probabilities = heart_model.predict_proba(input_scaled)[0]
        
        # Calculate risk percentage
        risk_percentage = max(probabilities) * 100
        
        # Determine risk level
        if risk_percentage < 30:
            risk_level = 'Low'
        elif risk_percentage < 70:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Get disease names
        disease_names = ['No Disease', 'Mild Risk', 'Moderate Risk', 'High Risk', 'Severe Risk', 'Critical Risk']
        predicted_condition = disease_names[min(prediction, len(disease_names)-1)]
        
        return jsonify({
            'riskPercentage': float(risk_percentage),
            'riskLevel': risk_level,
            'confidence': float(max(probabilities) * 100),
            'predictedCondition': predicted_condition,
            'prediction': int(prediction)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)