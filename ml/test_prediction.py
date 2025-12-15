#!/usr/bin/env python3
"""
Test script for the trained heart disease prediction model
"""

from heart_predictor import HeartPredictor
import pandas as pd

def test_model():
    """Test the trained model with sample data"""
    
    # Initialize predictor
    predictor = HeartPredictor()
    
    if predictor.model is None:
        print("Error: Model not loaded. Please run training first.")
        return
    
    print("Heart Disease Prediction Model Test")
    print("="*50)
    
    # Sample test cases
    test_cases = [
        {
            "name": "Low Risk Patient",
            "data": {
                "Age": 25,
                "WeightKg": 70,
                "HeightCm": 175,
                "FamilyHistoryHeartDisease": 0,
                "SmokingHistory": 0,
                "ChestPain": 0,
                "UnusualFatigue": 0,
                "IrregularHeartbeat": 0,
                "HeartPalpitations": 0,
                "JawOrBackPain": 0,
                "FaintingEpisodes": 0,
                "DifficultyDuringExercise": 0,
                "ShortnessOfBreath": 0,
                "SwellingLegsAnkles": 0,
                "DizzinessLightheaded": 0,
                "HighBloodPressure": 0,
                "NauseaVomiting": 0,
                "BlueLipsFingernails": 0,
                "PredictedCondition": 0
            }
        },
        {
            "name": "High Risk Patient",
            "data": {
                "Age": 65,
                "WeightKg": 90,
                "HeightCm": 170,
                "FamilyHistoryHeartDisease": 1,
                "SmokingHistory": 1,
                "ChestPain": 1,
                "UnusualFatigue": 1,
                "IrregularHeartbeat": 1,
                "HeartPalpitations": 1,
                "JawOrBackPain": 1,
                "FaintingEpisodes": 0,
                "DifficultyDuringExercise": 1,
                "ShortnessOfBreath": 1,
                "SwellingLegsAnkles": 1,
                "DizzinessLightheaded": 1,
                "HighBloodPressure": 1,
                "NauseaVomiting": 0,
                "BlueLipsFingernails": 0,
                "PredictedCondition": 3
            }
        }
    ]
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 30)
        
        result = predictor.predict(test_case['data'])
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            continue
        
        print(f"Prediction: {result['prediction']}")
        print(f"Risk Level: {result['risk_level']}")
        
        if 'probability' in result:
            probs = result['probability']
            print("Probabilities:")
            for condition, prob in probs.items():
                print(f"  {condition}: {prob:.4f}")
        
        # Show top 5 most important features
        if 'feature_importance' in result and result['feature_importance']:
            print("\nTop 5 Important Features:")
            for j, (feature, importance) in enumerate(list(result['feature_importance'].items())[:5]):
                print(f"  {j+1}. {feature}: {importance:.4f}")

def interactive_prediction():
    """Interactive prediction mode"""
    predictor = HeartPredictor()
    
    if predictor.model is None:
        print("Error: Model not loaded.")
        return
    
    print("\nInteractive Prediction Mode")
    print("="*50)
    print("Enter patient data (press Enter for default values):")
    
    # Get input for each feature
    patient_data = {}
    feature_defaults = {
        "Age": 40,
        "WeightKg": 70,
        "HeightCm": 170,
        "FamilyHistoryHeartDisease": 0,
        "SmokingHistory": 0,
        "ChestPain": 0,
        "UnusualFatigue": 0,
        "IrregularHeartbeat": 0,
        "HeartPalpitations": 0,
        "JawOrBackPain": 0,
        "FaintingEpisodes": 0,
        "DifficultyDuringExercise": 0,
        "ShortnessOfBreath": 0,
        "SwellingLegsAnkles": 0,
        "DizzinessLightheaded": 0,
        "HighBloodPressure": 0,
        "NauseaVomiting": 0,
        "BlueLipsFingernails": 0,
        "PredictedCondition": 0
    }
    
    for feature in predictor.feature_names:
        default_val = feature_defaults.get(feature, 0)
        user_input = input(f"{feature} (default {default_val}): ").strip()
        
        if user_input:
            try:
                patient_data[feature] = float(user_input)
            except ValueError:
                print(f"Invalid input for {feature}, using default: {default_val}")
                patient_data[feature] = default_val
        else:
            patient_data[feature] = default_val
    
    # Make prediction
    result = predictor.predict(patient_data)
    
    print("\nPrediction Results:")
    print("="*30)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Prediction: {result['prediction']}")
        print(f"Risk Level: {result['risk_level']}")
        
        if 'probability' in result:
            probs = result['probability']
            print("Probabilities:")
            for condition, prob in probs.items():
                print(f"  {condition}: {prob:.4f}")

if __name__ == "__main__":
    # Run automated tests
    test_model()
    
    # Ask if user wants interactive mode
    print("\n" + "="*50)
    choice = input("Do you want to try interactive prediction? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        interactive_prediction()
    
    print("\nTest completed!")