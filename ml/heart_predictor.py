import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Union

class HeartPredictor:
    def __init__(self, model_path='models/best_heart_model.pkl', 
                 scaler_path='models/heart_scaler.pkl',
                 features_path='models/heart_features.txt'):
        """Initialize the heart disease predictor"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            with open(features_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            print(f"Model loaded successfully. Expected features: {len(self.feature_names)}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
            self.feature_names = []
    
    def predict(self, input_data: Union[Dict, List, pd.DataFrame]) -> Dict:
        """Make prediction on input data"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                df = pd.DataFrame([input_data], columns=self.feature_names)
            else:
                df = input_data.copy()
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0  # Default value for missing features
            
            # Reorder columns to match training data
            df = df[self.feature_names]
            
            # Scale the data
            X_scaled = self.scaler.transform(df)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            # Get feature importance if available
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                feature_importance = dict(zip(self.feature_names, importance))
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
            
            return {
                "prediction": int(prediction),
                "probability": {
                    "no_disease": float(probability[0]),
                    "disease": float(probability[1]) if len(probability) > 1 else 0.0
                },
                "risk_level": self._get_risk_level(probability),
                "feature_importance": feature_importance
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def _get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if len(probability) < 2:
            return "Unknown"
        
        disease_prob = probability[1]
        if disease_prob < 0.3:
            return "Low"
        elif disease_prob < 0.7:
            return "Medium"
        else:
            return "High"
    
    def get_feature_names(self):
        """Return the expected feature names"""
        return self.feature_names

# Example usage
if __name__ == "__main__":
    predictor = HeartPredictor()
    
    # Example prediction with sample data
    sample_data = {
        # Add sample values based on your dataset features
        # This is just an example - replace with actual feature names
    }
    
    if predictor.model is not None:
        result = predictor.predict(sample_data)
        print("Prediction Result:")
        print(f"Prediction: {result.get('prediction')}")
        print(f"Risk Level: {result.get('risk_level')}")
        print(f"Probabilities: {result.get('probability')}")
        
        print("\nExpected features:")
        for i, feature in enumerate(predictor.get_feature_names()):
            print(f"{i+1}. {feature}")