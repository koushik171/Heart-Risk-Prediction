#!/usr/bin/env python3
"""
Simple script to run heart disease model training
"""

import os
import sys
from train_custom_dataset import load_and_preprocess_data, train_models

def main():
    # Dataset path
    dataset_path = r"C:\Project\Heart Prediction\HeartPredict_Training_2000.xlsx"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please check the file path and try again.")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        print("Starting Heart Disease Model Training")
        print("="*50)
        
        # Load and preprocess data
        print("Loading dataset...")
        X, y, label_encoders = load_and_preprocess_data(dataset_path)
        
        # Train models
        print("Training models...")
        models, results, scaler, best_model_name = train_models(X, y)
        
        # Display results
        print("\nTraining Complete!")
        print("="*50)
        print("Model Performance:")
        
        for model_name, result in results.items():
            accuracy = result['accuracy']
            status = "[BEST]" if model_name == best_model_name else "     "
            print(f"{status} {model_name}: {accuracy:.4f}")
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Models saved in 'models/' directory")
        
        # Test the predictor
        print("\nTesting predictor...")
        from heart_predictor import HeartPredictor
        
        predictor = HeartPredictor()
        if predictor.model is not None:
            print("Predictor loaded successfully!")
            print(f"Expected features ({len(predictor.feature_names)}):")
            for i, feature in enumerate(predictor.feature_names[:5]):  # Show first 5
                print(f"   {i+1}. {feature}")
            if len(predictor.feature_names) > 5:
                print(f"   ... and {len(predictor.feature_names) - 5} more")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return

if __name__ == "__main__":
    main()