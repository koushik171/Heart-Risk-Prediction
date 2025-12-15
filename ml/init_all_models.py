import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def train_heart_prediction():
    """Train heart disease prediction model"""
    df = pd.read_excel(r"C:\Project\Heart Prediction\HeartPredict_Training_2000.xlsx")
    
    # Encode categorical variables
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    # Save
    joblib.dump(model, 'models/heart_prediction_model.pkl')
    joblib.dump(scaler, 'models/heart_prediction_scaler.pkl')
    joblib.dump(encoders, 'models/heart_prediction_encoders.pkl')
    joblib.dump(list(X.columns), 'models/heart_prediction_features.pkl')
    
    print("Heart prediction model trained and saved")
    return model

def train_disease_classification():
    """Train disease classification model"""
    df = pd.read_excel(r"C:\Project\Heart Prediction\health_disease_dataset.xlsx")
    
    # Encode categorical variables
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save
    joblib.dump(model, 'models/disease_classification_model.pkl')
    joblib.dump(encoders, 'models/disease_classification_encoders.pkl')
    joblib.dump(list(X.columns), 'models/disease_classification_features.pkl')
    
    print("Disease classification model trained and saved")
    return model

def train_treatment_suggestions():
    """Train treatment suggestion models"""
    df = pd.read_excel(r"C:\Project\Heart Prediction\Heart_Safety_Medication_Diet_2000 (1).xlsx")
    
    # Encode categorical variables
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
    
    # Find suggestion columns
    suggestion_cols = []
    for col in df.columns:
        if any(word in col.lower() for word in ['medication', 'diet', 'safety', 'treatment']):
            suggestion_cols.append(col)
    
    if not suggestion_cols:
        suggestion_cols = df.columns[-3:].tolist()  # Last 3 columns
    
    X = df.drop(suggestion_cols, axis=1)
    
    # Train models for each suggestion type
    models = {}
    for target_col in suggestion_cols:
        y = df[target_col]
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        models[target_col] = model
    
    # Save
    joblib.dump(models, 'models/treatment_suggestion_models.pkl')
    joblib.dump(encoders, 'models/treatment_suggestion_encoders.pkl')
    joblib.dump(list(X.columns), 'models/treatment_suggestion_features.pkl')
    joblib.dump(suggestion_cols, 'models/treatment_suggestion_targets.pkl')
    
    print("Treatment suggestion models trained and saved")
    return models

def main():
    """Initialize all models"""
    os.makedirs('models', exist_ok=True)
    
    print("Initializing all ML models...")
    
    # Train all models
    heart_model = train_heart_prediction()
    disease_model = train_disease_classification()
    treatment_models = train_treatment_suggestions()
    
    print("\nAll models initialized successfully!")
    print("Available models:")
    print("- Heart Prediction Model")
    print("- Disease Classification Model") 
    print("- Treatment Suggestion Models")

if __name__ == "__main__":
    main()