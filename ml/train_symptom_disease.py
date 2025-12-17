import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

def train_symptom_disease_model():
    """Train model to predict disease from symptoms and risk factors"""
    
    # Load dataset
    df = pd.read_csv(r"C:\Users\KOUSHIK\Downloads\heart_disease_symptoms_dataset_20000.csv")
    print(f"Dataset loaded: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Handle missing values
    df = df.dropna()
    print(f"After removing NaN: {df.shape}")
    
    # Encode categorical variables
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
    
    # Find disease column
    disease_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['disease', 'diagnosis', 'condition', 'target']):
            disease_col = col
            break
    
    if not disease_col:
        disease_col = df.columns[-1]  # Use last column
    
    print(f"Target column: {disease_col}")
    
    # Prepare features and target
    X = df.drop(disease_col, axis=1)
    y = df[disease_col]
    
    print(f"Features: {list(X.columns)}")
    print(f"Disease classes: {sorted(y.unique())}")
    print(f"Disease distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    results = {}
    
    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'classification_report': classification_report(y_test, rf_pred, output_dict=True)
    }
    
    # Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    
    models['gradient_boosting'] = gb_model
    results['gradient_boosting'] = {
        'accuracy': accuracy_score(y_test, gb_pred),
        'classification_report': classification_report(y_test, gb_pred, output_dict=True)
    }
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    models['logistic_regression'] = lr_model
    results['logistic_regression'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'classification_report': classification_report(y_test, lr_pred, output_dict=True)
    }
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = models[best_model_name]
    
    # Save models
    joblib.dump(best_model, 'models/symptom_disease_model.pkl')
    joblib.dump(scaler, 'models/symptom_disease_scaler.pkl')
    joblib.dump(encoders, 'models/symptom_disease_encoders.pkl')
    joblib.dump(list(X.columns), 'models/symptom_disease_features.pkl')
    
    print("\n" + "="*60)
    print("SYMPTOM-DISEASE MODEL TRAINING RESULTS")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
    
    print(f"\nBest Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
    print("Models saved successfully!")
    
    return models, results, scaler, encoders

def predict_disease_from_symptoms(symptoms_data):
    """Predict disease from symptoms and risk factors"""
    try:
        # Load trained model
        model = joblib.load('models/symptom_disease_model.pkl')
        scaler = joblib.load('models/symptom_disease_scaler.pkl')
        encoders = joblib.load('models/symptom_disease_encoders.pkl')
        features = joblib.load('models/symptom_disease_features.pkl')
        
        # Prepare input
        input_array = []
        for feature in features:
            value = symptoms_data.get(feature, 0)
            input_array.append(value)
        
        # Scale and predict
        input_scaled = scaler.transform([input_array])
        disease_prediction = model.predict(input_scaled)[0]
        disease_probability = model.predict_proba(input_scaled)[0]
        
        # Get disease name if encoded
        if 'disease' in encoders:
            disease_name = encoders['disease'].inverse_transform([disease_prediction])[0]
        else:
            disease_name = str(disease_prediction)
        
        return {
            'predicted_disease': disease_name,
            'confidence': float(max(disease_probability)),
            'all_probabilities': {str(i): float(prob) for i, prob in enumerate(disease_probability)}
        }
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    models, results, scaler, encoders = train_symptom_disease_model()
    
    # Test prediction
    try:
        features = joblib.load('models/symptom_disease_features.pkl')
        sample_data = {feature: 0 for feature in features}
        test_result = predict_disease_from_symptoms(sample_data)
        print(f"\nSample prediction: {test_result}")
    except:
        print("\nSample prediction not available yet - run training first")