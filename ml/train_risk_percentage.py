import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import numpy as np

def train_risk_percentage_model():
    """Train model for heart risk percentage prediction"""
    
    # Load dataset
    df = pd.read_csv(r"C:\Users\KOUSHIK\Downloads\heart_risk_percentage_dataset_20000.csv")
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
    
    # Find risk percentage column
    risk_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['risk', 'percentage', 'percent']):
            risk_col = col
            break
    
    if not risk_col:
        risk_col = df.columns[-1]  # Use last column
    
    print(f"Target column: {risk_col}")
    
    # Prepare features and target
    X = df.drop(risk_col, axis=1)
    y = df[risk_col]
    
    print(f"Features: {list(X.columns)}")
    print(f"Risk percentage range: {y.min():.2f}% - {y.max():.2f}%")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    results = {}
    
    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'mse': mean_squared_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'mae': mean_absolute_error(y_test, rf_pred),
        'r2': r2_score(y_test, rf_pred)
    }
    
    # Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    
    models['gradient_boosting'] = gb_model
    results['gradient_boosting'] = {
        'mse': mean_squared_error(y_test, gb_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'mae': mean_absolute_error(y_test, gb_pred),
        'r2': r2_score(y_test, gb_pred)
    }
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    models['linear_regression'] = lr_model
    results['linear_regression'] = {
        'mse': mean_squared_error(y_test, lr_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'mae': mean_absolute_error(y_test, lr_pred),
        'r2': r2_score(y_test, lr_pred)
    }
    
    # Find best model (highest R²)
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = models[best_model_name]
    
    # Save models
    joblib.dump(best_model, 'models/risk_percentage_model.pkl')
    joblib.dump(scaler, 'models/risk_percentage_scaler.pkl')
    joblib.dump(encoders, 'models/risk_percentage_encoders.pkl')
    joblib.dump(list(X.columns), 'models/risk_percentage_features.pkl')
    
    print("\n" + "="*60)
    print("RISK PERCENTAGE MODEL TRAINING RESULTS")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  R² Score: {result['r2']:.4f}")
        print(f"  RMSE: {result['rmse']:.4f}%")
        print(f"  MAE: {result['mae']:.4f}%")
    
    print(f"\nBest Model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
    print("Models saved successfully!")
    
    return models, results, scaler, encoders

def predict_risk_percentage(input_data):
    """Predict risk percentage for new data"""
    try:
        # Load trained model
        model = joblib.load('models/risk_percentage_model.pkl')
        scaler = joblib.load('models/risk_percentage_scaler.pkl')
        features = joblib.load('models/risk_percentage_features.pkl')
        
        # Prepare input
        input_array = []
        for feature in features:
            value = input_data.get(feature, 0)
            input_array.append(value)
        
        # Scale and predict
        input_scaled = scaler.transform([input_array])
        risk_percentage = model.predict(input_scaled)[0]
        
        # Ensure percentage is within valid range
        risk_percentage = max(0, min(100, risk_percentage))
        
        return {
            'risk_percentage': round(risk_percentage, 2),
            'risk_level': 'High' if risk_percentage > 70 else 'Medium' if risk_percentage > 30 else 'Low'
        }
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    models, results, scaler, encoders = train_risk_percentage_model()
    
    # Test prediction
    sample_data = {feature: 0 for feature in joblib.load('models/risk_percentage_features.pkl')}
    test_result = predict_risk_percentage(sample_data)
    print(f"\nSample prediction: {test_result}")