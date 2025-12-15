import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_excel(r"C:\Project\Heart Prediction\health_disease_dataset.xlsx")
print(f"Dataset: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Encode categorical data
encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])

# Find disease/target column
disease_col = None
for col in df.columns:
    if any(word in col.lower() for word in ['disease', 'diagnosis', 'condition', 'target']):
        disease_col = col
        break

if not disease_col:
    disease_col = df.columns[-1]  # Use last column

print(f"Target column: {disease_col}")

# Prepare data
X = df.drop(disease_col, axis=1)
y = df[disease_col]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy:.4f}")

# Save model and encoders
joblib.dump(model, 'models/disease_model.pkl')
joblib.dump(encoders, 'models/disease_encoders.pkl')
joblib.dump(list(X.columns), 'models/disease_features.pkl')

print("Disease prediction model saved!")

# Create prediction function
def predict_disease(symptoms_dict):
    """Predict disease based on symptoms"""
    model = joblib.load('models/disease_model.pkl')
    encoders = joblib.load('models/disease_encoders.pkl')
    features = joblib.load('models/disease_features.pkl')
    
    # Create input array
    input_data = []
    for feature in features:
        value = symptoms_dict.get(feature, 0)
        if feature in encoders and isinstance(value, str):
            try:
                value = encoders[feature].transform([value])[0]
            except:
                value = 0
        input_data.append(value)
    
    # Predict
    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0].max()
    
    # Decode prediction if needed
    if disease_col in encoders:
        prediction = encoders[disease_col].inverse_transform([prediction])[0]
    
    return {
        'disease': prediction,
        'confidence': probability,
        'risk_level': 'High' if probability > 0.8 else 'Medium' if probability > 0.5 else 'Low'
    }

# Test prediction
if __name__ == "__main__":
    # Example usage
    sample_symptoms = {feature: 0 for feature in X.columns}  # Default values
    result = predict_disease(sample_symptoms)
    print(f"Sample prediction: {result}")