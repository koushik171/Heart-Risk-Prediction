import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_excel(r"C:\Project\Heart Prediction\Heart_Safety_Medication_Diet_2000 (1).xlsx")
print(f"Loaded: {df.shape}")

# Encode strings
encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])

# Find suggestion columns (medication, diet, safety)
suggestion_cols = []
for col in df.columns:
    if any(word in col.lower() for word in ['medication', 'diet', 'safety', 'suggestion', 'treatment']):
        suggestion_cols.append(col)

if not suggestion_cols:
    suggestion_cols = [df.columns[-1]]  # Use last column

print(f"Suggestion columns: {suggestion_cols}")

# Prepare features (exclude suggestion columns)
X = df.drop(suggestion_cols, axis=1)

# Train models for each suggestion type
models = {}
for target_col in suggestion_cols:
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"{target_col} accuracy: {accuracy:.4f}")
    
    models[target_col] = model

# Save models
joblib.dump(models, 'models/suggestion_models.pkl')
joblib.dump(encoders, 'models/suggestion_encoders.pkl')
joblib.dump(list(X.columns), 'models/suggestion_features.pkl')

print("Suggestion models saved!")