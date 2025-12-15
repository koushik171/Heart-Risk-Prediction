import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_excel(r"C:\Project\Heart Prediction\HeartPredict_Training_2000.xlsx")
print(f"Data loaded: {df.shape}")

# Encode categorical variables
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Prepare features and target
X = df.iloc[:, :-1]  # All columns except last
y = df.iloc[:, -1]   # Last column as target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_scaled, y_train)

# Test accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Save model
joblib.dump(model, 'models/fast_model.pkl')
joblib.dump(scaler, 'models/fast_scaler.pkl')

print("Model saved successfully!")