import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load and train heart disease model
df = pd.read_excel(r"C:\Project\Heart Prediction\HeartPredict_Training_2000.xlsx")
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

joblib.dump(model, 'models/quick_model.pkl')
joblib.dump(list(X.columns), 'models/quick_features.pkl')

print("Quick model trained and saved!")