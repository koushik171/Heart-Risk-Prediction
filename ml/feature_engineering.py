import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def create_features(df):
    """Create additional features from existing data"""
    df_new = df.copy()
    
    # Age groups
    if 'age' in df.columns:
        df_new['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 30, 45, 60, 100], 
                                   labels=['Young', 'Middle', 'Senior', 'Elderly'])
        df_new['age_group'] = LabelEncoder().fit_transform(df_new['age_group'])
    
    # BMI calculation if weight and height exist
    if 'weight' in df.columns and 'height' in df.columns:
        df_new['bmi'] = df['weight'] / ((df['height']/100) ** 2)
        df_new['bmi_category'] = pd.cut(df_new['bmi'], 
                                      bins=[0, 18.5, 25, 30, 100],
                                      labels=[0, 1, 2, 3])  # Underweight, Normal, Overweight, Obese
    
    # Risk score combination
    risk_features = ['cp', 'trestbps', 'chol', 'fbs', 'restecg'] if all(col in df.columns for col in ['cp', 'trestbps', 'chol', 'fbs', 'restecg']) else []
    if risk_features:
        df_new['risk_score'] = df[risk_features].sum(axis=1)
    
    # Interaction features
    if 'age' in df.columns and 'chol' in df.columns:
        df_new['age_chol_interaction'] = df['age'] * df['chol'] / 1000
    
    return df_new

def preprocess_features(df, target_col='target'):
    """Preprocess features for ML models"""
    # Separate features and target
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, scaler

def select_features(X, y, k=10):
    """Select top k features using univariate selection"""
    from sklearn.feature_selection import SelectKBest, f_classif
    
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Selected {k} features:")
    for i, feature in enumerate(selected_features):
        print(f"{i+1}. {feature}")
    
    return X_selected, selected_features, selector

if __name__ == "__main__":
    # Load data
    df = pd.read_excel('../public/data/heart_risk_dataset.xlsx')
    
    print("Original features:", list(df.columns))
    
    # Create new features
    df_enhanced = create_features(df)
    print("Enhanced features:", list(df_enhanced.columns))
    
    # Preprocess
    X, y, scaler = preprocess_features(df_enhanced)
    print(f"Preprocessed shape: {X.shape}")
    
    # Feature selection
    X_selected, selected_features, selector = select_features(X, y, k=min(10, X.shape[1]))
    print(f"Selected features shape: {X_selected.shape}")