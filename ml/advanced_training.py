import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

def advanced_feature_engineering(df):
    """Enhanced feature engineering"""
    df_new = df.copy()
    
    # Age-based features
    if 'age' in df.columns:
        df_new['age_squared'] = df['age'] ** 2
        df_new['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 70, 100], labels=[0, 1, 2, 3])
    
    # Cholesterol risk categories
    if 'chol' in df.columns:
        df_new['chol_risk'] = pd.cut(df['chol'], bins=[0, 200, 240, 1000], labels=[0, 1, 2])
    
    # Blood pressure categories
    if 'trestbps' in df.columns:
        df_new['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 1000], labels=[0, 1, 2])
    
    # Heart rate zones
    if 'thalach' in df.columns:
        df_new['hr_zone'] = pd.cut(df['thalach'], bins=[0, 100, 150, 200, 300], labels=[0, 1, 2, 3])
    
    # Interaction features
    if 'age' in df.columns and 'chol' in df.columns:
        df_new['age_chol'] = df['age'] * df['chol'] / 1000
    
    if 'age' in df.columns and 'trestbps' in df.columns:
        df_new['age_bp'] = df['age'] * df['trestbps'] / 1000
    
    # Risk score
    risk_cols = ['cp', 'fbs', 'restecg', 'exang']
    if all(col in df.columns for col in risk_cols):
        df_new['risk_score'] = df[risk_cols].sum(axis=1)
    
    return df_new

def create_ensemble_model():
    """Create advanced ensemble model"""
    # Individual models
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
    svm = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
    lr = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    xgb_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
    
    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('svm', svm),
            ('lr', lr),
            ('xgb', xgb_model)
        ],
        voting='soft'
    )
    
    return ensemble

def train_advanced_model():
    """Train advanced model with all improvements"""
    # Load and preprocess data
    df = pd.read_excel('../public/data/heart_risk_dataset.xlsx')
    df = df.dropna()
    
    # Advanced feature engineering
    df_enhanced = advanced_feature_engineering(df)
    
    # Prepare features and target
    X = df_enhanced.drop('target', axis=1) if 'target' in df_enhanced.columns else df_enhanced.iloc[:, :-1]
    y = df_enhanced['target'] if 'target' in df_enhanced.columns else df_enhanced.iloc[:, -1]
    
    # Handle categorical variables
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train ensemble
    ensemble = create_ensemble_model()
    ensemble.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Test evaluation
    y_pred = ensemble.predict(X_test_scaled)
    y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save models
    joblib.dump(ensemble, 'models/advanced_ensemble_model.pkl')
    joblib.dump(scaler, 'models/advanced_scaler.pkl')
    joblib.dump(selector, 'models/feature_selector.pkl')
    
    # Save feature names
    with open('models/selected_features.txt', 'w') as f:
        f.write('\n'.join(selected_features))
    
    return ensemble, scaler, selector, selected_features

if __name__ == "__main__":
    print("Training advanced ensemble model...")
    model, scaler, selector, features = train_advanced_model()
    print("Advanced model training complete!")