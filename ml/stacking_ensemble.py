import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib

def create_stacking_ensemble():
    """Create stacking ensemble with multiple base learners"""
    
    # Base learners (Level 0)
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)),
        ('lgb', lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1)),
        ('svm', SVC(C=1.0, kernel='rbf', probability=True, random_state=42))
    ]
    
    # Meta-learner (Level 1)
    meta_learner = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    
    # Stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    return stacking_clf

def train_stacking_model():
    """Train stacking ensemble model"""
    # Load data
    df = pd.read_excel('../public/data/heart_risk_dataset.xlsx')
    df = df.dropna()
    
    # Prepare features
    X = df.drop('target', axis=1) if 'target' in df.columns else df.iloc[:, :-1]
    y = df['target'] if 'target' in df.columns else df.iloc[:, -1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train stacking model
    stacking_model = create_stacking_ensemble()
    print("Training stacking ensemble...")
    stacking_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = stacking_model.predict(X_test_scaled)
    y_pred_proba = stacking_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Stacking Ensemble Accuracy: {accuracy:.4f}")
    print(f"Stacking Ensemble AUC: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(stacking_model, 'models/stacking_ensemble_model.pkl')
    joblib.dump(scaler, 'models/stacking_scaler.pkl')
    
    return stacking_model, scaler

if __name__ == "__main__":
    model, scaler = train_stacking_model()
    print("Stacking ensemble training complete!")