"""
Comprehensive accuracy improvement script combining all advanced techniques
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib

def run_accuracy_improvement():
    """Run comprehensive accuracy improvement pipeline"""
    
    print("🚀 Starting Accuracy Improvement Pipeline...")
    
    # 1. Load and enhance data
    df = pd.read_excel('../public/data/heart_risk_dataset.xlsx')
    df = df.dropna()
    print(f"📊 Dataset loaded: {df.shape}")
    
    # 2. Advanced feature engineering
    df_enhanced = create_advanced_features(df)
    print(f"🔧 Features enhanced: {df_enhanced.shape}")
    
    # 3. Prepare data
    X = df_enhanced.drop('target', axis=1) if 'target' in df_enhanced.columns else df_enhanced.iloc[:, :-1]
    y = df_enhanced['target'] if 'target' in df_enhanced.columns else df_enhanced.iloc[:, -1]
    
    # 4. Feature selection
    X_selected, selected_features = select_best_features(X, y)
    print(f"✨ Selected {len(selected_features)} best features")
    
    # 5. Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Advanced scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"⚖️ Balanced classes: {np.bincount(y_train_balanced)}")
    
    # 8. Train multiple high-performance models
    models = train_advanced_models(X_train_balanced, y_train_balanced)
    
    # 9. Evaluate all models
    best_model, best_accuracy = evaluate_models(models, X_test_scaled, y_test)
    
    # 10. Save best model
    save_best_model(best_model, scaler, selected_features)
    
    print(f"🎯 Best Model Accuracy: {best_accuracy:.4f}")
    return best_model, best_accuracy

def create_advanced_features(df):
    """Create advanced engineered features"""
    df_new = df.copy()
    
    # Age-based features
    if 'age' in df.columns:
        df_new['age_squared'] = df['age'] ** 2
        df_new['age_log'] = np.log(df['age'] + 1)
        df_new['age_group'] = pd.cut(df['age'], bins=[0, 35, 50, 65, 100], labels=[0, 1, 2, 3])
    
    # Cholesterol features
    if 'chol' in df.columns:
        df_new['chol_log'] = np.log(df['chol'] + 1)
        df_new['chol_risk'] = (df['chol'] > 240).astype(int)
    
    # Blood pressure features
    if 'trestbps' in df.columns:
        df_new['bp_high'] = (df['trestbps'] > 140).astype(int)
        df_new['bp_log'] = np.log(df['trestbps'] + 1)
    
    # Heart rate features
    if 'thalach' in df.columns:
        df_new['hr_low'] = (df['thalach'] < 100).astype(int)
        df_new['hr_high'] = (df['thalach'] > 180).astype(int)
    
    # Interaction features
    if 'age' in df.columns and 'chol' in df.columns:
        df_new['age_chol_interaction'] = df['age'] * df['chol'] / 1000
    
    if 'age' in df.columns and 'trestbps' in df.columns:
        df_new['age_bp_interaction'] = df['age'] * df['trestbps'] / 1000
    
    # Risk scores
    risk_features = ['cp', 'fbs', 'restecg', 'exang']
    if all(col in df.columns for col in risk_features):
        df_new['total_risk_score'] = df[risk_features].sum(axis=1)
        df_new['risk_score_squared'] = df_new['total_risk_score'] ** 2
    
    return df_new

def select_best_features(X, y, k=15):
    """Select best features using multiple methods"""
    # Method 1: Statistical selection
    selector_stats = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_stats = selector_stats.fit_transform(X, y)
    stats_features = X.columns[selector_stats.get_support()].tolist()
    
    # Method 2: Recursive feature elimination
    rf_selector = RandomForestClassifier(n_estimators=50, random_state=42)
    selector_rfe = RFE(rf_selector, n_features_to_select=min(k, X.shape[1]))
    X_rfe = selector_rfe.fit_transform(X, y)
    rfe_features = X.columns[selector_rfe.get_support()].tolist()
    
    # Combine features from both methods
    combined_features = list(set(stats_features + rfe_features))
    X_selected = X[combined_features]
    
    return X_selected, combined_features

def train_advanced_models(X_train, y_train):
    """Train multiple advanced models"""
    models = {}
    
    # 1. Optimized Random Forest
    models['rf_optimized'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
    
    # 2. Gradient Boosting
    models['gradient_boost'] = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    
    # 3. XGBoost
    models['xgboost'] = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # 4. LightGBM
    models['lightgbm'] = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    # 5. Voting Ensemble
    models['voting_ensemble'] = VotingClassifier(
        estimators=[
            ('rf', models['rf_optimized']),
            ('gb', models['gradient_boost']),
            ('xgb', models['xgboost']),
            ('lgb', models['lightgbm'])
        ],
        voting='soft'
    )
    
    # Train all models
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        model.fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and return the best one"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'model': model
        }
        
        print(f"📈 {name}: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}")
    
    # Find best model by accuracy
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"🏆 Best Model: {best_model_name}")
    
    return best_model, best_accuracy

def save_best_model(model, scaler, features):
    """Save the best performing model"""
    joblib.dump(model, 'models/best_accuracy_model.pkl')
    joblib.dump(scaler, 'models/best_accuracy_scaler.pkl')
    
    with open('models/best_model_features.txt', 'w') as f:
        f.write('\n'.join(features))
    
    print("💾 Best model saved!")

if __name__ == "__main__":
    best_model, accuracy = run_accuracy_improvement()
    print(f"\n🎉 Accuracy Improvement Complete!")
    print(f"🎯 Final Accuracy: {accuracy:.4f}")
    print("📁 Best model saved as 'best_accuracy_model.pkl'")