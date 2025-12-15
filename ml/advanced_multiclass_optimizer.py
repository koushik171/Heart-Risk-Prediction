import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def deep_feature_engineering(df):
    """Advanced feature engineering"""
    df_new = df.copy()
    
    # Polynomial features for key variables
    df_new['Age_squared'] = df['Age'] ** 2
    df_new['BMI_squared'] = df['BMI'] ** 2
    df_new['BP_Systolic_squared'] = df['BP_Systolic'] ** 2
    
    # Log transformations
    df_new['Age_log'] = np.log(df['Age'] + 1)
    df_new['BMI_log'] = np.log(df['BMI'] + 1)
    df_new['Cholesterol_log'] = np.log(df['Cholesterol'] + 1)
    df_new['Glucose_log'] = np.log(df['Glucose'] + 1)
    
    # Ratios and combinations
    df_new['Weight_Height_ratio'] = df['Weight_kg'] / df['Height_cm']
    df_new['BP_ratio'] = df['BP_Systolic'] / df['BP_Diastolic']
    df_new['Pulse_pressure'] = df['BP_Systolic'] - df['BP_Diastolic']
    df_new['Mean_arterial_pressure'] = df['BP_Diastolic'] + (df_new['Pulse_pressure'] / 3)
    
    # Health risk indicators
    df_new['Metabolic_syndrome'] = (
        (df['BMI'] > 30).astype(int) +
        (df['BP_Systolic'] > 130).astype(int) +
        (df['Glucose'] > 100).astype(int) +
        (df['Cholesterol'] > 200).astype(int)
    )
    
    # Age-based risk factors
    df_new['Age_risk_heart'] = ((df['Age'] > 45) & (df['Age'] < 65)).astype(int)
    df_new['Age_risk_diabetes'] = (df['Age'] > 40).astype(int)
    df_new['Age_risk_hypertension'] = (df['Age'] > 35).astype(int)
    
    # BMI categories (WHO classification)
    df_new['BMI_underweight'] = (df['BMI'] < 18.5).astype(int)
    df_new['BMI_normal'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).astype(int)
    df_new['BMI_overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
    df_new['BMI_obese_1'] = ((df['BMI'] >= 30) & (df['BMI'] < 35)).astype(int)
    df_new['BMI_obese_2'] = ((df['BMI'] >= 35) & (df['BMI'] < 40)).astype(int)
    df_new['BMI_obese_3'] = (df['BMI'] >= 40).astype(int)
    
    # Blood pressure categories (AHA guidelines)
    df_new['BP_normal'] = ((df['BP_Systolic'] < 120) & (df['BP_Diastolic'] < 80)).astype(int)
    df_new['BP_elevated'] = ((df['BP_Systolic'] >= 120) & (df['BP_Systolic'] < 130) & (df['BP_Diastolic'] < 80)).astype(int)
    df_new['BP_stage1'] = (((df['BP_Systolic'] >= 130) & (df['BP_Systolic'] < 140)) | ((df['BP_Diastolic'] >= 80) & (df['BP_Diastolic'] < 90))).astype(int)
    df_new['BP_stage2'] = ((df['BP_Systolic'] >= 140) | (df['BP_Diastolic'] >= 90)).astype(int)
    df_new['BP_crisis'] = ((df['BP_Systolic'] > 180) | (df['BP_Diastolic'] > 120)).astype(int)
    
    # Cholesterol categories
    df_new['Chol_desirable'] = (df['Cholesterol'] < 200).astype(int)
    df_new['Chol_borderline'] = ((df['Cholesterol'] >= 200) & (df['Cholesterol'] < 240)).astype(int)
    df_new['Chol_high'] = (df['Cholesterol'] >= 240).astype(int)
    
    # Glucose categories
    df_new['Glucose_normal'] = (df['Glucose'] < 100).astype(int)
    df_new['Glucose_prediabetes'] = ((df['Glucose'] >= 100) & (df['Glucose'] < 126)).astype(int)
    df_new['Glucose_diabetes'] = (df['Glucose'] >= 126).astype(int)
    
    # Complex interaction features
    df_new['Age_BMI_interaction'] = df['Age'] * df['BMI'] / 100
    df_new['Age_BP_interaction'] = df['Age'] * df['BP_Systolic'] / 100
    df_new['BMI_BP_interaction'] = df['BMI'] * df['BP_Systolic'] / 100
    df_new['Smoking_Age_interaction'] = df['Smoking'] * df['Age']
    df_new['Family_Age_interaction'] = df['Family_History'] * df['Age']
    df_new['Smoking_BMI_interaction'] = df['Smoking'] * df['BMI']
    
    # Disease-specific composite scores
    df_new['Heart_disease_score'] = (
        (df['Age'] > 50).astype(int) * 3 +
        (df['BMI'] > 30).astype(int) * 2 +
        (df['BP_Systolic'] > 140).astype(int) * 4 +
        (df['Cholesterol'] > 240).astype(int) * 3 +
        df['Smoking'] * 4 +
        df['Family_History'] * 2
    )
    
    df_new['Diabetes_score'] = (
        (df['Age'] > 45).astype(int) * 2 +
        (df['BMI'] > 25).astype(int) * 4 +
        (df['Glucose'] > 126).astype(int) * 5 +
        df['Family_History'] * 3
    )
    
    df_new['Hypertension_score'] = (
        (df['Age'] > 40).astype(int) * 2 +
        (df['BMI'] > 25).astype(int) * 2 +
        (df['BP_Systolic'] > 130).astype(int) * 5 +
        (df['BP_Diastolic'] > 80).astype(int) * 4 +
        df['Smoking'] * 2 +
        df['Family_History'] * 2
    )
    
    # Symptom encoding with more sophisticated approach
    symptom_encoder = LabelEncoder()
    df_new['Symptoms_encoded'] = symptom_encoder.fit_transform(df['Symptoms'])
    
    # Create binary features for each symptom
    unique_symptoms = df['Symptoms'].unique()
    for i, symptom in enumerate(unique_symptoms):
        df_new[f'Symptom_{i}'] = (df['Symptoms'] == symptom).astype(int)
    
    return df_new, symptom_encoder

def advanced_feature_selection(X, y, n_features=25):
    """Advanced feature selection using multiple methods"""
    
    # Method 1: Mutual Information
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_mi = mi_selector.fit_transform(X, y)
    mi_features = X.columns[mi_selector.get_support()].tolist()
    
    # Method 2: F-statistic
    f_selector = SelectKBest(score_func=f_classif, k=n_features)
    X_f = f_selector.fit_transform(X, y)
    f_features = X.columns[f_selector.get_support()].tolist()
    
    # Method 3: Recursive Feature Elimination
    rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    rfe_selector = RFE(rf_estimator, n_features_to_select=n_features)
    X_rfe = rfe_selector.fit_transform(X, y)
    rfe_features = X.columns[rfe_selector.get_support()].tolist()
    
    # Combine features from all methods
    all_selected = set(mi_features + f_features + rfe_features)
    final_features = list(all_selected)[:n_features]  # Take top n_features
    
    print(f"Selected {len(final_features)} features from {X.shape[1]} total features")
    
    return final_features

def hyperparameter_tuning(X_train, y_train):
    """Hyperparameter tuning for best models"""
    
    # XGBoost hyperparameter tuning
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [6, 8, 10],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(objective='multi:softprob', random_state=42)
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    
    print(f"Best XGBoost params: {xgb_grid.best_params_}")
    print(f"Best XGBoost CV score: {xgb_grid.best_score_:.4f}")
    
    # Random Forest hyperparameter tuning
    rf_params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_grid = GridSearchCV(rf_model, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    
    print(f"Best RF params: {rf_grid.best_params_}")
    print(f"Best RF CV score: {rf_grid.best_score_:.4f}")
    
    return xgb_grid.best_estimator_, rf_grid.best_estimator_

def train_advanced_models():
    """Train advanced models with comprehensive optimization"""
    print("Loading and preprocessing data...")
    df = pd.read_excel('../public/data/health_disease_dataset.xlsx')
    
    # Deep feature engineering
    df_enhanced, symptom_encoder = deep_feature_engineering(df)
    print(f"Enhanced dataset shape: {df_enhanced.shape}")
    
    # Prepare features and target
    feature_cols = [col for col in df_enhanced.columns if col not in ['Disease_Label', 'Symptoms']]
    X = df_enhanced[feature_cols]
    y = df_enhanced['Disease_Label']
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Advanced feature selection
    selected_features = advanced_feature_selection(X, y_encoded, n_features=30)
    X_selected = X[selected_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Try different scalers
    scalers = {
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'MinMaxScaler': MinMaxScaler()
    }
    
    best_scaler = None
    best_scaler_score = 0
    
    for scaler_name, scaler in scalers.items():
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Quick test with Random Forest
        rf_test = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_test.fit(X_train_scaled, y_train)
        score = rf_test.score(X_test_scaled, y_test)
        
        print(f"{scaler_name} test score: {score:.4f}")
        
        if score > best_scaler_score:
            best_scaler_score = score
            best_scaler = scaler
    
    print(f"Best scaler: {type(best_scaler).__name__}")
    
    # Use best scaler
    X_train_scaled = best_scaler.fit_transform(X_train)
    X_test_scaled = best_scaler.transform(X_test)
    
    # Try different sampling strategies
    sampling_strategies = {
        'SMOTE': SMOTE(random_state=42, k_neighbors=3),
        'ADASYN': ADASYN(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42)
    }
    
    best_sampler = None
    best_sampler_score = 0
    
    for sampler_name, sampler in sampling_strategies.items():
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
            
            # Quick test
            rf_test = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_test.fit(X_resampled, y_resampled)
            score = rf_test.score(X_test_scaled, y_test)
            
            print(f"{sampler_name} test score: {score:.4f}")
            
            if score > best_sampler_score:
                best_sampler_score = score
                best_sampler = sampler
        except:
            print(f"{sampler_name} failed")
    
    # Use best sampling strategy
    if best_sampler:
        X_train_final, y_train_final = best_sampler.fit_resample(X_train_scaled, y_train)
        print(f"Best sampler: {type(best_sampler).__name__}")
    else:
        X_train_final, y_train_final = X_train_scaled, y_train
    
    # Hyperparameter tuning
    print("Performing hyperparameter tuning...")
    best_xgb, best_rf = hyperparameter_tuning(X_train_final, y_train_final)
    
    # Train additional models
    models = {
        'Tuned_XGBoost': best_xgb,
        'Tuned_RandomForest': best_rf,
        'ExtraTrees': ExtraTreesClassifier(n_estimators=300, max_depth=15, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, max_depth=10, random_state=42, verbose=-1)
    }
    
    # Train and evaluate all models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_final, y_train_final)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Find best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_accuracy = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name} with {best_accuracy:.4f} accuracy")
    
    # Final evaluation
    y_pred_final = best_model.predict(X_test_scaled)
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred_final, target_names=label_encoder.classes_))
    
    # Save everything
    joblib.dump(best_model, 'models/optimized_best_model.pkl')
    joblib.dump(best_scaler, 'models/optimized_scaler.pkl')
    joblib.dump(label_encoder, 'models/optimized_label_encoder.pkl')
    joblib.dump(symptom_encoder, 'models/optimized_symptom_encoder.pkl')
    
    with open('models/optimized_features.txt', 'w') as f:
        f.write('\\n'.join(selected_features))
    
    print("Optimized models saved!")
    
    return best_model, best_accuracy

if __name__ == "__main__":
    print("Starting Advanced Multiclass Optimization...")
    best_model, accuracy = train_advanced_models()
    print(f"\nFinal Optimized Accuracy: {accuracy:.4f}")
    print("Advanced optimization complete!")