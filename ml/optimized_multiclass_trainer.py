import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_features(df):
    """Create disease-specific features"""
    df_new = df.copy()
    
    # Disease-specific risk scores
    df_new['heart_risk'] = (
        (df['Age'] > 50).astype(int) * 2 +
        (df['BMI'] > 30).astype(int) * 2 +
        (df['BP_Systolic'] > 140).astype(int) * 3 +
        (df['Cholesterol'] > 240).astype(int) * 2 +
        df['Smoking'] * 3 +
        df['Family_History'] * 2
    )
    
    df_new['diabetes_risk'] = (
        (df['Age'] > 45).astype(int) * 2 +
        (df['BMI'] > 25).astype(int) * 3 +
        (df['Glucose'] > 126).astype(int) * 4 +
        df['Family_History'] * 2
    )
    
    df_new['hypertension_risk'] = (
        (df['Age'] > 40).astype(int) * 2 +
        (df['BMI'] > 25).astype(int) * 2 +
        (df['BP_Systolic'] > 130).astype(int) * 4 +
        (df['BP_Diastolic'] > 80).astype(int) * 3
    )
    
    # BMI categories
    df_new['BMI_obese'] = (df['BMI'] > 30).astype(int)
    df_new['BMI_overweight'] = ((df['BMI'] >= 25) & (df['BMI'] <= 30)).astype(int)
    
    # Blood pressure categories
    df_new['BP_high'] = (df['BP_Systolic'] > 140).astype(int)
    df_new['BP_ratio'] = df['BP_Systolic'] / df['BP_Diastolic']
    
    # Cholesterol and glucose categories
    df_new['Chol_high'] = (df['Cholesterol'] > 240).astype(int)
    df_new['Glucose_high'] = (df['Glucose'] > 140).astype(int)
    
    # Age groups
    df_new['Age_senior'] = (df['Age'] > 60).astype(int)
    df_new['Age_middle'] = ((df['Age'] >= 40) & (df['Age'] <= 60)).astype(int)
    
    # Interaction features
    df_new['Age_BMI'] = df['Age'] * df['BMI'] / 100
    df_new['Smoking_Age'] = df['Smoking'] * df['Age']
    
    # Symptom encoding
    symptom_encoder = LabelEncoder()
    df_new['Symptoms_encoded'] = symptom_encoder.fit_transform(df['Symptoms'])
    
    return df_new, symptom_encoder

def train_optimized_models():
    """Train optimized models for multiclass disease prediction"""
    print("Loading Health Disease Dataset...")
    df = pd.read_excel('../public/data/health_disease_dataset.xlsx')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {df['Disease_Label'].unique()}")
    print(f"Class distribution:")
    print(df['Disease_Label'].value_counts())
    
    # Create enhanced features
    df_enhanced, symptom_encoder = create_enhanced_features(df)
    
    # Prepare features and target
    feature_cols = [col for col in df_enhanced.columns if col not in ['Disease_Label', 'Symptoms']]
    X = df_enhanced[feature_cols]
    y = df_enhanced['Disease_Label']
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector.fit_transform(X, y_encoded)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Selected {len(selected_features)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Balanced classes: {np.bincount(y_train_balanced)}")
    
    # Define models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=3,
            class_weight='balanced', random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=8,
            objective='multi:softprob', random_state=42
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=8,
            objective='multiclass', class_weight='balanced',
            random_state=42, verbose=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42
        )
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_balanced, y_train_balanced)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = accuracy
        trained_models[name] = model
        
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Create voting ensemble
    print("Training Voting Ensemble...")
    voting_ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )
    voting_ensemble.fit(X_train_balanced, y_train_balanced)
    
    ensemble_pred = voting_ensemble.predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    
    # Find best model
    best_model_name = max(results, key=results.get)
    best_accuracy = max(results.values())
    
    if ensemble_accuracy > best_accuracy:
        best_model = voting_ensemble
        best_accuracy = ensemble_accuracy
        best_model_name = "Voting Ensemble"
    else:
        best_model = trained_models[best_model_name]
    
    print(f"Best Model: {best_model_name} with {best_accuracy:.4f} accuracy")
    
    # Detailed evaluation
    best_pred = best_model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred, target_names=label_encoder.classes_))
    
    # Cross-validation
    print("\nPerforming 5-fold Cross-Validation...")
    cv_scores = cross_val_score(best_model, X_selected, y_encoded, cv=5, scoring='accuracy')
    print(f"CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save models
    joblib.dump(best_model, 'models/best_multiclass_model.pkl')
    joblib.dump(scaler, 'models/multiclass_scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    joblib.dump(symptom_encoder, 'models/symptom_encoder.pkl')
    joblib.dump(selector, 'models/multiclass_selector.pkl')
    
    with open('models/multiclass_features.txt', 'w') as f:
        f.write('\\n'.join(selected_features))
    
    print("Models saved successfully!")
    
    return best_model, best_accuracy, cv_scores.mean()

if __name__ == "__main__":
    print("Starting Multiclass Disease Prediction Optimization...")
    best_model, test_accuracy, cv_accuracy = train_optimized_models()
    
    print(f"\nFINAL RESULTS:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Cross-Validation Accuracy: {cv_accuracy:.4f}")
    print("Optimization Complete!")