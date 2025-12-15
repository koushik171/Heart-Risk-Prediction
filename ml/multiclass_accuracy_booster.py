import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def load_and_preprocess_data():
    """Load and preprocess the health disease dataset"""
    df = pd.read_excel('../public/data/health_disease_dataset.xlsx')
    
    # Feature engineering based on dataset analysis
    df['BMI_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])
    df['BP_ratio'] = df['BP_Systolic'] / df['BP_Diastolic']
    df['Cholesterol_risk'] = (df['Cholesterol'] > 240).astype(int)
    df['Glucose_risk'] = (df['Glucose'] > 140).astype(int)
    df['Weight_Height_ratio'] = df['Weight_kg'] / df['Height_cm']
    
    # Encode symptoms (most important feature)
    symptom_encoder = LabelEncoder()
    df['Symptoms_encoded'] = symptom_encoder.fit_transform(df['Symptoms'])
    
    # Create symptom-specific features
    symptom_mapping = {
        'chest pain, fatigue': 'heart_symptoms',
        'shortness of breath': 'respiratory_symptoms', 
        'headache, dizziness': 'hypertension_symptoms',
        'frequent urination, thirst': 'diabetes_symptoms',
        'abdominal pain, nausea': 'liver_symptoms',
        'leg swelling, fatigue': 'kidney_symptoms',
        'coughing, wheezing': 'asthma_symptoms',
        'blurred vision, weakness': 'diabetes_symptoms'
    }
    
    for symptom, category in symptom_mapping.items():
        df[f'has_{category}'] = (df['Symptoms'] == symptom).astype(int)
    
    # Interaction features
    df['Age_BMI'] = df['Age'] * df['BMI'] / 100
    df['Smoking_Age'] = df['Smoking'] * df['Age']
    df['Family_Age'] = df['Family_History'] * df['Age']
    
    return df, symptom_encoder

def create_multiclass_models():
    """Create models optimized for multiclass classification"""
    models = {}
    
    # Random Forest with multiclass optimization
    models['rf_multiclass'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    # XGBoost for multiclass
    models['xgb_multiclass'] = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        random_state=42
    )
    
    # LightGBM for multiclass
    models['lgb_multiclass'] = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multiclass',
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    # Gradient Boosting
    models['gb_multiclass'] = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        random_state=42
    )
    
    return models

def train_multiclass_ensemble():
    """Train ensemble for multiclass disease prediction"""
    print("🏥 Loading Health Disease Dataset...")
    df, symptom_encoder = load_and_preprocess_data()
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Classes: {df['Disease_Label'].unique()}")
    print(f"📈 Class distribution:\n{df['Disease_Label'].value_counts()}")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['Disease_Label', 'Symptoms']]
    X = df[feature_cols]
    y = df['Disease_Label']
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Feature selection - select top features
    selector = SelectKBest(score_func=f_classif, k=15)
    X_selected = selector.fit_transform(X, y_encoded)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"✨ Selected features: {selected_features}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"⚖️ Balanced classes: {np.bincount(y_train_balanced)}")
    
    # Create and train models
    models = create_multiclass_models()
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        model.fit(X_train_balanced, y_train_balanced)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        trained_models[name] = model
        results[name] = accuracy
        
        print(f"📈 {name} Accuracy: {accuracy:.4f}")
    
    # Create voting ensemble
    print("🔄 Training Voting Ensemble...")
    voting_ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )
    voting_ensemble.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate ensemble
    ensemble_pred = voting_ensemble.predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"🏆 Ensemble Accuracy: {ensemble_accuracy:.4f}")
    
    # Find best model
    best_model_name = max(results, key=results.get)
    best_accuracy = max(results.values())
    
    if ensemble_accuracy > best_accuracy:
        best_model = voting_ensemble
        best_accuracy = ensemble_accuracy
        best_model_name = "Voting Ensemble"
    else:
        best_model = trained_models[best_model_name]
    
    print(f"🎯 Best Model: {best_model_name} with {best_accuracy:.4f} accuracy")
    
    # Detailed evaluation
    best_pred = best_model.predict(X_test_scaled)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, best_pred, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Best Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('multiclass_confusion_matrix.png')
    plt.show()
    
    # Save best model and encoders
    joblib.dump(best_model, 'models/best_multiclass_model.pkl')
    joblib.dump(scaler, 'models/multiclass_scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    joblib.dump(symptom_encoder, 'models/symptom_encoder.pkl')
    joblib.dump(selector, 'models/multiclass_selector.pkl')
    
    # Save selected features
    with open('models/multiclass_features.txt', 'w') as f:
        f.write('\n'.join(selected_features))
    
    print("💾 Best model and encoders saved!")
    
    return best_model, best_accuracy, label_encoder

def cross_validate_best_model():
    """Perform cross-validation on the best model"""
    print("🔄 Performing Cross-Validation...")
    
    df, _ = load_and_preprocess_data()
    feature_cols = [col for col in df.columns if col not in ['Disease_Label', 'Symptoms']]
    X = df[feature_cols]
    y = df['Disease_Label']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Use the best model configuration
    best_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        random_state=42
    )
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(best_model, X, y_encoded, cv=5, scoring='accuracy')
    
    print(f"📊 Cross-Validation Results:")
    print(f"   Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"   Std Deviation: {cv_scores.std():.4f}")
    print(f"   Individual Scores: {cv_scores}")
    
    return cv_scores

if __name__ == "__main__":
    print("🚀 Starting Multiclass Disease Prediction Accuracy Improvement...")
    
    # Train best model
    best_model, accuracy, label_encoder = train_multiclass_ensemble()
    
    # Cross-validation
    cv_scores = cross_validate_best_model()
    
    print(f"\n🎉 FINAL RESULTS:")
    print(f"🎯 Best Test Accuracy: {accuracy:.4f}")
    print(f"📊 Cross-Validation Mean: {cv_scores.mean():.4f}")
    print(f"💾 Models saved in 'models/' directory")
    print(f"📈 Confusion matrix saved as 'multiclass_confusion_matrix.png'")