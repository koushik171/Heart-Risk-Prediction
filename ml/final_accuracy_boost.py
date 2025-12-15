import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality():
    """Analyze data quality and class separability"""
    df = pd.read_excel('../public/data/health_disease_dataset.xlsx')
    
    print("=== DATA QUALITY ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    
    # Check class balance
    print("\\nClass distribution:")
    class_counts = df['Disease_Label'].value_counts()
    print(class_counts)
    print(f"Class balance ratio: {class_counts.min() / class_counts.max():.3f}")
    
    # Analyze feature-target relationships
    print("\\n=== FEATURE-TARGET CORRELATION ===")
    
    # Encode categorical variables for correlation analysis
    df_encoded = df.copy()
    le_symptoms = LabelEncoder()
    df_encoded['Symptoms_encoded'] = le_symptoms.fit_transform(df['Symptoms'])
    le_disease = LabelEncoder()
    df_encoded['Disease_encoded'] = le_disease.fit_transform(df['Disease_Label'])
    
    # Calculate correlations
    numeric_cols = ['Age', 'Height_cm', 'Weight_kg', 'BMI', 'BP_Systolic', 
                   'BP_Diastolic', 'Cholesterol', 'Glucose', 'Smoking', 
                   'Family_History', 'Symptoms_encoded']
    
    correlations = []
    for col in numeric_cols:
        corr = df_encoded[col].corr(df_encoded['Disease_encoded'])
        correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("Feature correlations with target (absolute values):")
    for feature, corr in correlations:
        print(f"{feature:20s}: {corr:.4f}")
    
    return df, correlations

def create_robust_features(df):
    """Create robust features based on medical knowledge"""
    df_new = df.copy()
    
    # Medical risk scores based on established guidelines
    
    # Framingham Risk Score components
    df_new['framingham_age_points'] = np.where(df['Age'] < 35, 0,
                                     np.where(df['Age'] < 40, 1,
                                     np.where(df['Age'] < 45, 2,
                                     np.where(df['Age'] < 50, 3,
                                     np.where(df['Age'] < 55, 4,
                                     np.where(df['Age'] < 60, 5,
                                     np.where(df['Age'] < 65, 6, 7)))))))
    
    # BMI risk categories
    df_new['bmi_risk_category'] = np.where(df['BMI'] < 18.5, 1,  # Underweight
                                 np.where(df['BMI'] < 25, 0,    # Normal
                                 np.where(df['BMI'] < 30, 2,    # Overweight
                                 np.where(df['BMI'] < 35, 3,    # Obese I
                                 np.where(df['BMI'] < 40, 4, 5))))) # Obese II+
    
    # Blood pressure risk
    df_new['bp_risk'] = np.where((df['BP_Systolic'] < 120) & (df['BP_Diastolic'] < 80), 0,  # Normal
                       np.where((df['BP_Systolic'] < 130) & (df['BP_Diastolic'] < 80), 1,  # Elevated
                       np.where((df['BP_Systolic'] < 140) | (df['BP_Diastolic'] < 90), 2,  # Stage 1
                       np.where((df['BP_Systolic'] < 180) | (df['BP_Diastolic'] < 120), 3, 4))))  # Stage 2+
    
    # Cholesterol risk
    df_new['chol_risk'] = np.where(df['Cholesterol'] < 200, 0,  # Desirable
                         np.where(df['Cholesterol'] < 240, 1,  # Borderline
                         2))  # High
    
    # Glucose risk
    df_new['glucose_risk'] = np.where(df['Glucose'] < 100, 0,  # Normal
                            np.where(df['Glucose'] < 126, 1,  # Prediabetes
                            2))  # Diabetes
    
    # Composite risk scores
    df_new['cardiovascular_risk'] = (
        df_new['framingham_age_points'] +
        df_new['bmi_risk_category'] +
        df_new['bp_risk'] +
        df_new['chol_risk'] +
        df['Smoking'] * 2 +
        df['Family_History'] * 2
    )
    
    df_new['metabolic_risk'] = (
        df_new['bmi_risk_category'] +
        df_new['glucose_risk'] +
        df_new['bp_risk'] +
        df['Family_History']
    )
    
    # Symptom-disease mapping based on medical knowledge
    symptom_mapping = {
        'chest pain, fatigue': [1, 0, 0, 0, 0, 0],  # Heart Disease
        'shortness of breath': [1, 0, 1, 0, 0, 0],  # Heart Disease, Asthma
        'headache, dizziness': [0, 1, 0, 0, 0, 0],  # Hypertension
        'frequent urination, thirst': [0, 0, 0, 1, 0, 0],  # Diabetes
        'abdominal pain, nausea': [0, 0, 0, 0, 1, 0],  # Liver Disease
        'leg swelling, fatigue': [0, 0, 0, 0, 0, 1],  # Kidney Disease
        'coughing, wheezing': [0, 0, 1, 0, 0, 0],  # Asthma
        'blurred vision, weakness': [0, 0, 0, 1, 0, 0]  # Diabetes
    }
    
    disease_names = ['Heart Disease', 'Hypertension', 'Asthma', 'Diabetes', 'Liver Disease', 'Kidney Disease']
    
    for i, disease in enumerate(disease_names):
        df_new[f'symptom_match_{disease.lower().replace(" ", "_")}'] = 0
        for symptom, matches in symptom_mapping.items():
            mask = df['Symptoms'] == symptom
            df_new.loc[mask, f'symptom_match_{disease.lower().replace(" ", "_")}'] = matches[i]
    
    # Encode original symptoms
    le_symptoms = LabelEncoder()
    df_new['symptoms_encoded'] = le_symptoms.fit_transform(df['Symptoms'])
    
    return df_new, le_symptoms

def create_mega_ensemble():
    """Create a mega ensemble with diverse algorithms"""
    
    # Base classifiers with different strengths
    classifiers = [
        ('rf1', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        ('rf2', RandomForestClassifier(n_estimators=300, max_depth=15, random_state=43)),
        ('xgb1', xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42)),
        ('xgb2', xgb.XGBClassifier(n_estimators=150, learning_rate=0.2, max_depth=6, random_state=43)),
        ('lgb1', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42, verbose=-1)),
        ('lgb2', lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, max_depth=10, random_state=43, verbose=-1)),
        ('dt1', DecisionTreeClassifier(max_depth=15, random_state=42)),
        ('dt2', DecisionTreeClassifier(max_depth=20, random_state=43)),
        ('knn1', KNeighborsClassifier(n_neighbors=5)),
        ('knn2', KNeighborsClassifier(n_neighbors=7)),
        ('nb', GaussianNB()),
        ('svm', SVC(probability=True, random_state=42))
    ]
    
    # Create voting ensemble
    voting_ensemble = VotingClassifier(
        estimators=classifiers,
        voting='soft',
        n_jobs=-1
    )
    
    return voting_ensemble

def train_final_model():
    """Train the final optimized model"""
    print("Starting Final Accuracy Optimization...")
    
    # Analyze data quality first
    df, correlations = analyze_data_quality()
    
    # Create robust features
    df_enhanced, le_symptoms = create_robust_features(df)
    
    print(f"\\nEnhanced dataset shape: {df_enhanced.shape}")
    
    # Prepare features (exclude original categorical columns)
    feature_cols = [col for col in df_enhanced.columns 
                   if col not in ['Disease_Label', 'Symptoms']]
    
    X = df_enhanced[feature_cols]
    y = df_enhanced['Disease_Label']
    
    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for class balance
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Balanced training set shape: {X_train_balanced.shape}")
    print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")
    
    # Create and train mega ensemble
    print("\\nTraining Mega Ensemble...")
    mega_ensemble = create_mega_ensemble()
    mega_ensemble.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    y_pred = mega_ensemble.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\\nMega Ensemble Accuracy: {accuracy:.4f}")
    
    # Cross-validation
    print("\\nPerforming Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_encoded)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y_encoded[train_idx], y_encoded[val_idx]
        
        # Scale
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = scaler.transform(X_fold_val)
        
        # Balance
        X_fold_balanced, y_fold_balanced = smote.fit_resample(X_fold_train_scaled, y_fold_train)
        
        # Train
        fold_model = create_mega_ensemble()
        fold_model.fit(X_fold_balanced, y_fold_balanced)
        
        # Evaluate
        fold_score = fold_model.score(X_fold_val_scaled, y_fold_val)
        cv_scores.append(fold_score)
        
        print(f"Fold {fold + 1}: {fold_score:.4f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\\nCross-Validation Results:")
    print(f"Mean Accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    
    # Detailed classification report
    print("\\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    
    # Save final model
    joblib.dump(mega_ensemble, 'models/final_mega_ensemble.pkl')
    joblib.dump(scaler, 'models/final_scaler.pkl')
    joblib.dump(le_target, 'models/final_target_encoder.pkl')
    joblib.dump(le_symptoms, 'models/final_symptom_encoder.pkl')
    
    with open('models/final_features.txt', 'w') as f:
        f.write('\\n'.join(feature_cols))
    
    print("\\nFinal model saved!")
    
    return mega_ensemble, accuracy, cv_mean

if __name__ == "__main__":
    model, test_acc, cv_acc = train_final_model()
    
    print(f"\\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"CV Accuracy: {cv_acc:.4f}")
    print(f"{'='*50}")
    
    if test_acc > 0.25:  # If we achieve >25% accuracy (better than random for 6 classes)
        print("SUCCESS: Achieved above-random performance!")
    else:
        print("NOTE: Low accuracy suggests challenging dataset or need for domain expertise")