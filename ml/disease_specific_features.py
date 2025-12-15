import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def create_disease_specific_features(df):
    """Create features specific to each disease type"""
    df_enhanced = df.copy()
    
    # Heart Disease specific features
    df_enhanced['heart_risk_score'] = (
        (df['Age'] > 50).astype(int) * 2 +
        (df['BMI'] > 30).astype(int) * 2 +
        (df['BP_Systolic'] > 140).astype(int) * 3 +
        (df['Cholesterol'] > 240).astype(int) * 2 +
        df['Smoking'] * 3 +
        df['Family_History'] * 2
    )
    
    # Diabetes specific features
    df_enhanced['diabetes_risk_score'] = (
        (df['Age'] > 45).astype(int) * 2 +
        (df['BMI'] > 25).astype(int) * 3 +
        (df['Glucose'] > 126).astype(int) * 4 +
        df['Family_History'] * 2
    )
    
    # Hypertension specific features
    df_enhanced['hypertension_risk_score'] = (
        (df['Age'] > 40).astype(int) * 2 +
        (df['BMI'] > 25).astype(int) * 2 +
        (df['BP_Systolic'] > 130).astype(int) * 4 +
        (df['BP_Diastolic'] > 80).astype(int) * 3 +
        df['Smoking'] * 2
    )
    
    # Kidney Disease specific features
    df_enhanced['kidney_risk_score'] = (
        (df['Age'] > 60).astype(int) * 2 +
        (df['BP_Systolic'] > 140).astype(int) * 3 +
        (df['Glucose'] > 140).astype(int) * 2 +
        df['Family_History'] * 2
    )
    
    # Liver Disease specific features
    df_enhanced['liver_risk_score'] = (
        (df['Age'] > 40).astype(int) * 1 +
        (df['BMI'] > 30).astype(int) * 2 +
        df['Smoking'] * 3
    )
    
    # Asthma specific features
    df_enhanced['asthma_risk_score'] = (
        (df['Age'] < 40).astype(int) * 1 +
        df['Family_History'] * 3 +
        df['Smoking'] * 2
    )
    
    # Advanced BMI categories
    df_enhanced['BMI_severe_obesity'] = (df['BMI'] > 35).astype(int)
    df_enhanced['BMI_underweight'] = (df['BMI'] < 18.5).astype(int)
    
    # Blood pressure categories
    df_enhanced['BP_stage1_hypertension'] = ((df['BP_Systolic'] >= 130) | (df['BP_Diastolic'] >= 80)).astype(int)
    df_enhanced['BP_stage2_hypertension'] = ((df['BP_Systolic'] >= 140) | (df['BP_Diastolic'] >= 90)).astype(int)
    df_enhanced['BP_crisis'] = ((df['BP_Systolic'] > 180) | (df['BP_Diastolic'] > 120)).astype(int)
    
    # Cholesterol categories
    df_enhanced['cholesterol_borderline'] = ((df['Cholesterol'] >= 200) & (df['Cholesterol'] < 240)).astype(int)
    df_enhanced['cholesterol_high'] = (df['Cholesterol'] >= 240).astype(int)
    
    # Glucose categories
    df_enhanced['glucose_prediabetes'] = ((df['Glucose'] >= 100) & (df['Glucose'] < 126)).astype(int)
    df_enhanced['glucose_diabetes'] = (df['Glucose'] >= 126).astype(int)
    
    # Age-related features
    df_enhanced['age_young'] = (df['Age'] < 30).astype(int)
    df_enhanced['age_middle'] = ((df['Age'] >= 30) & (df['Age'] < 60)).astype(int)
    df_enhanced['age_senior'] = (df['Age'] >= 60).astype(int)
    
    # Interaction features
    df_enhanced['age_bmi_interaction'] = df['Age'] * df['BMI'] / 100
    df_enhanced['smoking_age_interaction'] = df['Smoking'] * df['Age']
    df_enhanced['family_age_interaction'] = df['Family_History'] * df['Age']
    df_enhanced['bp_age_interaction'] = (df['BP_Systolic'] + df['BP_Diastolic']) * df['Age'] / 1000
    
    # Symptom-based features
    symptom_disease_map = {
        'chest pain, fatigue': 'heart_disease',
        'shortness of breath': 'heart_disease', 
        'headache, dizziness': 'hypertension',
        'frequent urination, thirst': 'diabetes',
        'abdominal pain, nausea': 'liver_disease',
        'leg swelling, fatigue': 'kidney_disease',
        'coughing, wheezing': 'asthma',
        'blurred vision, weakness': 'diabetes'
    }
    
    for symptom, disease in symptom_disease_map.items():
        df_enhanced[f'symptom_indicates_{disease}'] = (df['Symptoms'] == symptom).astype(int)
    
    return df_enhanced

def select_optimal_features(df, target_col, method='mutual_info', k=20):
    """Select optimal features using different methods"""
    X = df.drop([target_col, 'Symptoms'], axis=1)
    y = df[target_col]
    
    # Encode target if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y
        le = None
    
    if method == 'mutual_info':
        # Mutual information for feature selection
        mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': mi_scores
        }).sort_values('score', ascending=False)
        
    elif method == 'f_classif':
        # F-statistic for feature selection
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y_encoded)
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
    
    # Select top k features
    selected_features = feature_scores.head(k)['feature'].tolist()
    
    print(f"Top {k} features selected using {method}:")
    for i, (_, row) in enumerate(feature_scores.head(k).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:30s} - Score: {row['score']:.4f}")
    
    return selected_features, feature_scores, le

def analyze_feature_importance():
    """Analyze feature importance for the health disease dataset"""
    # Load data
    df = pd.read_excel('../public/data/health_disease_dataset.xlsx')
    
    # Create enhanced features
    df_enhanced = create_disease_specific_features(df)
    
    print(f"Original features: {len(df.columns)}")
    print(f"Enhanced features: {len(df_enhanced.columns)}")
    
    # Feature selection using mutual information
    selected_features_mi, scores_mi, le = select_optimal_features(
        df_enhanced, 'Disease_Label', method='mutual_info', k=25
    )
    
    print("\n" + "="*50)
    
    # Feature selection using F-statistic
    selected_features_f, scores_f, _ = select_optimal_features(
        df_enhanced, 'Disease_Label', method='f_classif', k=25
    )
    
    # Find common features
    common_features = list(set(selected_features_mi) & set(selected_features_f))
    print(f"\nCommon features in both methods: {len(common_features)}")
    for feature in common_features:
        print(f"  - {feature}")
    
    return df_enhanced, selected_features_mi, selected_features_f, common_features

if __name__ == "__main__":
    df_enhanced, mi_features, f_features, common_features = analyze_feature_importance()