import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

def analyze_new_dataset():
    """Analyze the health disease dataset"""
    # Load the new dataset
    df = pd.read_excel('../public/data/health_disease_dataset.xlsx')
    
    print("=== DATASET ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())
    
    print("\n=== BASIC STATISTICS ===")
    print(df.describe())
    
    # Check target variable
    target_cols = [col for col in df.columns if 'target' in col.lower() or 'disease' in col.lower() or 'heart' in col.lower()]
    if target_cols:
        target_col = target_cols[0]
        print(f"\n=== TARGET VARIABLE: {target_col} ===")
        print(df[target_col].value_counts())
        print(f"Class distribution: {df[target_col].value_counts(normalize=True)}")
    else:
        # Assume last column is target
        target_col = df.columns[-1]
        print(f"\n=== ASSUMED TARGET: {target_col} ===")
        print(df[target_col].value_counts())
    
    print("\n=== CATEGORICAL FEATURES ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values")
        if df[col].nunique() < 10:
            print(f"  Values: {df[col].unique()}")
    
    print("\n=== NUMERICAL FEATURES ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != target_col:
            print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    return df, target_col

def feature_importance_analysis(df, target_col):
    """Analyze feature importance"""
    # Prepare data
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Handle categorical variables
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mi_scores
    }).sort_values('importance', ascending=False)
    
    print("\n=== FEATURE IMPORTANCE (Mutual Information) ===")
    print(feature_importance.head(10))
    
    return feature_importance

if __name__ == "__main__":
    df, target_col = analyze_new_dataset()
    feature_importance = feature_importance_analysis(df, target_col)