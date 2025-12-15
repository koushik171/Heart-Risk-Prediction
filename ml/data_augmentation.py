import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def handle_class_imbalance(X, y, method='smote'):
    """Handle class imbalance using various techniques"""
    
    print(f"Original class distribution: {np.bincount(y)}")
    
    if method == 'smote':
        sampler = SMOTE(random_state=42)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        return X, y
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled

def synthetic_data_generation(df, n_samples=1000):
    """Generate synthetic data using statistical methods"""
    synthetic_data = []
    
    for _ in range(n_samples):
        synthetic_row = {}
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # For numerical columns, use normal distribution
                mean = df[column].mean()
                std = df[column].std()
                synthetic_value = np.random.normal(mean, std)
                
                # Ensure values are within reasonable bounds
                min_val, max_val = df[column].min(), df[column].max()
                synthetic_value = np.clip(synthetic_value, min_val, max_val)
                
                if df[column].dtype == 'int64':
                    synthetic_value = int(round(synthetic_value))
                
                synthetic_row[column] = synthetic_value
            else:
                # For categorical columns, sample from existing values
                synthetic_row[column] = np.random.choice(df[column].values)
        
        synthetic_data.append(synthetic_row)
    
    return pd.DataFrame(synthetic_data)

def train_with_augmented_data():
    """Train model with data augmentation techniques"""
    # Load original data
    df = pd.read_excel('../public/data/heart_risk_dataset.xlsx')
    df = df.dropna()
    
    print(f"Original dataset size: {df.shape}")
    
    # Generate synthetic data
    synthetic_df = synthetic_data_generation(df, n_samples=500)
    
    # Combine original and synthetic data
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    print(f"Augmented dataset size: {augmented_df.shape}")
    
    # Prepare features and target
    X = augmented_df.drop('target', axis=1) if 'target' in augmented_df.columns else augmented_df.iloc[:, :-1]
    y = augmented_df['target'] if 'target' in augmented_df.columns else augmented_df.iloc[:, -1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    X_train_balanced, y_train_balanced = handle_class_imbalance(
        X_train_scaled, y_train, method='smote'
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
    
    print("Training model with augmented data...")
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Augmented Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'models/augmented_model.pkl')
    joblib.dump(scaler, 'models/augmented_scaler.pkl')
    
    return model, scaler

def compare_sampling_methods():
    """Compare different sampling methods"""
    # Load data
    df = pd.read_excel('../public/data/heart_risk_dataset.xlsx')
    df = df.dropna()
    
    X = df.drop('target', axis=1) if 'target' in df.columns else df.iloc[:, :-1]
    y = df['target'] if 'target' in df.columns else df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    methods = ['none', 'smote', 'adasyn', 'smote_tomek']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method.upper()} method...")
        
        if method == 'none':
            X_resampled, y_resampled = X_train_scaled, y_train
        else:
            X_resampled, y_resampled = handle_class_imbalance(
                X_train_scaled, y_train, method=method
            )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_resampled, y_resampled)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[method] = accuracy
        print(f"{method.upper()} Accuracy: {accuracy:.4f}")
    
    # Find best method
    best_method = max(results, key=results.get)
    print(f"\nBest sampling method: {best_method.upper()} with accuracy: {results[best_method]:.4f}")
    
    return results

if __name__ == "__main__":
    print("Comparing sampling methods...")
    comparison_results = compare_sampling_methods()
    
    print("\nTraining with data augmentation...")
    model, scaler = train_with_augmented_data()
    
    print("Data augmentation training complete!")