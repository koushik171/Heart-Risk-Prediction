import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_models():
    """Load trained models"""
    try:
        rf_model = joblib.load('models/random_forest_model.pkl')
        lr_model = joblib.load('models/logistic_regression_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return rf_model, lr_model, scaler
    except:
        print("Models not found. Run train_model.py first.")
        return None, None, None

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{model_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png')
    plt.show()
    
    return metrics

def compare_models():
    """Compare all trained models"""
    # Load data
    df = pd.read_excel('../public/data/heart_risk_dataset.xlsx')
    df = df.dropna()
    
    X = df.drop('target', axis=1) if 'target' in df.columns else df.iloc[:, :-1]
    y = df['target'] if 'target' in df.columns else df.iloc[:, -1]
    
    # Load models
    rf_model, lr_model, scaler = load_models()
    if rf_model is None:
        return
    
    # Split data (same as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    
    # Evaluate models
    rf_metrics = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    lr_metrics = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    
    # Create comparison chart
    metrics_df = pd.DataFrame({
        'Random Forest': rf_metrics,
        'Logistic Regression': lr_metrics
    })
    
    plt.figure(figsize=(10, 6))
    metrics_df.T.plot(kind='bar')
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

if __name__ == "__main__":
    compare_models()