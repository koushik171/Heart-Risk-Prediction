import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

def create_keras_model(input_dim):
    """Create deep neural network with Keras"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc']
    )
    
    return model

def train_neural_networks():
    """Train both sklearn MLP and Keras deep learning models"""
    # Load data
    df = pd.read_excel('../public/data/heart_risk_dataset.xlsx')
    df = df.dropna()
    
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
    
    # 1. Sklearn MLP Classifier
    print("Training MLP Classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)
    mlp_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]
    
    mlp_accuracy = accuracy_score(y_test, mlp_pred)
    mlp_auc = roc_auc_score(y_test, mlp_pred_proba)
    
    print(f"MLP Accuracy: {mlp_accuracy:.4f}")
    print(f"MLP AUC: {mlp_auc:.4f}")
    
    # 2. Keras Deep Learning Model
    print("\nTraining Keras Deep Neural Network...")
    keras_model = create_keras_model(X_train_scaled.shape[1])
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Train model
    history = keras_model.fit(
        X_train_scaled, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate Keras model
    keras_pred_proba = keras_model.predict(X_test_scaled).flatten()
    keras_pred = (keras_pred_proba > 0.5).astype(int)
    
    keras_accuracy = accuracy_score(y_test, keras_pred)
    keras_auc = roc_auc_score(y_test, keras_pred_proba)
    
    print(f"Keras NN Accuracy: {keras_accuracy:.4f}")
    print(f"Keras NN AUC: {keras_auc:.4f}")
    
    # Save models
    joblib.dump(mlp, 'models/mlp_model.pkl')
    keras_model.save('models/keras_model.h5')
    joblib.dump(scaler, 'models/nn_scaler.pkl')
    
    # Compare models
    print("\n=== Model Comparison ===")
    print(f"MLP Classifier: Accuracy={mlp_accuracy:.4f}, AUC={mlp_auc:.4f}")
    print(f"Keras Deep NN: Accuracy={keras_accuracy:.4f}, AUC={keras_auc:.4f}")
    
    return mlp, keras_model, scaler

if __name__ == "__main__":
    mlp_model, keras_model, scaler = train_neural_networks()
    print("Neural network training complete!")