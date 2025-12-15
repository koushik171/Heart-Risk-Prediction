"""Configuration settings for the ML pipeline"""

# Data paths
DATA_PATH = '../public/data/heart_risk_dataset.xlsx'
MODEL_DIR = 'models/'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_STATE
}

# Logistic Regression parameters
LR_PARAMS = {
    'random_state': RANDOM_STATE,
    'max_iter': 1000
}

# Feature engineering settings
FEATURE_SELECTION_K = 10
AGE_BINS = [0, 30, 45, 60, 100]
BMI_BINS = [0, 18.5, 25, 30, 100]

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = True

# Risk thresholds
RISK_THRESHOLDS = {
    'low': 40,
    'medium': 70
}

# Feature mappings for frontend
SYMPTOM_FEATURE_MAP = {
    'chest_pain': 'cp',
    'shortness_breath': 'shortness_of_breath',
    'high_bp': 'trestbps',
    'irregular_heartbeat': 'arrhythmia',
    'fatigue': 'fatigue',
    'swelling': 'edema',
    'dizziness': 'dizziness',
    'palpitations': 'palpitations'
}