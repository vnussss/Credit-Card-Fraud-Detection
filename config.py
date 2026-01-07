"""
Configuration Module
====================
Centralized configuration for the fraud detection system.

This module contains all configurable parameters including:
- File paths
- Model hyperparameters
- Preprocessing settings
- Visualization settings

Author: Your Name
Date: 2024
"""

import os

# ============================================================
# FILE PATHS
# ============================================================

# Data paths
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "creditcard.csv")

# Output paths
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
LOG_DIR = "."

# Model file paths
RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Log file
LOG_FILE = "fraud_detection.log"

# ============================================================
# DATA PREPROCESSING SETTINGS
# ============================================================

# Train-test split
TEST_SIZE = 0.2  # 20% of data for testing
RANDOM_STATE = 42  # For reproducibility

# SMOTE settings
APPLY_SMOTE = True  # Whether to apply SMOTE oversampling
SMOTE_RANDOM_STATE = 42

# Feature scaling
SCALING_METHOD = "standard"  # Options: 'standard', 'minmax', 'robust'

# ============================================================
# RANDOM FOREST HYPERPARAMETERS
# ============================================================

RF_CONFIG = {
    'n_estimators': 300,        # Number of trees
    'max_depth': 12,            # Maximum depth of trees
    'min_samples_split': 2,     # Minimum samples to split
    'min_samples_leaf': 1,      # Minimum samples in leaf
    'max_features': 'sqrt',     # Number of features per split
    'class_weight': 'balanced', # Handle class imbalance
    'random_state': RANDOM_STATE,
    'n_jobs': -1,               # Use all CPU cores
    'verbose': 0
}

# ============================================================
# XGBOOST HYPERPARAMETERS
# ============================================================

XGB_CONFIG = {
    'n_estimators': 200,         # Number of boosting rounds
    'max_depth': 8,              # Maximum tree depth
    'learning_rate': 0.05,       # Step size shrinkage
    'subsample': 0.8,            # Fraction of samples per tree
    'colsample_bytree': 0.8,     # Fraction of features per tree
    'colsample_bylevel': 0.8,    # Fraction of features per level
    'min_child_weight': 1,       # Minimum sum of instance weight
    'gamma': 0,                  # Minimum loss reduction
    'reg_alpha': 0,              # L1 regularization
    'reg_lambda': 1,             # L2 regularization
    'scale_pos_weight': 10,      # Balance positive/negative weights
    'random_state': RANDOM_STATE,
    'n_jobs': -1,                # Use all CPU cores
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

# Plot settings
PLOT_DPI = 300                  # High quality plots
PLOT_STYLE = 'whitegrid'        # Seaborn style
PLOT_PALETTE = 'husl'           # Color palette
FIGURE_SIZE_LARGE = (12, 8)     # For detailed plots
FIGURE_SIZE_MEDIUM = (10, 6)    # For standard plots
FIGURE_SIZE_SMALL = (8, 5)      # For compact plots

# Visualization file names
VIZ_FILES = {
    'class_balance': 'class_balance.png',
    'transaction_dist': 'transaction_amount_distribution.png',
    'fraud_density': 'fraud_density.png',
    'correlation_heatmap': 'correlation_heatmap.png',
    'boxplots': 'boxplots_key_features.png',
    'rf_confusion_matrix': 'confusion_matrix_random_forest.png',
    'rf_roc_curve': 'roc_curve_random_forest.png',
    'rf_pr_curve': 'precision_recall_curve_random_forest.png',
    'xgb_confusion_matrix': 'confusion_matrix_xgboost.png',
    'xgb_roc_curve': 'roc_curve_xgboost.png',
    'xgb_pr_curve': 'precision_recall_curve_xgboost.png',
    'model_comparison': 'model_comparison.png'
}

# ============================================================
# STATISTICAL ANALYSIS SETTINGS
# ============================================================

# Correlation analysis
TOP_CORRELATIONS = 15  # Number of top correlated features to display

# Key features for detailed analysis
KEY_FEATURES = ['V14', 'V12', 'V10', 'V4', 'V11', 'V17', 'Amount']

# ============================================================
# EVALUATION SETTINGS
# ============================================================

# Metrics to compute
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc',
    'pr_auc'
]

# Classification threshold
CLASSIFICATION_THRESHOLD = 0.5  # Can be adjusted based on business needs

# ============================================================
# LOGGING SETTINGS
# ============================================================

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_directories():
    """
    Create necessary directories if they don't exist.
    
    Creates:
    - data/
    - outputs/
    - models/
    - logs/
    """
    directories = [DATA_DIR, OUTPUT_DIR, MODEL_DIR]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"✓ Directories created: {', '.join(directories)}")


def get_config_summary():
    """
    Get a summary of the current configuration.
    
    Returns
    -------
    dict
        Dictionary containing configuration summary
    """
    summary = {
        'data_file': DATA_FILE,
        'test_size': TEST_SIZE,
        'apply_smote': APPLY_SMOTE,
        'random_forest': {
            'n_estimators': RF_CONFIG['n_estimators'],
            'max_depth': RF_CONFIG['max_depth']
        },
        'xgboost': {
            'n_estimators': XGB_CONFIG['n_estimators'],
            'max_depth': XGB_CONFIG['max_depth'],
            'learning_rate': XGB_CONFIG['learning_rate']
        }
    }
    
    return summary


def print_config():
    """Print the current configuration to console."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    summary = get_config_summary()
    
    print(f"\nData:")
    print(f"  File: {summary['data_file']}")
    print(f"  Test Size: {summary['test_size']}")
    print(f"  Apply SMOTE: {summary['apply_smote']}")
    
    print(f"\nRandom Forest:")
    print(f"  Trees: {summary['random_forest']['n_estimators']}")
    print(f"  Max Depth: {summary['random_forest']['max_depth']}")
    
    print(f"\nXGBoost:")
    print(f"  Estimators: {summary['xgboost']['n_estimators']}")
    print(f"  Max Depth: {summary['xgboost']['max_depth']}")
    print(f"  Learning Rate: {summary['xgboost']['learning_rate']}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test configuration
    print_config()
    create_directories()
