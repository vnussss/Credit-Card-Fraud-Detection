"""
Machine Learning Models Module
===============================
Implements Random Forest and XGBoost classifiers for fraud detection.

This module provides functions for:
- Training Random Forest classifier with optimized hyperparameters
- Training XGBoost classifier with custom configuration
- Model persistence (save/load functionality)
- Hyperparameter explanations and best practices

Author: Your Name
Date: 2024
"""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_rf(X, y, n_estimators=300, max_depth=12, class_weight='balanced', 
             random_state=42, save_model=True, model_path="models/random_forest_model.pkl"):
    """
    Train a Random Forest Classifier for fraud detection.
    
    Random Forest is an ensemble learning method that builds multiple decision
    trees and merges them to get a more accurate and stable prediction. It's
    particularly effective for fraud detection due to its ability to:
    - Handle non-linear relationships
    - Provide feature importance rankings
    - Resist overfitting through ensemble averaging
    - Handle imbalanced data with class weights
    
    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Training features (already scaled and resampled)
    y : numpy.ndarray or pandas.Series
        Training labels (0=non-fraud, 1=fraud)
    n_estimators : int, optional
        Number of trees in the forest (default: 300)
        More trees = better performance but slower training
    max_depth : int, optional
        Maximum depth of each tree (default: 12)
        Deeper trees can capture more patterns but may overfit
    class_weight : str or dict, optional
        Weights for classes (default: 'balanced')
        'balanced' automatically adjusts weights inversely proportional to class frequencies
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    save_model : bool, optional
        Whether to save the trained model (default: True)
    model_path : str, optional
        Path to save the model (default: 'models/random_forest_model.pkl')
    
    Returns
    -------
    RandomForestClassifier
        Trained Random Forest model
    
    Notes
    -----
    **Hyperparameter Choices Explained:**
    
    1. **n_estimators=300**: 
       - More trees generally improve performance
       - Diminishing returns after ~300-500 trees
       - 300 provides good balance of accuracy vs training time
    
    2. **max_depth=12**: 
       - Controls tree depth to prevent overfitting
       - 12 is deep enough for complex patterns but not too deep
       - Prevents memorizing training data
    
    3. **class_weight='balanced'**: 
       - Automatically adjusts for class imbalance
       - Gives higher weight to minority class (fraud)
       - Prevents model from predicting only majority class
    
    4. **random_state=42**: 
       - Ensures reproducible results
       - Same results across different runs
       - Critical for comparing model versions
    
    Examples
    --------
    >>> X_train, X_test, y_train, y_test = preprocess(df)
    >>> rf_model = train_rf(X_train, y_train)
    >>> predictions = rf_model.predict(X_test)
    
    >>> # Custom hyperparameters
    >>> rf_model = train_rf(X_train, y_train, n_estimators=500, max_depth=15)
    """
    try:
        logger.info("="*60)
        logger.info("TRAINING RANDOM FOREST CLASSIFIER")
        logger.info("="*60)
        
        # Log training configuration
        logger.info(f"Training set size: {X.shape[0]:,} samples, {X.shape[1]} features")
        logger.info(f"Fraud ratio: {y.sum()/len(y)*100:.2f}%")
        logger.info(f"\nHyperparameters:")
        logger.info(f"  n_estimators: {n_estimators}")
        logger.info(f"  max_depth: {max_depth}")
        logger.info(f"  class_weight: {class_weight}")
        logger.info(f"  random_state: {random_state}")
        
        # Initialize model
        logger.info("\nInitializing Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores for faster training
            verbose=0
        )
        
        # Train model
        logger.info("Training in progress...")
        start_time = datetime.now()
        model.fit(X, y)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✓ Training completed in {training_time:.2f} seconds")
        logger.info(f"✓ Model trained with {len(model.estimators_)} trees")
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            top_features = sorted(enumerate(model.feature_importances_), 
                                key=lambda x: x[1], reverse=True)[:5]
            logger.info("\nTop 5 Most Important Features:")
            for idx, importance in top_features:
                logger.info(f"  Feature {idx}: {importance:.4f}")
        
        # Save model
        if save_model:
            save_trained_model(model, model_path, model_type='Random Forest')
        
        logger.info("="*60)
        return model
    
    except Exception as e:
        logger.error(f"Error training Random Forest: {e}")
        raise


def train_xgb(X, y, n_estimators=200, max_depth=8, learning_rate=0.05, 
              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
              random_state=42, save_model=True, model_path="models/xgboost_model.pkl"):
    """
    Train an XGBoost Classifier for fraud detection.
    
    XGBoost (Extreme Gradient Boosting) is a powerful gradient boosting algorithm
    that builds trees sequentially, with each tree correcting errors from previous ones.
    It's highly effective for fraud detection because it:
    - Handles imbalanced data with scale_pos_weight
    - Provides excellent predictive performance
    - Includes built-in regularization to prevent overfitting
    - Offers fast training with GPU support (optional)
    
    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Training features (already scaled and resampled)
    y : numpy.ndarray or pandas.Series
        Training labels (0=non-fraud, 1=fraud)
    n_estimators : int, optional
        Number of boosting rounds (default: 200)
    max_depth : int, optional
        Maximum depth of trees (default: 8)
    learning_rate : float, optional
        Step size shrinkage (default: 0.05)
        Lower values require more trees but often improve generalization
    subsample : float, optional
        Fraction of samples used for each tree (default: 0.8)
    colsample_bytree : float, optional
        Fraction of features used for each tree (default: 0.8)
    scale_pos_weight : float, optional
        Balancing of positive and negative weights (default: 10)
        Set to: (# non-fraud) / (# fraud) for balanced importance
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    save_model : bool, optional
        Whether to save the trained model (default: True)
    model_path : str, optional
        Path to save the model (default: 'models/xgboost_model.pkl')
    
    Returns
    -------
    XGBClassifier
        Trained XGBoost model
    
    Notes
    -----
    **Hyperparameter Choices Explained:**
    
    1. **n_estimators=200**: 
       - Number of boosting rounds (trees built sequentially)
       - More rounds improve performance but risk overfitting
       - 200 provides good balance with low learning rate
    
    2. **max_depth=8**: 
       - Controls tree complexity
       - Shallower than Random Forest (boosting builds sequentially)
       - Prevents overfitting in gradient boosting
    
    3. **learning_rate=0.05**: 
       - Controls how much each tree contributes
       - Lower = more conservative learning = better generalization
       - Requires more n_estimators to achieve good performance
    
    4. **subsample=0.8**: 
       - Random sampling of training data for each tree
       - Introduces randomness to prevent overfitting
       - 0.8 = use 80% of data for each tree
    
    5. **colsample_bytree=0.8**: 
       - Random sampling of features for each tree
       - Similar to Random Forest's feature randomness
       - Improves model robustness
    
    6. **scale_pos_weight=10**: 
       - Adjusts weight of positive class (fraud)
       - Helps model focus on minority class
       - Typically set to (# non-fraud) / (# fraud)
    
    Examples
    --------
    >>> X_train, X_test, y_train, y_test = preprocess(df)
    >>> xgb_model = train_xgb(X_train, y_train)
    >>> predictions = xgb_model.predict(X_test)
    
    >>> # With custom learning rate and more trees
    >>> xgb_model = train_xgb(X_train, y_train, 
    ...                       n_estimators=500, 
    ...                       learning_rate=0.01)
    """
    try:
        logger.info("="*60)
        logger.info("TRAINING XGBOOST CLASSIFIER")
        logger.info("="*60)
        
        # Log training configuration
        logger.info(f"Training set size: {X.shape[0]:,} samples, {X.shape[1]} features")
        logger.info(f"Fraud ratio: {y.sum()/len(y)*100:.2f}%")
        logger.info(f"\nHyperparameters:")
        logger.info(f"  n_estimators: {n_estimators}")
        logger.info(f"  max_depth: {max_depth}")
        logger.info(f"  learning_rate: {learning_rate}")
        logger.info(f"  subsample: {subsample}")
        logger.info(f"  colsample_bytree: {colsample_bytree}")
        logger.info(f"  scale_pos_weight: {scale_pos_weight}")
        logger.info(f"  random_state: {random_state}")
        
        # Initialize model
        logger.info("\nInitializing XGBoost model...")
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1  # Use all CPU cores
        )
        
        # Train model
        logger.info("Training in progress...")
        start_time = datetime.now()
        model.fit(X, y)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✓ Training completed in {training_time:.2f} seconds")
        logger.info(f"✓ Model trained with {model.n_estimators} boosting rounds")
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            top_features = sorted(enumerate(model.feature_importances_), 
                                key=lambda x: x[1], reverse=True)[:5]
            logger.info("\nTop 5 Most Important Features:")
            for idx, importance in top_features:
                logger.info(f"  Feature {idx}: {importance:.4f}")
        
        # Save model
        if save_model:
            save_trained_model(model, model_path, model_type='XGBoost')
        
        logger.info("="*60)
        return model
    
    except Exception as e:
        logger.error(f"Error training XGBoost: {e}")
        raise


def save_trained_model(model, filepath, model_type='Model'):
    """
    Save a trained model to disk for later use.
    
    Parameters
    ----------
    model : sklearn or xgboost model
        Trained model to save
    filepath : str
        Path where model will be saved
    model_type : str, optional
        Type of model for logging (default: 'Model')
    
    Examples
    --------
    >>> save_trained_model(rf_model, "models/my_model.pkl", "Random Forest")
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(model, filepath)
        
        # Get file size
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        
        logger.info(f"\n✓ {model_type} model saved successfully")
        logger.info(f"  Path: {filepath}")
        logger.info(f"  Size: {file_size:.2f} MB")
    
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def load_trained_model(filepath):
    """
    Load a previously trained model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to the saved model file
    
    Returns
    -------
    model
        Loaded model object
    
    Examples
    --------
    >>> model = load_trained_model("models/random_forest_model.pkl")
    >>> predictions = model.predict(X_test)
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        logger.info(f"Loading model from: {filepath}")
        model = joblib.load(filepath)
        logger.info("✓ Model loaded successfully")
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
