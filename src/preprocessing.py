"""
Data Preprocessing Module
==========================
Handles feature scaling, train-test splitting, and class balancing.

This module implements the complete preprocessing pipeline required before
model training, including:
- Feature scaling using StandardScaler
- Train-test split with stratification
- SMOTE oversampling for handling class imbalance

Author: Your Name
Date: 2024
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess(df, test_size=0.2, random_state=42, apply_smote=True):
    """
    Complete preprocessing pipeline for credit card fraud detection.
    
    This function performs the following steps:
    1. Separates features (X) and target (y)
    2. Applies StandardScaler to normalize features
    3. Splits data into training and testing sets (stratified)
    4. Applies SMOTE to balance the training set
    
    Parameters
    ----------
    df : pandas.DataFrame
        Credit card transaction DataFrame with 'Class' column
    test_size : float, optional
        Proportion of dataset to include in test split (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    apply_smote : bool, optional
        Whether to apply SMOTE oversampling (default: True)
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) where:
        - X_train: Training features (possibly resampled with SMOTE)
        - X_test: Testing features
        - y_train: Training labels (possibly resampled with SMOTE)
        - y_test: Testing labels
    
    Notes
    -----
    **Why StandardScaler?**
    - Features have different scales (PCA features ~[-5,5], Amount ~[0, 25000])
    - ML algorithms perform better with normalized features
    - Prevents features with larger scales from dominating
    
    **Why Stratified Split?**
    - Maintains fraud/non-fraud ratio in both train and test sets
    - Critical for imbalanced datasets
    - Ensures representative evaluation
    
    **Why SMOTE?**
    - Original dataset is highly imbalanced (~0.17% fraud)
    - SMOTE creates synthetic minority class samples
    - Improves model's ability to learn fraud patterns
    - Better than simple oversampling (avoids exact duplicates)
    
    Examples
    --------
    >>> df = load_data()
    >>> X_train, X_test, y_train, y_test = preprocess(df)
    >>> print(f"Training set size: {X_train.shape}")
    >>> print(f"Fraud ratio in training: {y_train.sum()/len(y_train):.2%}")
    
    >>> # Without SMOTE
    >>> X_train, X_test, y_train, y_test = preprocess(df, apply_smote=False)
    """
    try:
        logger.info("="*60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Separate features and target
        logger.info("Step 1/4: Separating features and target...")
        if 'Class' not in df.columns:
            raise ValueError("DataFrame must contain 'Class' column as target variable")
        
        X = df.drop("Class", axis=1)
        y = df["Class"]
        
        logger.info(f"  Features shape: {X.shape}")
        logger.info(f"  Target shape: {y.shape}")
        logger.info(f"  Original fraud ratio: {y.sum()/len(y)*100:.2f}%")
        
        # Step 2: Feature Scaling
        logger.info("\nStep 2/4: Applying StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"  Scaled features shape: {X_scaled.shape}")
        logger.info(f"  Feature mean after scaling: {X_scaled.mean():.6f} (should be ~0)")
        logger.info(f"  Feature std after scaling: {X_scaled.std():.6f} (should be ~1)")
        
        # Step 3: Train-Test Split (Stratified)
        logger.info(f"\nStep 3/4: Splitting data (test_size={test_size}, stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y  # Maintains class distribution
        )
        
        logger.info(f"  Training set: {X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%)")
        logger.info(f"  Test set: {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
        logger.info(f"  Training fraud ratio: {y_train.sum()/len(y_train)*100:.4f}%")
        logger.info(f"  Test fraud ratio: {y_test.sum()/len(y_test)*100:.4f}%")
        
        # Step 4: SMOTE Oversampling (Optional)
        if apply_smote:
            logger.info("\nStep 4/4: Applying SMOTE oversampling...")
            logger.info(f"  Before SMOTE - Fraud cases: {y_train.sum():,}")
            logger.info(f"  Before SMOTE - Non-fraud cases: {(y_train==0).sum():,}")
            
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            
            logger.info(f"  After SMOTE - Fraud cases: {y_train.sum():,}")
            logger.info(f"  After SMOTE - Non-fraud cases: {(y_train==0).sum():,}")
            logger.info(f"  After SMOTE - Fraud ratio: {y_train.sum()/len(y_train)*100:.2f}%")
            logger.info(f"  ✓ SMOTE created {y_train.sum() - y.sum():,} synthetic fraud samples")
        else:
            logger.info("\nStep 4/4: Skipping SMOTE (apply_smote=False)")
        
        logger.info("\n" + "="*60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*60)
        logger.info(f"Final training set: {X_train.shape}")
        logger.info(f"Final test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def get_preprocessing_summary(X_train, X_test, y_train, y_test):
    """
    Generate a summary of the preprocessing results.
    
    Parameters
    ----------
    X_train, X_test : numpy.ndarray
        Training and testing features
    y_train, y_test : numpy.ndarray or pandas.Series
        Training and testing labels
    
    Returns
    -------
    dict
        Dictionary containing preprocessing statistics
    
    Examples
    --------
    >>> X_train, X_test, y_train, y_test = preprocess(df)
    >>> summary = get_preprocessing_summary(X_train, X_test, y_train, y_test)
    >>> print(summary)
    """
    try:
        summary = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'train_fraud_count': int(np.sum(y_train)),
            'test_fraud_count': int(np.sum(y_test)),
            'train_fraud_ratio': float(np.sum(y_train) / len(y_train)),
            'test_fraud_ratio': float(np.sum(y_test) / len(y_test)),
            'train_imbalance_ratio': float((y_train == 0).sum() / (y_train == 1).sum()),
        }
        
        logger.info("\nPreprocessing Summary:")
        logger.info(f"  Training samples: {summary['train_samples']:,}")
        logger.info(f"  Test samples: {summary['test_samples']:,}")
        logger.info(f"  Features: {summary['n_features']}")
        logger.info(f"  Train fraud ratio: {summary['train_fraud_ratio']:.2%}")
        
        return summary
    
    except Exception as e:
        logger.error(f"Error generating preprocessing summary: {e}")
        return {}


def save_scaler(scaler, filepath="models/scaler.pkl"):
    """
    Save the fitted scaler for later use in production.
    
    Parameters
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object
    filepath : str, optional
        Path to save the scaler (default: 'models/scaler.pkl')
    
    Examples
    --------
    >>> scaler = StandardScaler()
    >>> scaler.fit(X_train)
    >>> save_scaler(scaler, "models/my_scaler.pkl")
    """
    try:
        import joblib
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(scaler, filepath)
        logger.info(f"Scaler saved to: {filepath}")
    
    except Exception as e:
        logger.error(f"Error saving scaler: {e}")
        raise


def load_scaler(filepath="models/scaler.pkl"):
    """
    Load a saved scaler for preprocessing new data.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the saved scaler (default: 'models/scaler.pkl')
    
    Returns
    -------
    sklearn.preprocessing.StandardScaler
        Loaded scaler object
    
    Examples
    --------
    >>> scaler = load_scaler("models/my_scaler.pkl")
    >>> X_new_scaled = scaler.transform(X_new)
    """
    try:
        import joblib
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        
        scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded from: {filepath}")
        return scaler
    
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        raise
