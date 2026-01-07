"""
Data Utilities Module
=====================
Handles data loading, cleaning, and basic preprocessing operations.

This module provides functions for:
- Loading credit card transaction data from CSV
- Data cleaning (duplicate removal, null handling)
- Basic data validation

Author: Your Name
Date: 2024
"""

import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(path="data/creditcard.csv"):
    """
    Load credit card transaction data from CSV file.
    
    This function reads the credit card fraud dataset and performs basic validation.
    The dataset should contain PCA-transformed features (V1-V28), Time, Amount, and Class columns.
    
    Parameters
    ----------
    path : str, optional
        Path to the CSV file containing credit card transaction data.
        Default is 'data/creditcard.csv'
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the credit card transaction data with columns:
        - V1 to V28: PCA-transformed features
        - Time: Seconds elapsed since first transaction
        - Amount: Transaction amount
        - Class: Target variable (0=legitimate, 1=fraud)
    
    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist
    ValueError
        If the loaded data is empty or missing required columns
    
    Examples
    --------
    >>> df = load_data()
    >>> print(df.shape)
    (284807, 31)
    
    >>> df = load_data("custom_data/transactions.csv")
    """
    try:
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at path: {path}")
        
        logger.info(f"Loading data from: {path}")
        df = pd.read_csv(path)
        
        # Validate data
        if df.empty:
            raise ValueError("Loaded dataset is empty!")
        
        # Check for required columns
        required_columns = ['Class', 'Amount', 'Time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Fraud cases: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.2f}%)")
        
        return df
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
        raise ValueError("The CSV file is empty or corrupted")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def clean_data(df):
    """
    Clean the credit card transaction dataset.
    
    Performs the following cleaning operations:
    1. Removes duplicate transactions
    2. Handles missing values (if any)
    3. Validates data types
    4. Removes invalid entries
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw credit card transaction DataFrame
    
    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame ready for analysis and modeling
    
    Raises
    ------
    ValueError
        If DataFrame is empty or invalid
    
    Examples
    --------
    >>> df_raw = load_data()
    >>> df_clean = clean_data(df_raw)
    >>> print(f"Removed {len(df_raw) - len(df_clean)} duplicate rows")
    """
    try:
        if df is None or df.empty:
            raise ValueError("Cannot clean empty DataFrame")
        
        logger.info("Starting data cleaning process...")
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values")
            # For this dataset, we'll drop rows with missing values
            # In production, you might want to impute instead
            df = df.dropna()
            logger.info(f"Removed rows with missing values")
        
        # Remove negative amounts (invalid transactions)
        if 'Amount' in df.columns:
            invalid_amounts = (df['Amount'] < 0).sum()
            if invalid_amounts > 0:
                logger.warning(f"Found {invalid_amounts} transactions with negative amounts")
                df = df[df['Amount'] >= 0]
        
        # Validate Class column contains only 0 and 1
        if 'Class' in df.columns:
            valid_classes = df['Class'].isin([0, 1]).all()
            if not valid_classes:
                logger.warning("Found invalid values in Class column, filtering...")
                df = df[df['Class'].isin([0, 1])]
        
        final_rows = len(df)
        logger.info(f"Data cleaning complete: {final_rows} rows remaining ({initial_rows - final_rows} rows removed)")
        
        return df
    
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise


def get_data_summary(df):
    """
    Generate a summary of the dataset for quick analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Credit card transaction DataFrame
    
    Returns
    -------
    dict
        Dictionary containing summary statistics including:
        - total_transactions: Total number of transactions
        - fraud_count: Number of fraudulent transactions
        - legitimate_count: Number of legitimate transactions
        - fraud_percentage: Percentage of fraudulent transactions
        - avg_amount: Average transaction amount
        - total_amount: Total transaction amount
    
    Examples
    --------
    >>> df = load_data()
    >>> summary = get_data_summary(df)
    >>> print(f"Fraud rate: {summary['fraud_percentage']:.2f}%")
    """
    try:
        summary = {
            'total_transactions': len(df),
            'fraud_count': int(df['Class'].sum()),
            'legitimate_count': int((df['Class'] == 0).sum()),
            'fraud_percentage': float(df['Class'].sum() / len(df) * 100),
            'avg_amount': float(df['Amount'].mean()),
            'total_amount': float(df['Amount'].sum()),
            'date_range': f"{df['Time'].min():.0f}s to {df['Time'].max():.0f}s"
        }
        
        return summary
    
    except Exception as e:
        logger.error(f"Error generating data summary: {e}")
        return {}
