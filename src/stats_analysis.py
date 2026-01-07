"""
Statistical Analysis Module
============================
Performs comprehensive statistical analysis on credit card transaction data.

This module provides functions for:
- Descriptive statistics
- Fraud distribution analysis
- Comparative analysis between fraud and non-fraud transactions
- Feature correlation analysis

Author: Your Name
Date: 2024
"""

import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def descriptive_stats(df):
    """
    Display comprehensive descriptive statistics for the dataset.
    
    Provides statistical summary including:
    - Count, mean, std, min, max, quartiles for all numerical features
    - Helps identify data distribution and potential anomalies
    
    Parameters
    ----------
    df : pandas.DataFrame
        Credit card transaction DataFrame
    
    Returns
    -------
    pandas.DataFrame
        Descriptive statistics DataFrame
    
    Examples
    --------
    >>> df = load_data()
    >>> stats = descriptive_stats(df)
    """
    try:
        logger.info("Generating descriptive statistics...")
        print("\n" + "="*60)
        print("BASIC DESCRIPTIVE STATISTICS")
        print("="*60)
        
        stats = df.describe()
        print(stats)
        
        # Additional statistics
        print("\n" + "-"*60)
        print("ADDITIONAL STATISTICS")
        print("-"*60)
        print(f"Total Transactions: {len(df):,}")
        print(f"Total Features: {df.shape[1]}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Data Types:\n{df.dtypes.value_counts()}")
        
        return stats
    
    except Exception as e:
        logger.error(f"Error generating descriptive statistics: {e}")
        raise


def fraud_distribution(df):
    """
    Analyze and display the distribution of fraud vs non-fraud transactions.
    
    This is critical for understanding class imbalance in the dataset.
    Fraud detection datasets are typically highly imbalanced (< 1% fraud).
    
    Parameters
    ----------
    df : pandas.DataFrame
        Credit card transaction DataFrame with 'Class' column
    
    Returns
    -------
    pandas.Series
        Value counts for fraud (1) and non-fraud (0) classes
    
    Notes
    -----
    Class imbalance requires special handling:
    - SMOTE for oversampling minority class
    - Class weights in model training
    - Appropriate evaluation metrics (precision, recall, F1)
    
    Examples
    --------
    >>> df = load_data()
    >>> distribution = fraud_distribution(df)
    """
    try:
        logger.info("Analyzing fraud distribution...")
        print("\n" + "="*60)
        print("FRAUD DISTRIBUTION ANALYSIS")
        print("="*60)
        
        fraud_counts = df['Class'].value_counts()
        total = len(df)
        
        print(f"\nNon-Fraud (Class 0): {fraud_counts[0]:,} ({fraud_counts[0]/total*100:.2f}%)")
        print(f"Fraud (Class 1):     {fraud_counts[1]:,} ({fraud_counts[1]/total*100:.2f}%)")
        
        # Calculate imbalance ratio
        imbalance_ratio = fraud_counts[0] / fraud_counts[1]
        print(f"\nClass Imbalance Ratio: 1:{imbalance_ratio:.0f}")
        print(f"(For every 1 fraud transaction, there are {imbalance_ratio:.0f} legitimate ones)")
        
        # Alert if severe imbalance
        if imbalance_ratio > 100:
            print("\n⚠️  WARNING: Severe class imbalance detected!")
            print("   Recommendation: Use SMOTE, class weights, or other balancing techniques")
        
        return fraud_counts
    
    except Exception as e:
        logger.error(f"Error analyzing fraud distribution: {e}")
        raise


def compare_groups(df):
    """
    Compare statistical measures between fraud and non-fraud transactions.
    
    This analysis helps identify which features differ significantly between
    fraudulent and legitimate transactions, providing insights for feature selection.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Credit card transaction DataFrame with 'Class' column
    
    Returns
    -------
    pandas.DataFrame
        Mean values for each feature grouped by Class (fraud vs non-fraud)
    
    Notes
    -----
    Features with large differences between groups are typically more predictive.
    Look for features where fraud transactions show distinct patterns.
    
    Examples
    --------
    >>> df = load_data()
    >>> comparison = compare_groups(df)
    >>> # Features like V14, V12, V10 often show significant differences
    """
    try:
        logger.info("Comparing fraud vs non-fraud groups...")
        print("\n" + "="*60)
        print("FRAUD vs NON-FRAUD COMPARISON (Mean Values)")
        print("="*60)
        
        grouped_means = df.groupby('Class').mean()
        print(grouped_means)
        
        # Calculate difference magnitude for key features
        print("\n" + "-"*60)
        print("FEATURES WITH LARGEST DIFFERENCES")
        print("-"*60)
        
        # Calculate absolute difference between fraud and non-fraud means
        differences = (grouped_means.loc[1] - grouped_means.loc[0]).abs()
        top_differences = differences.nlargest(10)
        
        print("\nTop 10 Features by Mean Difference:")
        for feature, diff in top_differences.items():
            fraud_mean = grouped_means.loc[1, feature]
            nonfraud_mean = grouped_means.loc[0, feature]
            print(f"{feature:10s}: Fraud={fraud_mean:8.3f}, Non-Fraud={nonfraud_mean:8.3f}, Diff={diff:8.3f}")
        
        return grouped_means
    
    except Exception as e:
        logger.error(f"Error comparing groups: {e}")
        raise


def correlation_table(df, top_n=15):
    """
    Analyze and display feature correlations with the fraud target variable.
    
    Correlation analysis helps identify which features have the strongest
    relationship with fraudulent transactions, guiding feature selection
    and model interpretation.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Credit card transaction DataFrame
    top_n : int, optional
        Number of top correlated features to display (default: 15)
    
    Returns
    -------
    pandas.Series
        Correlation coefficients sorted by absolute value
    
    Notes
    -----
    - Positive correlation: Feature increases with fraud
    - Negative correlation: Feature decreases with fraud
    - Values closer to ±1 indicate stronger relationships
    - V14, V12, V10 typically show strong correlations
    
    Examples
    --------
    >>> df = load_data()
    >>> correlations = correlation_table(df, top_n=20)
    """
    try:
        logger.info("Calculating feature correlations with fraud...")
        print("\n" + "="*60)
        print(f"CORRELATION WITH FRAUD (Top {top_n} Features)")
        print("="*60)
        
        # Calculate correlation matrix
        corr = df.corr()['Class'].sort_values(ascending=False)
        
        # Display top positive correlations
        print("\n📈 POSITIVE CORRELATIONS (Features that increase with fraud):")
        print("-"*60)
        top_positive = corr.head(top_n)
        for feature, corr_value in top_positive.items():
            if feature != 'Class':  # Skip the target variable itself
                bar = '█' * int(abs(corr_value) * 50)
                print(f"{feature:10s}: {corr_value:7.4f} {bar}")
        
        # Display top negative correlations
        print("\n📉 NEGATIVE CORRELATIONS (Features that decrease with fraud):")
        print("-"*60)
        top_negative = corr.tail(top_n)
        for feature, corr_value in top_negative.items():
            if feature != 'Class':
                bar = '█' * int(abs(corr_value) * 50)
                print(f"{feature:10s}: {corr_value:7.4f} {bar}")
        
        # Interpretation guide
        print("\n" + "-"*60)
        print("INTERPRETATION GUIDE:")
        print("  Strong:   |correlation| > 0.5")
        print("  Moderate: 0.3 < |correlation| < 0.5")
        print("  Weak:     |correlation| < 0.3")
        print("-"*60)
        
        return corr
    
    except Exception as e:
        logger.error(f"Error calculating correlations: {e}")
        raise


def generate_statistical_report(df, output_path="outputs/statistical_report.txt"):
    """
    Generate a comprehensive statistical report and save to file.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Credit card transaction DataFrame
    output_path : str, optional
        Path to save the report file
    
    Returns
    -------
    str
        Path to the generated report file
    
    Examples
    --------
    >>> df = load_data()
    >>> report_path = generate_statistical_report(df)
    >>> print(f"Report saved to: {report_path}")
    """
    try:
        import os
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"Generating statistical report: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CREDIT CARD FRAUD DETECTION - STATISTICAL ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Transactions: {len(df):,}\n")
            f.write(f"Features: {df.shape[1]}\n")
            f.write(f"Fraud Cases: {df['Class'].sum():,} ({df['Class'].sum()/len(df)*100:.2f}%)\n\n")
            
            # Descriptive statistics
            f.write("\nDESCRIPTIVE STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(df.describe().to_string())
            f.write("\n\n")
            
            # Fraud distribution
            f.write("\nFRAUD DISTRIBUTION\n")
            f.write("-"*80 + "\n")
            f.write(df['Class'].value_counts().to_string())
            f.write("\n")
        
        logger.info(f"Statistical report saved successfully: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating statistical report: {e}")
        raise
