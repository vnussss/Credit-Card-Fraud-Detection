"""
Credit Card Fraud Detection System
===================================
Main execution script for the fraud detection pipeline.

This script orchestrates the complete machine learning pipeline:
1. Data loading and cleaning
2. Exploratory data analysis (EDA)
3. Statistical analysis
4. Visualization generation
5. Data preprocessing (scaling, SMOTE)
6. Model training (Random Forest & XGBoost)
7. Model evaluation and comparison

Usage:
    python main.py

Author: Your Name
Project: Credit Card Fraud Detection
Date: 2024
Internship: [Company Name]
"""

import sys
import logging
from datetime import datetime

# Import custom modules
from src.data_utils import load_data, clean_data, get_data_summary
from src.stats_analysis import (
    descriptive_stats,
    fraud_distribution,
    compare_groups,
    correlation_table
)
from src.visualizations import generate_all_visuals
from src.preprocessing import preprocess
from src.model import train_rf, train_xgb
from src.evaluation import evaluate, compare_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def print_header():
    """Print a professional header for the program."""
    print("\n" + "="*80)
    print(" "*20 + "CREDIT CARD FRAUD DETECTION SYSTEM")
    print(" "*25 + "Machine Learning Pipeline")
    print("="*80)
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def main():
    """
    Main execution function for the fraud detection pipeline.
    
    Executes the complete ML workflow from data loading to model evaluation.
    """
    try:
        # Print header
        print_header()
        
        # ============================================================
        # SECTION 1: DATA LOADING & CLEANING
        # ============================================================
        print_section("SECTION 1: DATA LOADING & CLEANING")
        logger.info("Starting data loading...")
        
        df = load_data()
        df = clean_data(df)
        
        # Display data summary
        summary = get_data_summary(df)
        logger.info(f"Dataset loaded: {summary['total_transactions']:,} transactions")
        logger.info(f"Fraud cases: {summary['fraud_count']:,} ({summary['fraud_percentage']:.2f}%)")
        
        # ============================================================
        # SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
        # ============================================================
        print_section("SECTION 2: EXPLORATORY DATA ANALYSIS")
        logger.info("Performing statistical analysis...")
        
        # Generate comprehensive statistics
        descriptive_stats(df)
        fraud_distribution(df)
        compare_groups(df)
        correlation_table(df)
        
        # ============================================================
        # SECTION 3: VISUALIZATION GENERATION
        # ============================================================
        print_section("SECTION 3: VISUALIZATION GENERATION")
        logger.info("Creating visualizations...")
        
        generate_all_visuals(df)
        
        # ============================================================
        # SECTION 4: DATA PREPROCESSING
        # ============================================================
        print_section("SECTION 4: DATA PREPROCESSING")
        logger.info("Preprocessing data for model training...")
        
        X_train, X_test, y_train, y_test = preprocess(df)
        
        logger.info(f"Training set: {X_train.shape[0]:,} samples")
        logger.info(f"Test set: {X_test.shape[0]:,} samples")
        
        # ============================================================
        # SECTION 5: MODEL TRAINING - RANDOM FOREST
        # ============================================================
        print_section("SECTION 5: MODEL TRAINING - RANDOM FOREST")
        logger.info("Training Random Forest classifier...")
        
        rf_model = train_rf(X_train, y_train)
        
        # ============================================================
        # SECTION 6: MODEL EVALUATION - RANDOM FOREST
        # ============================================================
        print_section("SECTION 6: MODEL EVALUATION - RANDOM FOREST")
        logger.info("Evaluating Random Forest model...")
        
        rf_metrics = evaluate(rf_model, X_test, y_test, model_name="Random Forest")
        
        # ============================================================
        # SECTION 7: MODEL TRAINING - XGBOOST
        # ============================================================
        print_section("SECTION 7: MODEL TRAINING - XGBOOST")
        logger.info("Training XGBoost classifier...")
        
        xgb_model = train_xgb(X_train, y_train)
        
        # ============================================================
        # SECTION 8: MODEL EVALUATION - XGBOOST
        # ============================================================
        print_section("SECTION 8: MODEL EVALUATION - XGBOOST")
        logger.info("Evaluating XGBoost model...")
        
        xgb_metrics = evaluate(xgb_model, X_test, y_test, model_name="XGBoost")
        
        # ============================================================
        # SECTION 9: MODEL COMPARISON
        # ============================================================
        print_section("SECTION 9: MODEL COMPARISON")
        logger.info("Comparing model performance...")
        
        compare_models({
            "Random Forest": rf_metrics,
            "XGBoost": xgb_metrics
        })
        
        # ============================================================
        # COMPLETION
        # ============================================================
        print("\n" + "="*80)
        print(" "*25 + "PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\n📊 Generated Outputs:")
        print("  ├── Visualizations: outputs/")
        print("  │   ├── class_balance.png")
        print("  │   ├── transaction_amount_distribution.png")
        print("  │   ├── fraud_density.png")
        print("  │   ├── correlation_heatmap.png")
        print("  │   └── boxplots_key_features.png")
        print("  ├── Model Evaluations: outputs/")
        print("  │   ├── confusion_matrix_random_forest.png")
        print("  │   ├── roc_curve_random_forest.png")
        print("  │   ├── precision_recall_curve_random_forest.png")
        print("  │   ├── confusion_matrix_xgboost.png")
        print("  │   ├── roc_curve_xgboost.png")
        print("  │   ├── precision_recall_curve_xgboost.png")
        print("  │   └── model_comparison.png")
        print("  ├── Trained Models: models/")
        print("  │   ├── random_forest_model.pkl")
        print("  │   └── xgboost_model.pkl")
        print("  └── Logs: fraud_detection.log")
        print("\n" + "="*80)
        print(f"Execution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        logger.info("✓ All tasks completed successfully")
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Please ensure 'data/creditcard.csv' exists")
        logger.error("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return 1
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print("\n" + "="*80)
        print(" "*30 + "ERROR OCCURRED")
        print("="*80)
        print(f"\nError: {e}")
        print("\nCheck fraud_detection.log for details")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    """
    Entry point of the program.
    Runs the main pipeline and exits with appropriate status code.
    """
    exit_code = main()
    sys.exit(exit_code)
