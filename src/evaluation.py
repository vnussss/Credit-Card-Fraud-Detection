"""
Model Evaluation Module
========================
Comprehensive evaluation metrics and visualizations for fraud detection models.

This module provides functions for:
- Classification reports (Precision, Recall, F1-Score)
- Confusion matrices with visualizations
- ROC-AUC scores and ROC curves
- Model comparison utilities
- Detailed performance analysis

Author: Your Name
Date: 2024
"""

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate(model, X_test, y_test, model_name="Model", save_plots=True, output_dir="outputs"):
    """
    Comprehensive evaluation of a fraud detection model.
    
    This function performs complete model evaluation including:
    - Classification report with precision, recall, F1-score
    - Confusion matrix visualization
    - ROC-AUC score calculation
    - ROC curve plotting (optional)
    - Precision-Recall curve (optional)
    
    Parameters
    ----------
    model : sklearn or xgboost model
        Trained model to evaluate
    X_test : numpy.ndarray or pandas.DataFrame
        Test features
    y_test : numpy.ndarray or pandas.Series
        True labels for test set
    model_name : str, optional
        Name of the model for logging and file naming (default: "Model")
    save_plots : bool, optional
        Whether to save visualization plots (default: True)
    output_dir : str, optional
        Directory to save plots (default: "outputs")
    
    Returns
    -------
    dict
        Dictionary containing all evaluation metrics:
        - classification_report: Detailed metrics by class
        - confusion_matrix: 2x2 matrix
        - roc_auc: ROC-AUC score
        - accuracy: Overall accuracy
        - precision: Precision for fraud class
        - recall: Recall for fraud class
        - f1_score: F1-score for fraud class
    
    Notes
    -----
    **Understanding the Metrics:**
    
    1. **Precision (Positive Predictive Value)**:
       - Of all transactions flagged as fraud, what % were actually fraud?
       - High precision = Few false alarms
       - Formula: TP / (TP + FP)
       - Important when false alarms are costly
    
    2. **Recall (Sensitivity, True Positive Rate)**:
       - Of all actual fraud cases, what % did we catch?
       - High recall = Catching most fraud
       - Formula: TP / (TP + FN)
       - Critical in fraud detection (missing fraud is very costly)
    
    3. **F1-Score**:
       - Harmonic mean of precision and recall
       - Balances both metrics
       - Formula: 2 * (Precision * Recall) / (Precision + Recall)
       - Best single metric for imbalanced classes
    
    4. **ROC-AUC (Area Under ROC Curve)**:
       - Measures model's ability to distinguish classes
       - Range: 0.5 (random) to 1.0 (perfect)
       - >0.9 = Excellent, >0.8 = Good, >0.7 = Fair
       - Threshold-independent metric
    
    **Confusion Matrix Interpretation:**
    ```
                    Predicted
                    0       1
    Actual  0     TN      FP    (False Positive = False Alarm)
            1     FN      TP    (False Negative = Missed Fraud)
    ```
    
    - **True Negative (TN)**: Correctly identified non-fraud
    - **False Positive (FP)**: Legitimate transaction flagged as fraud
    - **False Negative (FN)**: Fraud transaction missed (WORST CASE)
    - **True Positive (TP)**: Correctly identified fraud
    
    Examples
    --------
    >>> rf_model = train_rf(X_train, y_train)
    >>> metrics = evaluate(rf_model, X_test, y_test, model_name="Random Forest")
    >>> print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    >>> # Evaluate without saving plots
    >>> metrics = evaluate(model, X_test, y_test, save_plots=False)
    """
    try:
        logger.info("="*60)
        logger.info(f"EVALUATING {model_name.upper()}")
        logger.info("="*60)
        
        # Make predictions
        logger.info("Generating predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of fraud
        
        # Calculate metrics
        metrics = {}
        
        # 1. Classification Report
        logger.info("\n" + "="*60)
        logger.info("CLASSIFICATION REPORT")
        logger.info("="*60)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))
        
        metrics['classification_report'] = report
        metrics['precision'] = report['1']['precision']
        metrics['recall'] = report['1']['recall']
        metrics['f1_score'] = report['1']['f1-score']
        metrics['accuracy'] = report['accuracy']
        
        # 2. Confusion Matrix
        logger.info("\n" + "="*60)
        logger.info("CONFUSION MATRIX")
        logger.info("="*60)
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Extract values
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n{'':12s} {'Predicted Non-Fraud':>20s} {'Predicted Fraud':>20s}")
        print(f"{'Actual Non-Fraud':12s} {tn:>20,} (TN) {fp:>20,} (FP)")
        print(f"{'Actual Fraud':12s} {fn:>20,} (FN) {tp:>20,} (TP)")
        
        print(f"\nDetailed Breakdown:")
        print(f"  True Negatives (TN):  {tn:,} - Correctly identified non-fraud")
        print(f"  False Positives (FP): {fp:,} - False alarms (legitimate flagged as fraud)")
        print(f"  False Negatives (FN): {fn:,} - Missed fraud (CRITICAL ERRORS)")
        print(f"  True Positives (TP):  {tp:,} - Correctly caught fraud")
        
        # Calculate additional metrics
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        print(f"\nError Rates:")
        print(f"  False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
        print(f"  False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%) ⚠️")
        
        # 3. ROC-AUC Score
        logger.info("\n" + "="*60)
        logger.info("ROC-AUC SCORE")
        logger.info("="*60)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        metrics['roc_auc'] = roc_auc
        
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        # Interpret ROC-AUC
        if roc_auc >= 0.9:
            interpretation = "Excellent 🌟"
        elif roc_auc >= 0.8:
            interpretation = "Good ✓"
        elif roc_auc >= 0.7:
            interpretation = "Fair"
        else:
            interpretation = "Poor ⚠️"
        print(f"Model Performance: {interpretation}")
        
        # 4. Precision-Recall AUC
        pr_auc = average_precision_score(y_test, y_pred_proba)
        metrics['pr_auc'] = pr_auc
        print(f"Precision-Recall AUC: {pr_auc:.4f}")
        
        # Save visualizations
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            
            # Confusion Matrix Heatmap
            plot_confusion_matrix(cm, model_name, output_dir)
            
            # ROC Curve
            plot_roc_curve(y_test, y_pred_proba, roc_auc, model_name, output_dir)
            
            # Precision-Recall Curve
            plot_precision_recall_curve(y_test, y_pred_proba, pr_auc, model_name, output_dir)
        
        logger.info("\n" + "="*60)
        logger.info(f"EVALUATION COMPLETE - {model_name}")
        logger.info("="*60)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def plot_confusion_matrix(cm, model_name, output_dir):
    """
    Create and save confusion matrix heatmap.
    
    Parameters
    ----------
    cm : numpy.ndarray
        Confusion matrix
    model_name : str
        Name of the model
    output_dir : str
        Directory to save the plot
    """
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Fraud', 'Fraud'],
                   yticklabels=['Non-Fraud', 'Fraud'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        
        filepath = os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Confusion matrix saved: {filepath}")
    
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")


def plot_roc_curve(y_test, y_pred_proba, roc_auc, model_name, output_dir):
    """
    Create and save ROC curve.
    
    Parameters
    ----------
    y_test : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    roc_auc : float
        ROC-AUC score
    model_name : str
        Name of the model
    output_dir : str
        Directory to save the plot
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        filepath = os.path.join(output_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ ROC curve saved: {filepath}")
    
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}")


def plot_precision_recall_curve(y_test, y_pred_proba, pr_auc, model_name, output_dir):
    """
    Create and save Precision-Recall curve.
    
    Parameters
    ----------
    y_test : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    pr_auc : float
        Precision-Recall AUC score
    model_name : str
        Name of the model
    output_dir : str
        Directory to save the plot
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        filepath = os.path.join(output_dir, f'precision_recall_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Precision-Recall curve saved: {filepath}")
    
    except Exception as e:
        logger.error(f"Error plotting Precision-Recall curve: {e}")


def compare_models(models_metrics, output_dir="outputs"):
    """
    Compare multiple models and generate comparison visualizations.
    
    Parameters
    ----------
    models_metrics : dict
        Dictionary with model names as keys and their metrics as values
    output_dir : str, optional
        Directory to save comparison plots (default: "outputs")
    
    Examples
    --------
    >>> rf_metrics = evaluate(rf_model, X_test, y_test, "Random Forest")
    >>> xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost")
    >>> compare_models({
    ...     "Random Forest": rf_metrics,
    ...     "XGBoost": xgb_metrics
    ... })
    """
    try:
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        # Extract metrics for comparison
        comparison_data = {
            'Model': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'ROC-AUC': []
        }
        
        for model_name, metrics in models_metrics.items():
            comparison_data['Model'].append(model_name)
            comparison_data['Precision'].append(metrics['precision'])
            comparison_data['Recall'].append(metrics['recall'])
            comparison_data['F1-Score'].append(metrics['f1_score'])
            comparison_data['ROC-AUC'].append(metrics['roc_auc'])
        
        # Print comparison table
        print(f"\n{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-" * 68)
        for i in range(len(comparison_data['Model'])):
            print(f"{comparison_data['Model'][i]:<20} "
                  f"{comparison_data['Precision'][i]:<12.4f} "
                  f"{comparison_data['Recall'][i]:<12.4f} "
                  f"{comparison_data['F1-Score'][i]:<12.4f} "
                  f"{comparison_data['ROC-AUC'][i]:<12.4f}")
        
        # Create comparison bar plot
        plot_model_comparison(comparison_data, output_dir)
        
        logger.info("\n✓ Model comparison complete")
    
    except Exception as e:
        logger.error(f"Error comparing models: {e}")


def plot_model_comparison(comparison_data, output_dir):
    """
    Create bar plot comparing model metrics.
    
    Parameters
    ----------
    comparison_data : dict
        Dictionary containing model comparison data
    output_dir : str
        Directory to save the plot
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            ax.bar(comparison_data['Model'], comparison_data[metric], color='skyblue', edgecolor='navy')
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'Model Comparison - {metric}', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(comparison_data[metric]):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Model comparison plot saved: {filepath}")
    
    except Exception as e:
        logger.error(f"Error plotting model comparison: {e}")
