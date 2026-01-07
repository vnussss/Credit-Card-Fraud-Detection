# 💳 Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive machine learning system for detecting fraudulent credit card transactions using ensemble methods (Random Forest & XGBoost) with advanced data preprocessing techniques including SMOTE for handling class imbalance.

![Project Banner](https://img.shields.io/badge/ML-Fraud%20Detection-red?style=for-the-badge&logo=security&logoColor=white)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technologies Used](#-technologies-used)
- [Project Architecture](#-project-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Results & Visualizations](#-results--visualizations)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **binary classification system** to identify fraudulent credit card transactions from a highly imbalanced dataset. The system uses advanced machine learning techniques to achieve high accuracy while maintaining excellent fraud detection rates.

### Problem Statement
Credit card fraud is a critical issue in the financial industry. With millions of transactions occurring daily, manual detection is impractical. This system automates fraud detection using machine learning, helping financial institutions prevent fraudulent activities in real-time.

### Solution Approach
- **Data Preprocessing**: Feature scaling and handling class imbalance with SMOTE
- **Exploratory Data Analysis**: Comprehensive statistical analysis and visualizations
- **Model Training**: Implementation of Random Forest and XGBoost classifiers
- **Model Evaluation**: Performance assessment using multiple metrics (Precision, Recall, F1-Score, ROC-AUC)

---

## ✨ Key Features

### 🔍 Data Analysis
- **Automated EDA Pipeline**: Descriptive statistics, correlation analysis, and distribution analysis
- **Visual Analytics**: 5+ high-quality visualizations including heatmaps, density plots, and boxplots
- **Class Imbalance Analysis**: Detailed fraud vs non-fraud transaction comparison

### 🤖 Machine Learning Models
- **Random Forest Classifier**: 300 estimators with balanced class weights
- **XGBoost Classifier**: Gradient boosting with custom hyperparameters
- **SMOTE Oversampling**: Synthetic Minority Over-sampling Technique for handling imbalanced data
- **Feature Scaling**: StandardScaler normalization for optimal model performance

### 📊 Evaluation Metrics
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix with visualizations
- ROC-AUC Score for model comparison
- Detailed performance analysis for both models

---

## 🛠️ Technologies Used

### Core Technologies
| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Programming Language | 3.8+ |
| **pandas** | Data Manipulation | 2.0.3 |
| **NumPy** | Numerical Computing | 1.24.3 |
| **scikit-learn** | Machine Learning Framework | 1.3.0 |
| **XGBoost** | Gradient Boosting | 2.0.3 |
| **imbalanced-learn** | SMOTE Implementation | 0.11.0 |
| **Matplotlib** | Data Visualization | 3.7.2 |
| **Seaborn** | Statistical Visualization | 0.12.2 |

### Why These Technologies?
- **scikit-learn**: Industry-standard ML library with excellent documentation
- **XGBoost**: State-of-the-art gradient boosting for superior performance
- **SMOTE**: Proven technique for handling imbalanced datasets
- **Seaborn/Matplotlib**: Professional-quality visualizations for analysis

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION SYSTEM                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. DATA INGESTION                                           │
│     ├── Load CSV Data (creditcard.csv)                       │
│     └── Remove Duplicates                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. EXPLORATORY DATA ANALYSIS (EDA)                          │
│     ├── Descriptive Statistics                               │
│     ├── Fraud Distribution Analysis                          │
│     ├── Feature Correlation Analysis                         │
│     └── Visualization Generation (5 plots)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. DATA PREPROCESSING                                       │
│     ├── Feature Scaling (StandardScaler)                     │
│     ├── Train-Test Split (80/20)                             │
│     └── SMOTE Oversampling (Handle Imbalance)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. MODEL TRAINING                                           │
│     ├── Random Forest (300 estimators)                       │
│     └── XGBoost (200 estimators)                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. MODEL EVALUATION                                         │
│     ├── Classification Report                                │
│     ├── Confusion Matrix                                     │
│     └── ROC-AUC Score                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd webapp
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Dataset**
   - Download the Credit Card Fraud Detection dataset
   - Place `creditcard.csv` in the `data/` directory
   - Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🚀 Usage

### Quick Start

```bash
# Run the complete pipeline
python main.py
```

### What Happens When You Run?

1. **Data Loading**: Loads the credit card transaction dataset
2. **Statistical Analysis**: Prints comprehensive statistics to console
3. **Visualization Generation**: Creates 5 high-quality PNG files in `outputs/`
4. **Model Training**: Trains Random Forest and XGBoost models
5. **Evaluation**: Displays performance metrics for both models

### Output Locations

```
outputs/
├── class_balance.png                    # Fraud vs Non-Fraud distribution
├── transaction_amount_distribution.png  # Transaction amounts histogram
├── fraud_density.png                    # Density plot comparison
├── correlation_heatmap.png              # Feature correlation matrix
└── boxplots_key_features.png            # Key features by class
```

### Understanding the Outputs

#### Console Output
- **Descriptive Stats**: Mean, std, min, max for all features
- **Fraud Count**: Number of fraudulent vs legitimate transactions
- **Feature Correlations**: Top features correlated with fraud
- **Model Metrics**: Precision, Recall, F1-Score, ROC-AUC

#### Visual Outputs
All visualizations are saved as high-resolution (300 DPI) PNG files for presentation and analysis.

---

## 📈 Model Performance

### Dataset Characteristics
- **Total Transactions**: ~284,807
- **Fraudulent Transactions**: ~492 (0.17%)
- **Legitimate Transactions**: ~284,315 (99.83%)
- **Class Imbalance Ratio**: 1:577

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | ~99.9% | High | High | High | ~0.98+ |
| **XGBoost** | ~99.9% | High | High | High | ~0.98+ |

> **Note**: Exact metrics depend on the random state and SMOTE sampling. Run the pipeline to see specific results.

### Why These Metrics Matter?

- **Precision**: Of all transactions flagged as fraud, how many are actually fraudulent?
- **Recall**: Of all actual fraudulent transactions, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall (balanced metric)
- **ROC-AUC**: Model's ability to distinguish between classes (closer to 1 is better)

### Handling Class Imbalance

The dataset is highly imbalanced (0.17% fraud). We address this using:

1. **SMOTE (Synthetic Minority Over-sampling Technique)**
   - Generates synthetic fraud samples
   - Balances training data without duplicating existing samples
   
2. **Class Weights in Random Forest**
   - `class_weight="balanced"` parameter
   - Penalizes misclassification of minority class more heavily

3. **Scale Position Weight in XGBoost**
   - `scale_pos_weight=10` parameter
   - Adjusts algorithm to focus more on minority class

---

## 📊 Dataset

### Dataset Information

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: ~150 MB
- **Transactions**: 284,807
- **Features**: 30 (28 PCA-transformed + Time + Amount)
- **Target**: Class (0 = Legitimate, 1 = Fraud)

### Feature Description

| Feature | Description |
|---------|-------------|
| **V1-V28** | PCA-transformed features (anonymized for privacy) |
| **Time** | Seconds elapsed between this transaction and first transaction |
| **Amount** | Transaction amount |
| **Class** | Target variable (0 = Non-Fraud, 1 = Fraud) |

### Data Privacy
The dataset contains PCA-transformed features to protect sensitive customer information while maintaining analytical value.

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data_utils.py           # Data loading and cleaning
│   ├── stats_analysis.py       # Statistical analysis functions
│   ├── visualizations.py       # Visualization generation
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── model.py                # Model training functions
│   └── evaluation.py           # Model evaluation metrics
│
├── data/                        # Dataset directory (not tracked)
│   └── creditcard.csv          # Raw dataset (download separately)
│
├── outputs/                     # Generated visualizations
│   ├── class_balance.png
│   ├── transaction_amount_distribution.png
│   ├── fraud_density.png
│   ├── correlation_heatmap.png
│   └── boxplots_key_features.png
│
├── models/                      # Saved model files (to be created)
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
└── notebooks/                   # Jupyter notebooks (optional)
    └── exploratory_analysis.ipynb
```

---

## 🎨 Results & Visualizations

### 1. Class Balance Distribution
Shows the severe imbalance between fraud and non-fraud transactions.

### 2. Transaction Amount Distribution
Histogram showing the distribution of transaction amounts across all data.

### 3. Fraud Density Plot
Kernel density estimation comparing transaction amounts for fraud vs non-fraud cases.

### 4. Correlation Heatmap
Complete correlation matrix showing relationships between all features.

### 5. Key Features Boxplot
Box plots for the most important features (V14, V12, V10, Amount) separated by class.

---

## 🚀 Future Enhancements

### Planned Features
- [ ] **Model Persistence**: Save and load trained models using joblib
- [ ] **REST API**: Flask/FastAPI endpoint for real-time predictions
- [ ] **Web Dashboard**: Interactive Streamlit dashboard for model monitoring
- [ ] **Feature Importance Analysis**: Detailed analysis of which features matter most
- [ ] **Hyperparameter Tuning**: Grid search / Random search for optimal parameters
- [ ] **Cross-Validation**: K-fold cross-validation for robust performance estimates
- [ ] **Additional Models**: Neural Networks, Gradient Boosting variations
- [ ] **Model Explainability**: SHAP values for interpreting predictions
- [ ] **Automated Retraining**: Pipeline for periodic model updates
- [ ] **Docker Containerization**: Containerized deployment

### Potential Improvements
- Real-time fraud detection pipeline
- Integration with payment processing systems
- Alert system for high-risk transactions
- A/B testing framework for model comparison
- Multi-model ensemble for improved accuracy

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- GitHub: [Your GitHub](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Inspired by real-world fraud detection systems in the financial industry
- Built during internship at [Company Name] (2024)

---

## 📚 References

1. [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
2. [Random Forest Classifier - scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
3. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
4. [Handling Imbalanced Datasets in Machine Learning](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data)

---

<p align="center">
  <strong>⭐ If you found this project helpful, please consider giving it a star! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ for Financial Security
</p>
