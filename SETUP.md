# 🚀 Quick Setup Guide

Follow these steps to get the project running on your machine.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning)

## Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd fraud-detection-system
```

## Step 2: Create Virtual Environment (Recommended)

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy (data processing)
- scikit-learn (ML algorithms)
- xgboost (gradient boosting)
- imbalanced-learn (SMOTE)
- matplotlib, seaborn (visualizations)

## Step 4: Download the Dataset

1. Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the `data/` directory:

```bash
mkdir -p data
# Move your downloaded file to data/creditcard.csv
```

## Step 5: Run the Pipeline

```bash
python main.py
```

This will:
1. Load and clean the data
2. Perform statistical analysis
3. Generate visualizations (saved to `outputs/`)
4. Preprocess data with SMOTE
5. Train Random Forest and XGBoost models
6. Evaluate and compare models
7. Save trained models to `models/`

## Step 6: View Results

Check the generated outputs:

```bash
# View visualizations
ls outputs/

# View trained models
ls models/

# View execution log
cat fraud_detection.log
```

## Expected Output Structure

```
fraud-detection-system/
├── data/
│   └── creditcard.csv          # Your dataset (not tracked in git)
├── outputs/                     # Generated visualizations
│   ├── class_balance.png
│   ├── transaction_amount_distribution.png
│   ├── fraud_density.png
│   ├── correlation_heatmap.png
│   ├── boxplots_key_features.png
│   ├── confusion_matrix_random_forest.png
│   ├── roc_curve_random_forest.png
│   ├── precision_recall_curve_random_forest.png
│   ├── confusion_matrix_xgboost.png
│   ├── roc_curve_xgboost.png
│   ├── precision_recall_curve_xgboost.png
│   └── model_comparison.png
├── models/                      # Trained models
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
└── fraud_detection.log          # Execution log
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Make sure you've installed all dependencies
```bash
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: data/creditcard.csv"
**Solution**: Download and place the dataset in the `data/` directory

### Issue: "MemoryError" during SMOTE
**Solution**: Your system might have insufficient RAM. You can:
1. Disable SMOTE in `config.py`: Set `APPLY_SMOTE = False`
2. Use a subset of data for testing

### Issue: Visualizations not displaying
**Solution**: The project saves plots as PNG files. Check the `outputs/` directory

## Running Individual Components

You can also run individual modules:

```python
# Test data loading
from src.data_utils import load_data, clean_data
df = load_data()
df = clean_data(df)
print(df.shape)

# Test preprocessing
from src.preprocessing import preprocess
X_train, X_test, y_train, y_test = preprocess(df)

# Load a saved model
from src.model import load_trained_model
model = load_trained_model("models/random_forest_model.pkl")
predictions = model.predict(X_test)
```

## Configuration

Customize the project by editing `config.py`:
- Model hyperparameters
- File paths
- Preprocessing settings
- Visualization settings

## Next Steps

1. ✅ Review the generated visualizations
2. ✅ Check model performance in console output
3. ✅ Read the INTERVIEW_GUIDE.md for project explanation
4. ✅ Explore the code in `src/` directory
5. ✅ Experiment with different hyperparameters
6. ✅ Try adding new models or features

## Getting Help

- Check the main README.md for detailed documentation
- Review INTERVIEW_GUIDE.md for technical explanations
- Check fraud_detection.log for execution details
- All code has comprehensive docstrings

## Tips for Development

1. Always activate virtual environment before running
2. Use `python -m pip list` to see installed packages
3. Check logs if something fails
4. Start with a small data sample for testing
5. Keep the virtual environment separate per project

---

**Happy Fraud Detection! 🎯**
