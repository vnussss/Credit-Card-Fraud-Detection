from src.data_utils import load_data, clean_data
from src.stats_analysis import (
    descriptive_stats,
    fraud_distribution,
    compare_groups,
    correlation_table
)
from src.visualizations import (
    plot_class_balance,
    plot_transaction_distribution,
    correlation_heatmap,
    boxplot_key_features,
    fraud_density_plot,
    generate_all_visuals
)
from src.preprocessing import preprocess
from src.model import train_rf, train_xgb
from src.evaluation import evaluate


# ------------------ Load & Clean ------------------
df = load_data()
df = clean_data(df)

# ------------------ Stats Analysis ------------------
descriptive_stats(df)
fraud_distribution(df)
compare_groups(df)
correlation_table(df)

generate_all_visuals(df)

X_train, X_test, y_train, y_test = preprocess(df)

rf_model = train_rf(X_train, y_train)
evaluate(rf_model, X_test, y_test)

xgb_model = train_xgb(X_train, y_train)
evaluate(xgb_model, X_test, y_test)
