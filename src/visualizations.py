# visualizations.py
# High-quality visuals saved as PNGs (no GUI popups)
# ---------------------------------------------------

import os
import matplotlib
matplotlib.use("Agg")  # Disable GUI backend (prevents freezing)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Create "outputs" folder if not exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")


# ---------------------------------------------------
# 1. Transaction Amount Distribution
# ---------------------------------------------------
def plot_transaction_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Amount"], kde=True, bins=50)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Count")

    plt.savefig("outputs/transaction_amount_distribution.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------
# 2. Fraud vs Non-Fraud Density Plot
# ---------------------------------------------------
def fraud_density_plot(df):
    fraud = df[df["Class"] == 1]["Amount"]
    non_fraud = df[df["Class"] == 0]["Amount"]

    plt.figure(figsize=(10, 5))
    sns.kdeplot(non_fraud, fill=True, alpha=0.45, label="Non-Fraud")
    sns.kdeplot(fraud, fill=True, alpha=0.45, label="Fraud", color="red")
    plt.title("Fraud vs Non-Fraud Transaction Amount Density")
    plt.xlabel("Amount")
    plt.ylabel("Density")
    plt.legend()

    plt.savefig("outputs/fraud_density.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------
# 3. Correlation Heatmap (all features)
# ---------------------------------------------------
def correlation_heatmap(df):
    corr = df.corr()

    plt.figure(figsize=(18, 18))
    sns.heatmap(corr, cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")

    plt.savefig("outputs/correlation_heatmap.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------
# 4. Class Balance Bar Plot
# ---------------------------------------------------
def plot_class_balance(df):
    plt.figure(figsize=(6, 5))
    sns.countplot(x=df["Class"], palette="coolwarm")
    plt.title("Fraud Class Distribution")
    plt.xticks([0, 1], ["Non-Fraud", "Fraud"])

    plt.savefig("outputs/class_balance.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------
# 5. Boxplots for Key Features
# ---------------------------------------------------
def boxplot_key_features(df):
    important_features = ["V14", "V12", "V10", "Amount"]
    df_sample = df.sample(5000)  # reduces load for plotting

    plt.figure(figsize=(12, 8))
    for i, col in enumerate(important_features, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=df_sample["Class"], y=df_sample[col], palette="coolwarm")
        plt.title(f"{col} by Class")
        plt.xlabel("Class")

    plt.tight_layout()
    plt.savefig("outputs/boxplots_key_features.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------
# 6. Pairplot (optional – heavy but saved only, not shown)
# ---------------------------------------------------
def pairplot_features(df):
    df_sample = df.sample(3000)  # pairplot would explode otherwise
    sns.pairplot(df_sample[["V14", "V12", "V10", "Amount", "Class"]],
                 hue="Class", diag_kind="kde")

    plt.savefig("outputs/pairplot.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------
# Run all visuals (you call this from main.py)
# ---------------------------------------------------
def generate_all_visuals(df):
    print("Generating visuals…")
    plot_transaction_distribution(df)
    fraud_density_plot(df)
    correlation_heatmap(df)
    plot_class_balance(df)
    boxplot_key_features(df)
    print("All visuals saved in /outputs folder ✓")
