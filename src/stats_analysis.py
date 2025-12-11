import pandas as pd

def descriptive_stats(df):
    print("\n===== BASIC DESCRIPTIVE STATS =====")
    print(df.describe())

def fraud_distribution(df):
    print("\n===== FRAUD COUNT =====")
    print(df['Class'].value_counts())

def compare_groups(df):
    print("\n===== FRAUD vs NON-FRAUD MEANS =====")
    print(df.groupby('Class').mean())

def correlation_table(df):
    print("\n===== CORRELATION WITH FRAUD (TOP FEATURES) =====")
    corr = df.corr()['Class'].sort_values(ascending=False)
    print(corr.head(15))
