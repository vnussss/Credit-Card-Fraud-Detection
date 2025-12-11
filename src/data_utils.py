import pandas as pd

def load_data(path="data/creditcard.csv"):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.drop_duplicates()
    return df
