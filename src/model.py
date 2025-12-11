from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_rf(X, y):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)
    return model

def train_xgb(X, y):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,
        scale_pos_weight=10
    )
    model.fit(X, y)
    return model
