import pickle
import os
import numpy as np
from typing import Dict, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# RandomForest
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# LinearRegression
def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# XGBoost
def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train)
    return model


# Оценка модели
def evaluate_model(model, X, y) -> float:
    preds = model.predict(X)
    return mean_absolute_error(y, preds)


# Кросс-валидация
def cv_score(model, X, y) -> float:
    scores = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=5)
    return -scores.mean()


# Обучение всех моделей
def train_all_models(X_train, y_train, X_test, y_test) -> Dict[str, Tuple[float, object]]:
    models = {
        "RandomForest": train_random_forest(X_train, y_train),
        "LinearRegression": train_linear(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train)
    }

    results = {}
    for name, model in models.items():
        mae = evaluate_model(model, X_test, y_test)
        results[name] = (mae, model)

    return results


# Выбор лучшей модели
def select_best_model(results: Dict[str, Tuple[float, object]]):
    best = min(results.items(), key=lambda x: x[1][0])
    return best[0], best[1][1]


# Обучение
def train_model(X_train, y_train, X_test=None, y_test=None):
    if X_test is not None and y_test is not None:
        results = train_all_models(X_train, y_train, X_test, y_test)
        _, model = select_best_model(results)
        return model

    model = train_random_forest(X_train, y_train)
    return model


# Сохранение модели
def save_model(model, path="models/model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


# Загрузка модели
def load_model(path="models/model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
