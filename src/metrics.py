from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)
import numpy as np
import pandas as pd


# Основные метрики
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    medae = median_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("MAE:", mae)
    print("Median AE:", medae)
    print("RMSE:", rmse)
    print("R2:", r2)

    return {
        "mae": mae,
        "medae": medae,
        "rmse": rmse,
        "r2": r2
    }


# Ошибки по объектам
def prediction_errors(model, X_test, y_test):
    preds = model.predict(X_test)
    return preds - y_test


# Таблица метрик в DataFrame
def metrics_table(model, X_test, y_test):
    preds = model.predict(X_test)

    data = {
        "y_true": y_test,
        "y_pred": preds,
        "abs_error": np.abs(preds - y_test),
        "squared_error": (preds - y_test) ** 2
    }

    df = pd.DataFrame(data)
    return df


# Сводная метрика MAPE
def mape(model, X_test, y_test):
    """
    MAPE с защитой от деления на ноль:
    строки, где y_test == 0, в расчёт не попадают.
    """
    preds = model.predict(X_test)
    y_true = np.array(y_test, dtype=float)

    # маска только там, где y != 0
    mask = y_true != 0
    if not np.any(mask):
        return np.nan

    mape_vals = np.abs((y_true[mask] - preds[mask]) / y_true[mask]) * 100
    return float(np.mean(mape_vals))

