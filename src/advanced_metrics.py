from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# SMAPE
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not mask.any():
        return 0.0
    diff = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(np.mean(diff) * 100.0)


# NRMSE
def nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    norm: str = "mean",
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if norm == "mean":
        denom = np.mean(np.abs(y_true))
    elif norm == "range":
        denom = np.max(y_true) - np.min(y_true)
    else:
        denom = 1.0
    if denom == 0:
        return 0.0
    return float(rmse / denom)


# Метрики по группам
def grouped_rank_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    bins: List[int] | None = None,
) -> pd.DataFrame:
    if bins is None:
        bins = [0, 10, 25, 50, 100, 200]
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["rank_group"] = pd.cut(
        df["y_true"],
        bins=bins,
        labels=[f"{bins[i] + 1}-{bins[i + 1]}" for i in range(len(bins) - 1)],
        include_lowest=True,
    )
    rows: List[Dict[str, Any]] = []
    for group, gdf in df.groupby("rank_group"):
        if gdf.empty:
            continue
        yt = gdf["y_true"].to_numpy()
        yp = gdf["y_pred"].to_numpy()
        mae = mean_absolute_error(yt, yp)
        rmse_val = np.sqrt(mean_squared_error(yt, yp))
        r2_val = r2_score(yt, yp) if len(gdf) > 1 else np.nan
        mape_val = float(np.mean(np.abs((yt - yp) / yt)) * 100.0)
        smape_val = smape(yt, yp)
        rows.append(
            {
                "group": str(group),
                "count": len(gdf),
                "mae": mae,
                "rmse": rmse_val,
                "r2": r2_val,
                "mape": mape_val,
                "smape": smape_val,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# Распределение ошибок
def error_distribution(
    y_true: pd.Series,
    y_pred: np.ndarray,
    quantiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> pd.DataFrame:
    y_true_arr = y_true.to_numpy()
    y_pred_arr = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred_arr - y_true_arr)
    rel_err = np.where(y_true_arr != 0, abs_err / np.abs(y_true_arr), 0.0)
    data = {
        "metric": [],
        "value": [],
    }
    data["metric"].append("mean_abs_error")
    data["value"].append(float(abs_err.mean()))
    data["metric"].append("max_abs_error")
    data["value"].append(float(abs_err.max()))
    data["metric"].append("mean_rel_error")
    data["value"].append(float(rel_err.mean()))
    for q in quantiles:
        data["metric"].append(f"q{int(q * 100)}_abs_error")
        data["value"].append(float(np.quantile(abs_err, q)))
    return pd.DataFrame(data)


# Диагностика регрессии
def regression_diagnostics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    y_true_arr = y_true.to_numpy()
    y_pred_arr = np.asarray(y_pred, dtype=float)
    diff = y_pred_arr - y_true_arr
    bias = float(np.mean(diff))
    over = float(np.mean(diff > 0))
    under = float(np.mean(diff < 0))
    corr = float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1]) if len(y_true_arr) > 1 else 0.0
    return {
        "bias": bias,
        "over_ratio": over,
        "under_ratio": under,
        "corr_true_pred": corr,
    }


# Сводный отчёт по метрикам
def full_advanced_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    y_true_arr = y_true.to_numpy()
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse_val = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2_val = r2_score(y_true_arr, y_pred_arr) if len(y_true_arr) > 1 else np.nan
    mape_val = float(np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100.0)
    smape_val = smape(y_true_arr, y_pred_arr)
    nrmse_mean = nrmse(y_true_arr, y_pred_arr, norm="mean")
    nrmse_range = nrmse(y_true_arr, y_pred_arr, norm="range")
    diag = regression_diagnostics(y_true, y_pred_arr)
    result: Dict[str, Any] = {
        "mae": mae,
        "rmse": rmse_val,
        "r2": r2_val,
        "mape": mape_val,
        "smape": smape_val,
        "nrmse_mean": nrmse_mean,
        "nrmse_range": nrmse_range,
    }
    result.update(diag)
    return result
