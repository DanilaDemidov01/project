from __future__ import annotations

from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
)
from sklearn.metrics import make_scorer, mean_absolute_error
from xgboost import XGBRegressor


def mae_scorer() -> Any:
    return make_scorer(mean_absolute_error, greater_is_better=False)


def get_cv(n_splits: int = 5, random_state: int = 42) -> KFold:
    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def get_param_grid_rf() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }


def get_param_grid_xgb() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [200, 400, 600, 800],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [1.0, 1.5, 2.0],
    }


def get_param_grid_lr() -> Dict[str, List[Any]]:
    return {
        "fit_intercept": [True, False],
        "positive": [False, True],
    }


def grid_search(
    model_name: str,
    estimator,
    param_grid: Dict[str, List[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    n_jobs: int = -1,
) -> Tuple[Any, float, pd.DataFrame]:
    scorer = mae_scorer()
    cv = get_cv(n_splits=cv_splits)
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    gs.fit(X, y)
    best_estimator = gs.best_estimator_
    best_score = -float(gs.best_score_)
    results = pd.DataFrame(gs.cv_results_)
    results["model"] = model_name
    return best_estimator, best_score, results


def random_search(
    model_name: str,
    estimator,
    param_distributions: Dict[str, List[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 30,
    cv_splits: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Tuple[Any, float, pd.DataFrame]:
    scorer = mae_scorer()
    cv = get_cv(n_splits=cv_splits, random_state=random_state)
    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        verbose=0,
    )
    rs.fit(X, y)
    best_estimator = rs.best_estimator_
    best_score = -float(rs.best_score_)
    results = pd.DataFrame(rs.cv_results_)
    results["model"] = model_name
    return best_estimator, best_score, results


def prepare_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )


def prepare_xgb() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=400,
        n_jobs=-1,
    )


def prepare_lr() -> LinearRegression:
    return LinearRegression()


def run_rf_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    mode: str = "random",
    n_iter: int = 30,
    cv_splits: int = 5,
) -> Tuple[Any, float, pd.DataFrame]:
    base_model = prepare_rf()
    param_grid = get_param_grid_rf()
    if mode == "grid":
        best_model, best_score, results = grid_search(
            "RandomForest",
            base_model,
            param_grid,
            X,
            y,
            cv_splits=cv_splits,
        )
    else:
        best_model, best_score, results = random_search(
            "RandomForest",
            base_model,
            param_grid,
            X,
            y,
            n_iter=n_iter,
            cv_splits=cv_splits,
        )
    return best_model, best_score, results


def run_xgb_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    mode: str = "random",
    n_iter: int = 30,
    cv_splits: int = 5,
) -> Tuple[Any, float, pd.DataFrame]:
    base_model = prepare_xgb()
    param_grid = get_param_grid_xgb()
    if mode == "grid":
        best_model, best_score, results = grid_search(
            "XGBoost",
            base_model,
            param_grid,
            X,
            y,
            cv_splits=cv_splits,
        )
    else:
        best_model, best_score, results = random_search(
            "XGBoost",
            base_model,
            param_grid,
            X,
            y,
            n_iter=n_iter,
            cv_splits=cv_splits,
        )
    return best_model, best_score, results


def run_lr_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> Tuple[Any, float, pd.DataFrame]:
    base_model = prepare_lr()
    param_grid = get_param_grid_lr()
    best_model, best_score, results = grid_search(
        "LinearRegression",
        base_model,
        param_grid,
        X,
        y,
        cv_splits=cv_splits,
    )
    return best_model, best_score, results


def collect_best_result(
    model_name: str,
    best_score: float,
    best_estimator,
) -> Dict[str, Any]:
    params = getattr(best_estimator, "get_params", lambda: {})()
    return {
        "model": model_name,
        "best_mae": best_score,
        "params": params,
    }


def run_full_hyperparameter_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    mode_rf: str = "random",
    mode_xgb: str = "random",
    n_iter_rf: int = 30,
    n_iter_xgb: int = 30,
    cv_splits: int = 5,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    best_models: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    all_results: List[pd.DataFrame] = []

    rf_best, rf_score, rf_res = run_rf_tuning(
        X,
        y,
        mode=mode_rf,
        n_iter=n_iter_rf,
        cv_splits=cv_splits,
    )
    best_models["RandomForest"] = rf_best
    summary_rows.append(collect_best_result("RandomForest", rf_score, rf_best))
    all_results.append(rf_res)

    xgb_best, xgb_score, xgb_res = run_xgb_tuning(
        X,
        y,
        mode=mode_xgb,
        n_iter=n_iter_xgb,
        cv_splits=cv_splits,
    )
    best_models["XGBoost"] = xgb_best
    summary_rows.append(collect_best_result("XGBoost", xgb_score, xgb_best))
    all_results.append(xgb_res)

    lr_best, lr_score, lr_res = run_lr_tuning(
        X,
        y,
        cv_splits=cv_splits,
    )
    best_models["LinearRegression"] = lr_best
    summary_rows.append(collect_best_result("LinearRegression", lr_score, lr_best))
    all_results.append(lr_res)

    summary_df = pd.DataFrame(summary_rows)
    results_df = pd.concat(all_results, ignore_index=True)

    results_df_sorted = results_df.sort_values(
        by=["model", "mean_test_score"],
        ascending=[True, False],
    )

    return {
        "best_models": best_models,
        "summary": summary_df,
        "all_results": results_df_sorted,
    }, results_df_sorted
