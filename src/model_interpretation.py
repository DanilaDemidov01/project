from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt
import os

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 20,
    random_state: int = 42,
    scoring: str = "neg_mean_absolute_error",
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    df = df.sort_values("importance_mean", ascending=False)
    return df


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    path: str = "plots/interpretation/permutation_importance.png",
    top_n: int = 20,
) -> None:
    ensure_dir(os.path.dirname(path))
    data = importance_df.head(top_n)
    plt.figure(figsize=(10, 8))
    plt.barh(data["feature"][::-1], data["importance_mean"][::-1])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_pdp_values(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 50,
) -> pd.DataFrame:
    xs = np.linspace(X[feature].min(), X[feature].max(), grid_resolution)
    X_temp = X.copy()
    preds_mean = []
    for v in xs:
        X_temp[feature] = v
        preds = model.predict(X_temp)
        preds_mean.append(preds.mean())
    df = pd.DataFrame(
        {
            feature: xs,
            "partial_dependence": preds_mean,
        }
    )
    return df


def plot_pdp(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    path: str = "plots/interpretation/pdp.png",
    grid_resolution: int = 50,
) -> None:
    ensure_dir(os.path.dirname(path))
    df = compute_pdp_values(model, X, feature, grid_resolution)
    plt.figure(figsize=(8, 6))
    plt.plot(df[feature], df["partial_dependence"])
    plt.xlabel(feature)
    plt.ylabel("partial dependence")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_ice_values(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 30,
    max_curves: int = 50,
) -> pd.DataFrame:
    xs = np.linspace(X[feature].min(), X[feature].max(), grid_resolution)
    idx = np.arange(len(X))
    if len(idx) > max_curves:
        idx = np.random.choice(idx, size=max_curves, replace=False)
    rows: List[Dict[str, Any]] = []
    for i in idx:
        x_i = X.iloc[[i]].copy()
        for v in xs:
            x_i[feature] = v
            pred = float(model.predict(x_i)[0])
            rows.append({"sample": int(i), feature: v, "prediction": pred})
    df = pd.DataFrame(rows)
    return df


def plot_ice(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    path: str = "plots/interpretation/ice.png",
    grid_resolution: int = 30,
    max_curves: int = 50,
) -> None:
    ensure_dir(os.path.dirname(path))
    df = compute_ice_values(
        model,
        X,
        feature,
        grid_resolution=grid_resolution,
        max_curves=max_curves,
    )
    plt.figure(figsize=(8, 6))
    for _, g in df.groupby("sample"):
        plt.plot(g[feature], g["prediction"], alpha=0.3)
    plt.xlabel(feature)
    plt.ylabel("prediction")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_shap_values_tree(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    if not _HAS_SHAP:
        return np.array([]), np.array([])
    x_sample = X
    if len(x_sample) > max_samples:
        x_sample = X.sample(n=max_samples, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        base = base[0]
    return shap_values, x_sample.to_numpy()


def plot_shap_summary(
    model: Any,
    X: pd.DataFrame,
    path: str = "plots/interpretation/shap_summary.png",
    max_samples: int = 200,
) -> None:
    if not _HAS_SHAP:
        return
    ensure_dir(os.path.dirname(path))
    x_sample = X
    if len(x_sample) > max_samples:
        x_sample = X.sample(n=max_samples, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        x_sample,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def compute_shap_importance(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 300,
) -> pd.DataFrame:
    if not _HAS_SHAP:
        return pd.DataFrame()
    x_sample = X
    if len(x_sample) > max_samples:
        x_sample = X.sample(n=max_samples, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    abs_vals = np.abs(shap_values)
    mean_abs = abs_vals.mean(axis=0)
    df = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": mean_abs,
        }
    )
    df = df.sort_values("mean_abs_shap", ascending=False)
    return df


def plot_shap_importance(
    model: Any,
    X: pd.DataFrame,
    path: str = "plots/interpretation/shap_importance.png",
    max_samples: int = 300,
    top_n: int = 20,
) -> None:
    if not _HAS_SHAP:
        return
    ensure_dir(os.path.dirname(path))
    df = compute_shap_importance(model, X, max_samples)
    if df.empty:
        return
    data = df.head(top_n)
    plt.figure(figsize=(10, 8))
    plt.barh(data["feature"][::-1], data["mean_abs_shap"][::-1])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_interpretation_full(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature_for_pdp: str,
    out_dir: str = "reports/interpretation",
    plots_dir: str = "plots/interpretation",
) -> None:
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    perm_df = compute_permutation_importance(model, X, y)
    perm_path = os.path.join(out_dir, "permutation_importance.csv")
    perm_df.to_csv(perm_path, index=False, encoding="utf-8-sig")
    plot_permutation_importance(
        perm_df,
        path=os.path.join(plots_dir, "permutation_importance.png"),
        top_n=20,
    )

    pdp_df = compute_pdp_values(model, X, feature_for_pdp)
    pdp_path = os.path.join(out_dir, f"pdp_{feature_for_pdp}.csv")
    pdp_df.to_csv(pdp_path, index=False, encoding="utf-8-sig")
    plot_pdp(
        model,
        X,
        feature_for_pdp,
        path=os.path.join(plots_dir, f"pdp_{feature_for_pdp}.png"),
    )

    ice_df = compute_ice_values(model, X, feature_for_pdp)
    ice_path = os.path.join(out_dir, f"ice_{feature_for_pdp}.csv")
    ice_df.to_csv(ice_path, index=False, encoding="utf-8-sig")
    plot_ice(
        model,
        X,
        feature_for_pdp,
        path=os.path.join(plots_dir, f"ice_{feature_for_pdp}.png"),
    )

    if _HAS_SHAP:
        shap_imp = compute_shap_importance(model, X)
        if not shap_imp.empty:
            shap_imp_path = os.path.join(out_dir, "shap_importance.csv")
            shap_imp.to_csv(shap_imp_path, index=False, encoding="utf-8-sig")
            plot_shap_importance(
                model,
                X,
                path=os.path.join(plots_dir, "shap_importance.png"),
            )
            plot_shap_summary(
                model,
                X,
                path=os.path.join(plots_dir, "shap_summary.png"),
            )
