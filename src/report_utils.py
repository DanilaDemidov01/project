import os
from typing import Dict, Any, List

import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_metrics_report(
    metrics: Dict[str, Any],
    mape_value: float,
    path: str = "reports/metrics.txt",
) -> None:
    _ensure_dir(os.path.dirname(path))
    lines = []
    lines.append("Метрики модели")
    lines.append("----------------")
    for name, value in metrics.items():
        lines.append(f"{name.upper()}: {value}")
    lines.append(f"MAPE: {mape_value}")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_experiment_report(
    experiments: pd.DataFrame,
    path: str = "reports/experiments.txt",
) -> None:
    _ensure_dir(os.path.dirname(path))
    lines = []
    lines.append("Результаты экспериментов с моделями")
    lines.append("-----------------------------------")
    if "model" in experiments.columns:
        models = experiments["model"].unique().tolist()
        lines.append("Использованные модели:")
        for m in models:
            lines.append(f"- {m}")
        lines.append("")
    metric_cols = [c for c in experiments.columns if c not in ["model", "params"]]
    for idx, row in experiments.iterrows():
        model_name = row.get("model", f"model_{idx}")
        lines.append(f"Модель: {model_name}")
        params = row.get("params", "")
        if isinstance(params, dict):
            lines.append("Параметры:")
            for k, v in params.items():
                lines.append(f"  {k}: {v}")
        elif isinstance(params, str) and params:
            lines.append(f"Параметры: {params}")
        for col in metric_cols:
            if col in ("model", "params"):
                continue
            value = row[col]
            lines.append(f"{col}: {value}")
        lines.append("")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_feature_report(
    features: List[str],
    path: str = "reports/features.txt",
) -> None:
    _ensure_dir(os.path.dirname(path))
    lines = []
    lines.append("Список признаков по значимости или корреляции")
    lines.append("---------------------------------------------")
    for i, name in enumerate(features, 1):
        lines.append(f"{i}. {name}")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_model_summary(
    model_name: str,
    metrics: Dict[str, Any],
    path: str = "reports/model_summary.txt",
) -> None:
    _ensure_dir(os.path.dirname(path))
    lines = []
    lines.append("Итоговая модель")
    lines.append("---------------")
    lines.append(f"Модель: {model_name}")
    lines.append("")
    lines.append("Метрики:")
    for k, v in metrics.items():
        lines.append(f"{k}: {v}")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_table_as_markdown(
    df: pd.DataFrame,
    path: str = "reports/table.md",
    title: str | None = None,
) -> None:
    _ensure_dir(os.path.dirname(path))
    lines = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append(df.to_markdown(index=False))
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
