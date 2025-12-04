import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_dir_for_file(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def plot_corr(df: pd.DataFrame, path: str = "plots/correlation.png") -> None:
    """
    Тепловая карта корреляций. Если числовых признаков нет или матрица пустая,
    функция просто ничего не рисует и не падает.
    """
    corr = df.corr(numeric_only=True)

    if corr.empty or corr.shape[0] == 0:
        _ensure_dir_for_file(path)
        # Можно сохранить пустой файл или просто пропустить построение
        return

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm")
    _ensure_dir_for_file(path)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_feature_importance(model, df: pd.DataFrame, path: str = "plots/feature_importance.png") -> None:
    """
    Гистограмма важности признаков на основе feature_importances_.
    Ожидается, что в df есть столбец 'rank', который удаляется из X.
    """
    if "rank" in df.columns:
        features = df.drop(columns=["rank"]).columns
    else:
        features = df.columns

    # не все модели имеют feature_importances_
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    data = pd.DataFrame({"feature": features, "importance": importances})
    data = data.sort_values("importance", ascending=False)

    top = data.head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top, x="importance", y="feature")
    _ensure_dir_for_file(path)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_distributions(df: pd.DataFrame, path: str = "plots/distributions.png") -> None:
    """
    Объединённые гистограммы распределений числовых признаков.
    Если числовых признаков нет — функция спокойно завершает работу.
    """
    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] == 0:
        _ensure_dir_for_file(path)
        return

    cols = numeric.columns.tolist()
    n_cols = min(4, len(cols))
    n_rows = int(np.ceil(len(cols) / n_cols))

    plt.figure(figsize=(4 * n_cols, 3 * n_rows))

    for i, col in enumerate(cols, start=1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(numeric[col].dropna(), kde=False)
        plt.title(col)

    _ensure_dir_for_file(path)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_boxplots(df: pd.DataFrame, path: str = "plots/boxplots.png") -> None:
    """
    Boxplot'ы числовых признаков. Если числовых признаков нет — ничего не рисует.
    """
    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] == 0:
        _ensure_dir_for_file(path)
        return

    plt.figure(figsize=(max(8, 0.7 * numeric.shape[1]), 6))
    sns.boxplot(data=numeric, orient="h")
    _ensure_dir_for_file(path)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_errors(model, X_test: pd.DataFrame, y_test: pd.Series, path: str = "plots/errors.png") -> None:
    """
    График ошибок: истинный рейтинг vs предсказанный рейтинг.
    """
    preds = model.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Истинный рейтинг")
    plt.ylabel("Предсказанный рейтинг")
    plt.grid(True)
    _ensure_dir_for_file(path)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scatter(df: pd.DataFrame, path: str = "plots/scatter.png") -> None:
    """
    Простая scatter-диаграмма: rank vs первый числовой признак.
    Если нет rank или нет числовых признаков — функция завершается.
    """
    if "rank" not in df.columns:
        return

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return

    # берём первый числовой столбец, отличный от rank
    cols = [c for c in numeric.columns if c != "rank"]
    if not cols:
        return

    x_col = cols[0]

    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df["rank"], alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel("rank")
    plt.grid(True)
    _ensure_dir_for_file(path)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
