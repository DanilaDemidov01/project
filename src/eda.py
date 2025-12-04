import os
from typing import List

import numpy as np
import pandas as pd


def save_basic_info(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "basic_info.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Строк: {df.shape[0]}\n")
        f.write(f"Столбцов: {df.shape[1]}\n\n")
        f.write("Типы данных:\n")
        f.write(str(df.dtypes))
        f.write("\n\nПервые строки:\n")
        f.write(str(df.head()))


def save_describe(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    desc = df.describe(include="all").transpose()
    path = os.path.join(out_dir, "describe.csv")
    desc.to_csv(path, encoding="utf-8-sig")


def save_missing_report(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    missing_count = df.isna().sum()
    missing_ratio = df.isna().mean()
    report = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_ratio": missing_ratio,
        }
    )
    path = os.path.join(out_dir, "missing_report.csv")
    report.to_csv(path, encoding="utf-8-sig")


def save_unique_counts(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    uniques = df.nunique()
    path = os.path.join(out_dir, "unique_counts.csv")
    uniques.to_csv(path, header=["unique_count"], encoding="utf-8-sig")


def save_numeric_summary(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    numeric = df.select_dtypes(include=[np.number])
    summary = pd.DataFrame(
        {
            "min": numeric.min(),
            "max": numeric.max(),
            "mean": numeric.mean(),
            "median": numeric.median(),
            "std": numeric.std(),
            "skew": numeric.skew(),
            "kurtosis": numeric.kurtosis(),
        }
    )
    path = os.path.join(out_dir, "numeric_summary.csv")
    summary.to_csv(path, encoding="utf-8-sig")


def save_correlation_matrix(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    corr = df.corr(numeric_only=True)
    path = os.path.join(out_dir, "correlation_matrix.csv")
    corr.to_csv(path, encoding="utf-8-sig")


def save_top_correlated_with_rank(df: pd.DataFrame, out_dir: str, top_n: int = 20) -> None:
    os.makedirs(out_dir, exist_ok=True)
    corr = df.corr(numeric_only=True)
    if "rank" not in corr.columns:
        return
    series = corr["rank"].drop("rank", errors="ignore").sort_values(ascending=False)
    series = series.head(top_n)
    path = os.path.join(out_dir, "top_correlated_with_rank.csv")
    series.to_csv(path, header=["correlation_with_rank"], encoding="utf-8-sig")


def save_rank_groups(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if "rank" not in df.columns:
        return

    # Пробуем привести rank к числовому виду
    rank_numeric = pd.to_numeric(df["rank"], errors="coerce")

    # Если после приведения всё пропало, шаг пропускаем
    if rank_numeric.isna().all():
        return

    df_work = df.copy()
    df_work["rank_numeric"] = rank_numeric
    df_work = df_work.dropna(subset=["rank_numeric"])

    # Если после очистки строк почти не осталось — тоже пропускаем
    if df_work.empty:
        return

    df_work["rank_group"] = pd.cut(
        df_work["rank_numeric"],
        bins=[0, 10, 25, 50, 100, np.inf],
        labels=["1-10", "11-25", "26-50", "51-100", "100+"],
    )

    numeric_cols = df_work.select_dtypes(include=[np.number]).columns
    grouped = df_work.groupby("rank_group")[numeric_cols].agg(
        ["mean", "median", "min", "max"]
    )

    path = os.path.join(out_dir, "rank_groups_summary.csv")
    grouped.to_csv(path, encoding="utf-8-sig")


def save_value_distributions(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    numeric = df.select_dtypes(include=[np.number])
    stats_list = []
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) == 0:
            continue
        q1 = s.quantile(0.25)
        q2 = s.quantile(0.5)
        q3 = s.quantile(0.75)
        stats_list.append(
            {
                "feature": col,
                "q1": q1,
                "median": q2,
                "q3": q3,
                "min": s.min(),
                "max": s.max(),
            }
        )
    if not stats_list:
        return
    df_stats = pd.DataFrame(stats_list)
    path = os.path.join(out_dir, "value_distributions.csv")
    df_stats.to_csv(path, index=False, encoding="utf-8-sig")


def save_category_summary(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    object_cols = df.select_dtypes(include=["object"]).columns
    rows: List[dict] = []
    for col in object_cols:
        vc = df[col].value_counts(dropna=False)
        for value, count in vc.items():
            rows.append(
                {
                    "column": col,
                    "value": str(value),
                    "count": count,
                    "ratio": count / len(df) if len(df) > 0 else 0.0,
                }
            )
    if not rows:
        return
    df_cat = pd.DataFrame(rows)
    path = os.path.join(out_dir, "category_summary.csv")
    df_cat.to_csv(path, index=False, encoding="utf-8-sig")


def run_full_eda(df: pd.DataFrame, out_dir: str = "reports/eda") -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_basic_info(df, out_dir)
    save_describe(df, out_dir)
    save_missing_report(df, out_dir)
    save_unique_counts(df, out_dir)
    save_numeric_summary(df, out_dir)
    save_correlation_matrix(df, out_dir)
    save_top_correlated_with_rank(df, out_dir)
    save_rank_groups(df, out_dir)
    save_value_distributions(df, out_dir)
    save_category_summary(df, out_dir)
