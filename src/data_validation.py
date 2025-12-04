import numpy as np
import pandas as pd
from typing import List, Any, Dict


def check_missing_values(df: pd.DataFrame) -> List[str]:
    issues = []
    missing = df.isnull().sum()
    for col, cnt in missing.items():
        if cnt > 0:
            issues.append(f"Пропуски в столбце '{col}': {cnt}")
    return issues


def check_duplicates(df: pd.DataFrame) -> List[str]:
    issues = []
    dups = df.duplicated().sum()
    if dups > 0:
        issues.append(f"Дубликатов строк: {dups}")
    return issues


def check_dtypes(df: pd.DataFrame) -> List[str]:
    issues = []
    for col, dtype in df.dtypes.items():
        if dtype not in [np.int64, np.float64, object]:
            issues.append(f"Подозрительный тип данных '{col}': {dtype}")
    return issues


def check_negative_values(df: pd.DataFrame, exclude: List[str] | None = None) -> List[str]:
    issues = []
    numeric = df.select_dtypes(include=[np.number])
    for col in numeric.columns:
        if exclude and col in exclude:
            continue
        if (numeric[col] < 0).any():
            issues.append(f"Негативные значения в '{col}'")
    return issues


def check_constant_columns(df: pd.DataFrame) -> List[str]:
    issues = []
    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append(f"Константный столбец '{col}' (1 уникальное значение)")
    return issues


def check_outliers(df: pd.DataFrame) -> List[str]:
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) < 10:
            continue
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outliers = values[(values < low) | (values > high)]
        if len(outliers) > 0:
            issues.append(f"Выбросы в столбце '{col}': {len(outliers)}")
    return issues


def check_value_ranges(df: pd.DataFrame) -> List[str]:
    issues = []
    numeric = df.select_dtypes(include=[np.number])
    for col in numeric.columns:
        min_val = numeric[col].min()
        max_val = numeric[col].max()
        if min_val == max_val:
            issues.append(f"Подозрительный диапазон '{col}': один уровень ({min_val})")
        if np.isinf(min_val) or np.isinf(max_val):
            issues.append(f"Бесконечные значения в '{col}'")
    return issues


def check_correlation_anomalies(df: pd.DataFrame, threshold: float = 0.98) -> List[str]:
    issues = []
    corr = df.corr(numeric_only=True)
    for col1 in corr.columns:
        for col2 in corr.columns:
            if col1 >= col2:
                continue
            if abs(corr.loc[col1, col2]) >= threshold:
                issues.append(
                    f"Сильная корреляция ({corr.loc[col1, col2]:.2f}) между '{col1}' и '{col2}'"
                )
    return issues


def validate_data(df: pd.DataFrame) -> List[str]:
    issues = []
    issues.extend(check_missing_values(df))
    issues.extend(check_duplicates(df))
    issues.extend(check_dtypes(df))
    issues.extend(check_negative_values(df, exclude=["rank"]))
    issues.extend(check_constant_columns(df))
    issues.extend(check_outliers(df))
    issues.extend(check_value_ranges(df))
    issues.extend(check_correlation_anomalies(df))
    if not issues:
        issues.append("Проблемы не обнаружены.")
    return issues
