import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA


def select_numeric(df: pd.DataFrame, exclude: List[str] | None = None) -> pd.DataFrame:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return df[cols].copy()


def scale_standard(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    s = StandardScaler()
    return s.fit_transform(X_train), s.transform(X_test), s


def scale_minmax(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    s = MinMaxScaler()
    return s.fit_transform(X_train), s.transform(X_test), s


def scale_robust(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, RobustScaler]:
    s = RobustScaler()
    return s.fit_transform(X_train), s.transform(X_test), s


def log_transform(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df_new = df.copy()
    for c in cols:
        s = df_new[c]
        s = s.replace([np.inf, -np.inf], np.nan)
        if (s <= 0).any():
            continue
        df_new[c] = np.log1p(s)
    return df_new


def remove_low_variance(
    df: pd.DataFrame,
    threshold: float = 0.0,
) -> pd.DataFrame:
    df_new = df.copy()
    cols = df_new.columns
    to_drop = []
    for c in cols:
        if df_new[c].std() <= threshold:
            to_drop.append(c)
    return df_new.drop(columns=to_drop)


def remove_high_correlation(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    corr = df.corr(numeric_only=True).abs()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    upper = corr.where(mask)
    to_drop = [
        col
        for col in upper.columns
        if any(upper[col] > threshold)
    ]
    return df.drop(columns=to_drop)


def create_polynomial(
    df: pd.DataFrame,
    degree: int = 2,
) -> Tuple[pd.DataFrame, PolynomialFeatures]:
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
    )
    arr = poly.fit_transform(df)
    cols = poly.get_feature_names_out(df.columns)
    df_poly = pd.DataFrame(arr, columns=cols, index=df.index)
    return df_poly, poly


def apply_pca(
    df: pd.DataFrame,
    n: int = 10,
) -> Tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=n, random_state=42)
    arr = pca.fit_transform(df)
    cols = [f"pca_{i+1}" for i in range(n)]
    df_pca = pd.DataFrame(arr, columns=cols, index=df.index)
    return df_pca, pca


def bin_rank(df: pd.DataFrame) -> pd.DataFrame:
    if "rank" not in df.columns:
        return df
    df_new = df.copy()
    df_new["rank_bin"] = pd.cut(
        df_new["rank"],
        bins=[0, 10, 25, 50, 100, np.inf],
        labels=["1-10", "11-25", "26-50", "51-100", "100+"],
    )
    return df_new


def quantile_transform(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    numeric = df_new.select_dtypes(include=[np.number]).columns
    for c in numeric:
        df_new[c] = df_new[c].rank(pct=True)
    return df_new


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    numeric = df_new.select_dtypes(include=[np.number]).columns
    for c in numeric:
        series = df_new[c]
        min_val = series.min()
        max_val = series.max()
        if max_val > min_val:
            df_new[c] = (series - min_val) / (max_val - min_val)
    return df_new


def engineer_features(
    df: pd.DataFrame,
    use_log: bool = True,
    use_poly: bool = False,
    poly_degree: int = 2,
    use_pca: bool = False,
    pca_dim: int = 10,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    df_num = select_numeric(df, exclude=["rank"])
    result["numeric_base"] = df_num.copy()

    if use_log:
        df_log = log_transform(df_num, df_num.columns.tolist())
        result["log"] = df_log
    else:
        df_log = df_num.copy()

    df_var = remove_low_variance(df_log, threshold=0.0)
    df_corr = remove_high_correlation(df_var, threshold=0.95)
    result["filtered"] = df_corr

    if use_poly:
        df_poly, poly_model = create_polynomial(df_corr, degree=poly_degree)
        result["poly"] = df_poly
        result["poly_model"] = poly_model
        df_final = df_poly
    else:
        df_final = df_corr.copy()

    if use_pca:
        df_pca, pca_model = apply_pca(df_final, n=pca_dim)
        result["pca"] = df_pca
        result["pca_model"] = pca_model
    else:
        result["pca"] = None
        result["pca_model"] = None

    result["final"] = df_final
    return result
