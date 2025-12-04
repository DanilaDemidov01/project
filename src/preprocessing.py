import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import get_default_config


def load_data(path: str | None = None) -> pd.DataFrame:
    """
    Загрузка датасета из CSV с автоматическим подбором кодировки и разделителя.
    Сначала берётся путь из конфигурации, если не передан явно.
    """
    if path is None:
        cfg = get_default_config()
        path = cfg["data_path"]

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл с данными не найден: {path}")

    # варианты кодировок и разделителей, которые будем пробовать
    encodings = ["cp1251", "utf-8-sig", "utf-8"]
    seps = [";", ","]

    last_error: Exception | None = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                # Если файл прочитался и хотя бы один столбец есть — считаем успехом
                if df.shape[1] > 0:
                    df = df.drop_duplicates().reset_index(drop=True)
                    return df
            except Exception as e:  # UnicodeDecodeError, ParserError и т.п.
                last_error = e
                continue

    # если ни один вариант не сработал — явно сообщаем об ошибке
    raise RuntimeError(
        f"Не удалось прочитать файл {path} ни с одной из кодировок/разделителей. "
        f"Последняя ошибка: {last_error}"
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая очистка таблицы:
    - удаление полностью пустых столбцов;
    - удаление полностью пустых строк;
    - обрезка пробелов в названиях столбцов.
    """
    df_clean = df.copy()
    df_clean.columns = [str(c).strip() for c in df_clean.columns]
    df_clean = df_clean.dropna(axis=1, how="all")
    df_clean = df_clean.dropna(axis=0, how="all")
    return df_clean.reset_index(drop=True)


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Определение целевого столбца для прогнозирования.
    Для датасета RAEX по умолчанию используется показатель e1.
    """
    # Варианты целевого признака в порядке приоритета
    candidates = [
        "e1",              # основной показатель для модели
        "rank",
        "Rank",
        "RANK",
        "raex_rank",
        "rating",
        "score",
        "Score",
        "Итоговый рейтинг",
        "Итоговый_рейтинг",
    ]

    for name in candidates:
        if name in df.columns:
            return name

    # Если ничего не найдено — явная ошибка
    raise ValueError(
        "Не удалось определить целевой столбец для прогнозирования. "
        "Убедитесь, что в датасете есть колонка 'e1' или добавьте нужное имя "
        "в список candidates в функции detect_target_column()."
    )


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Формирование X и y с очисткой целевого признака и приведением признаков:
    - очистка датафрейма;
    - поиск столбца рейтинга;
    - перевод рейтинга в число;
    - удаление строк с неопределённым рейтингом;
    - попытка привести объектные столбцы к числам;
    - формирование числовой матрицы признаков.
    Если после всех преобразований признаков не осталось, создаётся
    фиктивный числовой признак dummy_feature.
    """
    df_clean = clean_dataframe(df)

    target_col = detect_target_column(df_clean)

    y_raw = df_clean[target_col]
    y_num = pd.to_numeric(y_raw, errors="coerce")

    mask = ~y_num.isna()
    df_work = df_clean.loc[mask].copy()
    y = y_num.loc[mask]

    X = df_work.drop(columns=[target_col])

    # Пытаемся превратить объектные столбцы в числовые там, где это возможно
    X_conv = X.copy()
    for col in X_conv.columns:
        if X_conv[col].dtype == "object":
            s = X_conv[col].astype(str).str.strip().str.replace(" ", "", regex=False)
            s = s.str.replace(",", ".", regex=False)
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() > 0:
                X_conv[col] = num

    # После конвертации делим на числовые и категориальные
    num_cols = X_conv.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_conv.select_dtypes(include=["object"]).columns.tolist()

    X_num: pd.DataFrame

    if num_cols:
        # Обычный случай: есть числовые признаки
        X_num = X_conv[num_cols].copy()
    else:
        # Чисто категориальный случай
        if cat_cols:
            X_cat = X_conv[cat_cols].copy()
            X_cat = X_cat.dropna(axis=1, how="all")
            if X_cat.shape[1] > 0:
                X_num = pd.get_dummies(X_cat, drop_first=True)
            else:
                X_num = pd.DataFrame()
        else:
            X_num = pd.DataFrame()

    # Если после всех операций признаков всё равно нет — создаём фиктивный
    if X_num.shape[1] == 0:
        X_num = pd.DataFrame(
            {"dummy_feature": np.ones(len(df_work), dtype=float)},
            index=df_work.index,
        )

    # Заполняем пропуски медианой
    X_num = X_num.fillna(X_num.median(numeric_only=True))

    return X_num, y



def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разбиение на обучающую и тестовую выборки.
    """
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return X_train, X_test, y_train, y_test
