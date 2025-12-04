import os
import random
from typing import Any, Dict

import numpy as np


# Фиксация зерна для воспроизводимости
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# Создание каталога
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# Базовая конфигурация проекта
def get_default_config() -> Dict[str, Any]:
    return {
        # путь к датасету
        "data_path": os.path.join("data", "data.csv"),

        # путь к модели
        "model_path": os.path.join("models", "model.pkl"),

        # директория для графиков
        "plots_dir": "plots",

        # пропорция тестовой выборки
        "test_size": 0.2,

        # случайное зерно
        "random_state": 42,
    }


# Чтение текстового отчёта в список строк
def read_text_file(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


# Запись строк в файл
def write_text_file(path: str, lines: list[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# Проверка существования файла
def check_file_exists(path: str) -> bool:
    return os.path.isfile(path)


# Безопасная загрузка numpy файла (если понадобится)
def load_np(path: str) -> np.ndarray:
    return np.load(path)


# Безопасное сохранение numpy файла
def save_np(path: str, arr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)
