import os
import sys
from typing import Any, Dict

# ---------------------------
# Добавляем src в PYTHONPATH
# ---------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------
# Импорты модулей
# ---------------------------
from utils import set_seed, get_default_config, ensure_dir
from src.preprocessing import load_data, split_data, prepare_features
from src.data_validation import validate_data
from src.eda import run_full_eda
from src.model_training import train_model, save_model, load_model
from src.metrics import evaluate, mape, metrics_table
from src.plots import (
    plot_corr,
    plot_feature_importance,
    plot_distributions,
    plot_boxplots,
    plot_errors,
    plot_scatter,
)
from src.hyperparameter_tuning import run_full_hyperparameter_tuning
from src.report_utils import (
    save_metrics_report,
    save_experiment_report,
    save_feature_report,
)

# ---------------------------
# Блоки пайплайна
# ---------------------------

def run_validation(cfg: Dict[str, Any]) -> None:
    df = load_data(cfg["data_path"])
    issues = validate_data(df)
    ensure_dir("reports")
    with open("reports/data_validation.txt", "w", encoding="utf-8") as f:
        for line in issues:
            f.write(line + "\n")


def run_eda(cfg: Dict[str, Any]) -> None:
    df = load_data(cfg["data_path"])
    ensure_dir("reports/eda")
    run_full_eda(df, out_dir="reports/eda")
    ensure_dir(cfg["plots_dir"])
    plot_corr(df, path=os.path.join(cfg["plots_dir"], "correlation.png"))
    plot_distributions(df, path=os.path.join(cfg["plots_dir"], "distributions.png"))
    plot_boxplots(df, path=os.path.join(cfg["plots_dir"], "boxplots.png"))


def run_training(cfg: Dict[str, Any]) -> Dict[str, Any]:
    df = load_data(cfg["data_path"])
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )

    model = train_model(X_train, y_train, X_test, y_test)
    ensure_dir("models")
    save_model(model, cfg["model_path"])

    eval_res = evaluate(model, X_test, y_test)
    mape_val = mape(model, X_test, y_test)
    mt = metrics_table(model, X_test, y_test)

    ensure_dir("reports")
    mt.to_csv(os.path.join("reports", "metrics_table.csv"), index=False)
    save_metrics_report(eval_res, mape_val, path=os.path.join("reports", "metrics.txt"))

    return {"df": df, "model": model, "X_test": X_test, "y_test": y_test}


def run_plots(cfg: Dict[str, Any], df, model) -> None:
    X, y = prepare_features(df)
    tmp = X.copy()
    tmp["rank"] = y

    ensure_dir(cfg["plots_dir"])

    plot_feature_importance(
        model, tmp, path=os.path.join(cfg["plots_dir"], "feature_importance.png")
    )

    _, X_test, _, y_test = split_data(
        df, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )

    plot_errors(model, X_test, y_test, path=os.path.join(cfg["plots_dir"], "errors.png"))
    plot_scatter(tmp, path=os.path.join(cfg["plots_dir"], "scatter.png"))


def run_model_experiments(cfg: Dict[str, Any]) -> None:
    df = load_data(cfg["data_path"])
    X, y = prepare_features(df)

    best, full = run_full_hyperparameter_tuning(X, y)

    ensure_dir("reports")
    best["summary"].to_csv("reports/experiments_summary.csv", index=False)
    save_experiment_report(best["summary"], path=os.path.join("reports", "experiments.txt"))


def run_feature_summary(cfg: Dict[str, Any]) -> None:
    df = load_data(cfg["data_path"])
    _X, _y = prepare_features(df)

    corr_all = df.corr(numeric_only=True)
    if "rank" not in corr_all.columns:
        return

    corr = corr_all["rank"].abs().sort_values(ascending=False)
    top_features = corr.index.tolist()

    ensure_dir("reports")
    save_feature_report(top_features, path=os.path.join("reports", "features.txt"))



# ---------------------------
# Полный пайплайн
# ---------------------------

def run_full_pipeline() -> None:
    set_seed(42)
    cfg = get_default_config()

    ensure_dir(cfg["plots_dir"])
    ensure_dir("reports")

    run_validation(cfg)
    run_eda(cfg)
    train_res = run_training(cfg)
    run_plots(cfg, train_res["df"], train_res["model"])
    run_model_experiments(cfg)
    run_feature_summary(cfg)


if __name__ == "__main__":
    run_full_pipeline()
