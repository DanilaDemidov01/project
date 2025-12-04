import os
from typing import Dict, Any, Optional

import pandas as pd

from .metrics import metrics_table
from .advanced_metrics import full_advanced_report, grouped_rank_metrics, error_distribution
from .plots import (
    plot_corr,
    plot_feature_importance,
    plot_distributions,
    plot_boxplots,
    plot_errors,
    plot_scatter,
)
from .data_export import (
    export_df_csv,
    export_df_excel,
    export_df_markdown,
    export_metrics_json,
    export_metrics_csv,
    export_model_metadata,
    export_experiment_results,
    save_text_report,
    export_full_package,
)
from .report_utils import (
    save_metrics_report,
    save_experiment_report,
    save_feature_report,
    save_model_summary,
    save_table_as_markdown,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_paths(base_dir: str = "reports") -> Dict[str, str]:
    ensure_dir(base_dir)
    paths = {
        "base": base_dir,
        "metrics": os.path.join(base_dir, "metrics"),
        "experiments": os.path.join(base_dir, "experiments"),
        "features": os.path.join(base_dir, "features"),
        "tables": os.path.join(base_dir, "tables"),
        "text": os.path.join(base_dir, "text"),
    }
    for p in paths.values():
        ensure_dir(p)
    return paths


def save_basic_overview(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    lines = []
    lines.append("Общая информация о датасете")
    lines.append("---------------------------")
    lines.append(f"Строк: {df.shape[0]}")
    lines.append(f"Столбцов: {df.shape[1]}")
    lines.append("")
    lines.append("Столбцы:")
    for col in df.columns:
        lines.append(f"- {col}")
    path = os.path.join(out_dir, "dataset_overview.txt")
    save_text_report(lines, path)


def save_correlation_artifacts(df: pd.DataFrame, out_dir: str, plots_dir: str) -> None:
    ensure_dir(out_dir)
    ensure_dir(plots_dir)
    corr = df.corr(numeric_only=True)
    csv_path = os.path.join(out_dir, "correlation_matrix.csv")
    md_path = os.path.join(out_dir, "correlation_matrix.md")
    export_df_csv(corr, csv_path)
    export_df_markdown(corr, md_path)
    plot_corr(df, path=os.path.join(plots_dir, "correlation.png"))


def save_feature_artifacts(
    df: pd.DataFrame,
    model,
    out_dir: str,
    plots_dir: str,
) -> None:
    ensure_dir(out_dir)
    ensure_dir(plots_dir)
    X = df.drop(columns=["rank"])
    corr = df.corr(numeric_only=True)["rank"].drop("rank", errors="ignore").abs().sort_values(
        ascending=False
    )
    top_features = corr.index.tolist()
    save_feature_report(
        top_features,
        path=os.path.join(out_dir, "features_by_correlation.txt"),
    )
    plot_feature_importance(
        model,
        df,
        path=os.path.join(plots_dir, "feature_importance.png"),
    )


def save_distribution_artifacts(df: pd.DataFrame, out_dir: str, plots_dir: str) -> None:
    ensure_dir(out_dir)
    ensure_dir(plots_dir)
    desc = df.describe(include="all").transpose()
    csv_path = os.path.join(out_dir, "describe.csv")
    md_path = os.path.join(out_dir, "describe.md")
    export_df_csv(desc, csv_path)
    export_df_markdown(desc, md_path)
    plot_distributions(df, path=os.path.join(plots_dir, "distributions.png"))
    plot_boxplots(df, path=os.path.join(plots_dir, "boxplots.png"))


def save_error_artifacts(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: str,
    plots_dir: str,
) -> None:
    ensure_dir(out_dir)
    ensure_dir(plots_dir)
    mt = metrics_table(model, X_test, y_test)
    csv_path = os.path.join(out_dir, "errors_table.csv")
    md_path = os.path.join(out_dir, "errors_table.md")
    export_df_csv(mt, csv_path)
    export_df_markdown(mt, md_path)
    plot_errors(model, X_test, y_test, path=os.path.join(plots_dir, "errors.png"))
    plot_scatter(
        pd.DataFrame({"rank": y_test}),
        path=os.path.join(plots_dir, "scatter_target.png"),
    )


def save_advanced_metrics(
    y_true: pd.Series,
    y_pred,
    out_dir: str,
) -> None:
    ensure_dir(out_dir)
    y_pred_arr = getattr(y_pred, "to_numpy", lambda: y_pred)()
    adv = full_advanced_report(y_true, y_pred_arr)
    grouped = grouped_rank_metrics(y_true, y_pred_arr)
    dist = error_distribution(y_true, y_pred_arr)

    metrics_json_path = os.path.join(out_dir, "advanced_metrics.json")
    metrics_csv_path = os.path.join(out_dir, "advanced_metrics.csv")
    group_csv_path = os.path.join(out_dir, "grouped_metrics.csv")
    dist_csv_path = os.path.join(out_dir, "error_distribution.csv")

    export_metrics_json(adv, metrics_json_path)
    export_metrics_csv(adv, metrics_csv_path)
    if not grouped.empty:
        export_df_csv(grouped, group_csv_path, index=False)
    if not dist.empty:
        export_df_csv(dist, dist_csv_path, index=False)

    lines = []
    lines.append("Расширенные метрики модели")
    lines.append("--------------------------")
    for k, v in adv.items():
        lines.append(f"{k}: {v}")
    path = os.path.join(out_dir, "advanced_metrics.txt")
    save_text_report(lines, path)


def save_main_metrics(
    basic_metrics: Dict[str, Any],
    advanced_metrics: Dict[str, Any],
    mape_value: float,
    out_dir: str,
) -> None:
    ensure_dir(out_dir)
    metrics_all = dict(basic_metrics)
    metrics_all["mape"] = mape_value
    metrics_all.update(advanced_metrics)
    path_txt = os.path.join(out_dir, "metrics_full.txt")
    path_csv = os.path.join(out_dir, "metrics_full.csv")
    export_metrics_csv(metrics_all, path_csv)
    save_metrics_report(metrics_all, metrics_all.get("mape", 0.0), path=path_txt)


def generate_text_summary(
    model_name: str,
    basic_metrics: Dict[str, Any],
    advanced_metrics: Dict[str, Any],
    out_dir: str,
) -> None:
    ensure_dir(out_dir)
    lines = []
    lines.append("Итоговое краткое описание результатов")
    lines.append("-------------------------------------")
    lines.append(f"Модель: {model_name}")
    lines.append("")
    lines.append("Основные метрики:")
    for k, v in basic_metrics.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Расширенные метрики:")
    for k, v in advanced_metrics.items():
        lines.append(f"- {k}: {v}")
    path = os.path.join(out_dir, "summary.txt")
    save_text_report(lines, path)
    save_model_summary(model_name, basic_metrics, path=os.path.join(out_dir, "model_summary.txt"))


def generate_markdown_overview(
    df: pd.DataFrame,
    experiments: Optional[pd.DataFrame],
    out_dir: str,
) -> None:
    ensure_dir(out_dir)
    lines = []
    lines.append("# Итоговый обзор проекта")
    lines.append("")
    lines.append("## Датасет")
    lines.append("")
    lines.append(f"- Строк: {df.shape[0]}")
    lines.append(f"- Столбцов: {df.shape[1]}")
    lines.append("")
    lines.append("## Столбцы датасета")
    lines.append("")
    for col in df.columns:
        lines.append(f"- {col}")
    lines.append("")
    if experiments is not None and not experiments.empty:
        lines.append("## Эксперименты с моделями")
        lines.append("")
        lines.append(experiments.to_markdown(index=False))
    path = os.path.join(out_dir, "overview.md")
    text = "\n".join(lines)
    save_text_report(text.split("\n"), path)


def generate_full_report(
    df: pd.DataFrame,
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    basic_metrics: Dict[str, Any],
    experiments: Optional[pd.DataFrame] = None,
    output_dir: str = "reports",
) -> None:
    paths = build_paths(output_dir)

    save_basic_overview(df, paths["text"])
    save_correlation_artifacts(
        df,
        out_dir=paths["tables"],
        plots_dir=os.path.join("plots", "correlation"),
    )
    save_distribution_artifacts(
        df,
        out_dir=paths["tables"],
        plots_dir=os.path.join("plots", "distributions"),
    )
    save_feature_artifacts(
        df,
        model,
        out_dir=paths["features"],
        plots_dir=os.path.join("plots", "features"),
    )
    save_error_artifacts(
        model,
        X_test,
        y_test,
        out_dir=paths["metrics"],
        plots_dir=os.path.join("plots", "errors"),
    )

    preds = model.predict(X_test)
    adv = full_advanced_report(y_test, preds)
    save_advanced_metrics(y_test, preds, out_dir=paths["metrics"])
    save_main_metrics(basic_metrics, adv, basic_metrics.get("mape", 0.0), out_dir=paths["metrics"])
    generate_text_summary(model_name, basic_metrics, adv, out_dir=paths["text"])

    if experiments is not None and not experiments.empty:
        save_experiment_report(
            experiments,
            path=os.path.join(paths["experiments"], "experiments.txt"),
        )
        export_experiment_results(
            experiments,
            base_path=paths["experiments"],
        )

    generate_markdown_overview(df, experiments, out_dir=paths["text"])

    export_full_package(
        reports_dir=output_dir,
        plots_dir="plots",
        archive_path="exports/project_package.zip",
    )
