import os
import json
import shutil
from typing import Any, Dict, List, Optional

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_df_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index, encoding="utf-8-sig")


def export_df_excel(df: pd.DataFrame, path: str, index: bool = False) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_excel(path, index=index)


def export_df_json(df: pd.DataFrame, path: str, orient: str = "records") -> None:
    ensure_dir(os.path.dirname(path))
    df.to_json(path, orient=orient, force_ascii=False)


def export_df_markdown(df: pd.DataFrame, path: str, index: bool = False) -> None:
    ensure_dir(os.path.dirname(path))
    text = df.to_markdown(index=index)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def export_metrics_json(metrics: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def export_metrics_csv(metrics: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")


def export_model_metadata(
    model_name: str,
    params: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    path: str = "reports/model_metadata.json",
) -> None:
    ensure_dir(os.path.dirname(path))
    payload: Dict[str, Any] = {
        "model_name": model_name,
        "params": params,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_experiment_results(
    df: pd.DataFrame,
    base_path: str = "reports/experiments",
) -> None:
    ensure_dir(base_path)
    csv_path = os.path.join(base_path, "experiments.csv")
    xlsx_path = os.path.join(base_path, "experiments.xlsx")
    json_path = os.path.join(base_path, "experiments.json")
    md_path = os.path.join(base_path, "experiments.md")

    export_df_csv(df, csv_path, index=False)
    export_df_excel(df, xlsx_path, index=False)
    export_df_json(df, json_path, orient="records")
    export_df_markdown(df, md_path, index=False)


def save_text_report(lines: List[str], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def archive_reports(
    source_dir: str = "reports",
    archive_path: str = "reports_archive.zip",
) -> None:
    base_dir = os.path.dirname(archive_path)
    if base_dir:
        ensure_dir(base_dir)
    root, _ = os.path.splitext(archive_path)
    shutil.make_archive(root, "zip", source_dir)


def export_full_package(
    reports_dir: str = "reports",
    plots_dir: str = "plots",
    archive_path: str = "exports/project_package.zip",
) -> None:
    tmp_root = "tmp_export"
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)

    if os.path.exists(reports_dir):
        dst_reports = os.path.join(tmp_root, "reports")
        shutil.copytree(reports_dir, dst_reports)

    if os.path.exists(plots_dir):
        dst_plots = os.path.join(tmp_root, "plots")
        shutil.copytree(plots_dir, dst_plots)

    ensure_dir(os.path.dirname(archive_path))
    root, _ = os.path.splitext(archive_path)
    shutil.make_archive(root, "zip", tmp_root)
    shutil.rmtree(tmp_root)