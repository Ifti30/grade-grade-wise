#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime as dt
import json
import os
import sys
import zipfile
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    _PLOT_IMPORT_ERROR = exc
else:
    _PLOT_IMPORT_ERROR = None

try:
    import pandas as pd
except Exception as exc:
    pd = None
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None


def emit_result(payload):
    print("__RESULT__" + json.dumps(payload, separators=(",", ":")), flush=True)


def read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def configure_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "0.85",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6
    })


def save_figure(fig, base_path: Path):
    pdf_path = base_path.with_suffix(".pdf")
    png_path = base_path.with_suffix(".png")
    pdf_ok = False
    try:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        pdf_ok = True
    except Exception:
        pdf_ok = False
    try:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    return pdf_ok, pdf_path, png_path


def plot_feature_importance(items, out_path: Path):
    if not items:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No feature importance data", ha="center", va="center")
        ax.set_axis_off()
        return save_figure(fig, out_path)
    items = sorted(items, key=lambda item: abs(item.get("importance", 0)), reverse=True)
    features = [item.get("feature", "") for item in items]
    values = [float(item.get("importance", 0)) for item in items]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(features))))
    ax.barh(features[::-1], values[::-1], color="#4b4b4b")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    ax.grid(True, axis="x")
    return save_figure(fig, out_path)


def plot_confusion_matrix(matrix, labels, out_path: Path):
    if not matrix or not labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No confusion matrix data", ha="center", va="center")
        ax.set_axis_off()
        return save_figure(fig, out_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Greys")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Risk Classification Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    return save_figure(fig, out_path)


def plot_predicted_vs_actual(points, out_path: Path, title: str):
    if not points:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No prediction samples", ha="center", va="center")
        ax.set_axis_off()
        return save_figure(fig, out_path)
    actual = [p.get("actual") for p in points if isinstance(p, dict)]
    predicted = [p.get("predicted") for p in points if isinstance(p, dict)]
    if not actual or not predicted:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No prediction samples", ha="center", va="center")
        ax.set_axis_off()
        return save_figure(fig, out_path)
    min_val = min(actual + predicted)
    max_val = max(actual + predicted)
    if min_val == max_val:
        min_val -= 1
        max_val += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, predicted, color="#5b5b5b", alpha=0.7, edgecolors="none")
    ax.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    return save_figure(fig, out_path)


def plot_learning_curve(curve, out_path: Path):
    if not curve:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No learning curve data", ha="center", va="center")
        ax.set_axis_off()
        return save_figure(fig, out_path)
    train = curve.get("train") or []
    valid = curve.get("valid") or []
    length = max(len(train), len(valid))
    if length == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No learning curve data", ha="center", va="center")
        ax.set_axis_off()
        return save_figure(fig, out_path)
    epochs = list(range(1, length + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    if train:
        ax.plot(epochs[: len(train)], train, label="Train RMSE", color="black", linewidth=1.4)
    if valid:
        ax.plot(epochs[: len(valid)], valid, label="Validation RMSE", color="#6a6a6a", linewidth=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Learning Curve")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5)
    return save_figure(fig, out_path)


def build_class_metrics(matrix, labels):
    rows = []
    if not matrix or not labels:
        return rows
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append({
            "Class": label,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })
    return rows


def write_table(df, base_path: Path, include_index=False):
    csv_path = base_path.with_suffix(".csv")
    tex_path = base_path.with_suffix(".tex")
    df.to_csv(csv_path, index=include_index)
    df.to_latex(tex_path, index=include_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    report = read_json(out_dir / "report.json")
    meta = read_json(out_dir / "metadata.json")
    config = read_json(out_dir / "config.json") or {}

    if report is None:
        emit_result({"status": "error", "error": "Missing report.json"})
        return 1
    if meta is None:
        emit_result({"status": "error", "error": "Missing metadata.json"})
        return 1
    if plt is None:
        emit_result({"status": "error", "error": f"matplotlib unavailable: {_PLOT_IMPORT_ERROR}"})
        return 1
    if pd is None:
        emit_result({"status": "error", "error": f"pandas unavailable: {_PANDAS_IMPORT_ERROR}"})
        return 1

    regression = report.get("regression") or {}
    final_reg = regression.get("final_cgpa") or {}
    next_reg = regression.get("next_sem_cgpa") or {}
    best_model = final_reg.get("bestModel") or meta.get("best_model") or meta.get("bestModel")

    if not best_model:
        emit_result({"status": "error", "error": "Best model not found in report metadata."})
        return 1

    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root_dir = out_dir / f"thesis_results_{timestamp}"
    figures_dir = root_dir / "figures"
    tables_dir = root_dir / "tables"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    configure_plot_style()

    feature_importance = (final_reg.get("metrics") or {}).get("featureImportance") or {}
    plot_feature_importance(
        feature_importance.get(best_model, []),
        figures_dir / "feature_importance"
    )

    classification = report.get("classification") or {}
    plot_confusion_matrix(
        classification.get("confusion_matrix") or [],
        classification.get("labels") or [],
        figures_dir / "confusion_matrix"
    )

    final_predictions = (final_reg.get("metrics") or {}).get("predictions") or {}
    plot_predicted_vs_actual(
        final_predictions.get(best_model, []),
        figures_dir / "predicted_vs_actual_final",
        "Predicted vs Actual (Final CGPA)"
    )

    next_predictions = (next_reg.get("metrics") or {}).get("predictions") or {}
    plot_predicted_vs_actual(
        next_predictions.get(best_model, []),
        figures_dir / "predicted_vs_actual_next",
        "Predicted vs Actual (Next-Semester CGPA)"
    )

    learning_curves = (final_reg.get("metrics") or {}).get("learningCurves") or {}
    plot_learning_curve(
        learning_curves.get(best_model) or {},
        figures_dir / "learning_curve"
    )

    dataset_stats = (report.get("dataset") or {}).get("stats") or {}
    final_split = final_reg.get("split") or {}
    next_split = next_reg.get("split") or {}
    dataset_rows = [
        {"Metric": "Total students", "Value": dataset_stats.get("students_total")},
        {"Metric": "Total samples (final CGPA)", "Value": dataset_stats.get("rows_final")},
        {"Metric": "Total samples (next-semester CGPA)", "Value": dataset_stats.get("rows_next")},
        {"Metric": "Train samples (final CGPA)", "Value": final_split.get("train_size")},
        {"Metric": "Test samples (final CGPA)", "Value": final_split.get("test_size")},
        {"Metric": "Train samples (next-semester CGPA)", "Value": next_split.get("train_size")},
        {"Metric": "Test samples (next-semester CGPA)", "Value": next_split.get("test_size")}
    ]
    dataset_df = pd.DataFrame(dataset_rows)
    write_table(dataset_df, tables_dir / "dataset_overview", include_index=False)

    def build_perf_table(metrics_block):
        models = metrics_block.get("models") or {}
        rows = []
        for name, values in models.items():
            rows.append({
                "Model": name,
                "MAE": values.get("mae"),
                "RMSE": values.get("rmse"),
                "R²": values.get("r2")
            })
        return pd.DataFrame(rows)

    final_perf = build_perf_table(final_reg.get("metrics") or {})
    write_table(final_perf, tables_dir / "final_cgpa_results", include_index=False)

    next_perf = build_perf_table(next_reg.get("metrics") or {})
    write_table(next_perf, tables_dir / "next_sem_results", include_index=False)

    class_metrics = build_class_metrics(
        classification.get("confusion_matrix") or [],
        classification.get("labels") or []
    )
    risk_df = pd.DataFrame(class_metrics)
    write_table(risk_df, tables_dir / "risk_classification_metrics", include_index=False)

    confusion_matrix = classification.get("confusion_matrix") or []
    labels = classification.get("labels") or []
    cm_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    write_table(cm_df, tables_dir / "confusion_matrix", include_index=True)

    evaluation_metrics = {
        "final_cgpa": (final_reg.get("metrics") or {}).get("models", {}).get(best_model),
        "next_sem_cgpa": (next_reg.get("metrics") or {}).get("models", {}).get(best_model),
        "classification": {
            "accuracy": classification.get("accuracy"),
            "precision_macro": classification.get("precision_macro"),
            "recall_macro": classification.get("recall_macro"),
            "f1_macro": classification.get("f1_macro"),
            "precision_weighted": classification.get("precision_weighted"),
            "recall_weighted": classification.get("recall_weighted"),
            "f1_weighted": classification.get("f1_weighted")
        }
    }
    export_metadata = {
        "best_model": best_model,
        "hyperparameters": config,
        "dataset_size": dataset_stats,
        "feature_list": meta.get("feature_order") or meta.get("featureOrder") or [],
        "evaluation_metrics": evaluation_metrics,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "random_seed": config.get("RANDOM_SEED")
    }
    with open(root_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(export_metadata, handle, indent=2)

    zip_path = out_dir / f"thesis_results_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in root_dir.rglob("*"):
            if path.is_file():
                zipf.write(path, arcname=path.relative_to(root_dir))

    emit_result({"status": "ok", "zipPath": str(zip_path), "timestamp": timestamp})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        emit_result({"status": "error", "error": str(exc)})
        sys.exit(1)
