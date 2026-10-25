"""Evaluation metrics calculation for repeat benchmark.

Calculates:
- Overall Accuracy
- Precision (Macro & Weighted)
- Recall (Macro & Weighted)
- F1 Score (Macro & Weighted)
- Matthews Correlation Coefficient (MCC, multiclass)
- Invalid Output Rate
- Per-class classification metrics
- Confusion matrix heatmap saved as PNG
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)


def compute_metrics(
    predictions_df: pd.DataFrame,
    true_col: str = "true_label",
    pred_col: str = "pred_label",
) -> Dict[str, Any]:
    """Compute benchmark metrics from a dataframe of predictions."""
    y_true = predictions_df[true_col].astype(str).tolist()
    y_pred = predictions_df[pred_col].astype(str).tolist()
    total = len(y_true)

    # Count invalid predictions
    invalid_mask = [p == "INVALID" for p in y_pred]
    n_invalid = sum(invalid_mask)
    invalid_rate = n_invalid / total if total > 0 else 0.0

    # Unique labels in ground truth
    labels = sorted(list(set(y_true)))

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)

    # Macro & Weighted Precision, Recall, F1
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    # Matthews Correlation Coefficient
    # Note: MCC handles multiclass by projecting into contingency matrices
    mcc = matthews_corrcoef(y_true, y_pred)

    # Per-class report
    class_rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    metrics = {
        "total_samples": total,
        "valid_predictions": total - n_invalid,
        "invalid_predictions": n_invalid,
        "invalid_rate": round(invalid_rate, 4),
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(p_weighted), 4),
        "recall_weighted": round(float(r_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "mcc": round(float(mcc), 4),
        "per_class": class_rep,
    }
    return metrics


def plot_confusion_matrix(
    predictions_df: pd.DataFrame,
    output_png: Path,
    true_col: str = "true_label",
    pred_col: str = "pred_label",
    title: str = "Repeat Classification Confusion Matrix",
    max_classes_full_annot: int = 25,
) -> None:
    """Generate and save confusion matrix heatmap."""
    y_true = predictions_df[true_col].astype(str).tolist()
    y_pred = predictions_df[pred_col].astype(str).tolist()
    labels = sorted(list(set(y_true)))

    # Include INVALID in predicted axis if present
    pred_labels = list(labels)
    if "INVALID" in y_pred:
        pred_labels.append("INVALID")

    cm = confusion_matrix(y_true, y_pred, labels=pred_labels)
    # Normalize row-wise (recall per class)
    with np.errstate(all="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    n_classes = len(labels)
    fig_size = max(10, min(24, int(n_classes * 0.45)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    annot = n_classes <= max_classes_full_annot
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt=".2f" if annot else "",
        cmap="Blues",
        xticklabels=pred_labels,
        yticklabels=labels,
        cbar_kws={"label": "Normalized Recall"},
        ax=ax,
    )
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()


def print_summary_table(metrics: Dict[str, Any], task_name: str = "Benchmark") -> None:
    """Print clean terminal table with primary evaluation metrics."""
    print(f"\n=======================================================")
    print(f"  {task_name.upper()} RESULTS SUMMARY")
    print(f"=======================================================")
    print(f"Total Sequences Evaluated : {metrics['total_samples']}")
    print(f"Invalid Prediction Rate   : {metrics['invalid_rate']*100:.2f}% ({metrics['invalid_predictions']} samples)")
    print(f"Accuracy                  : {metrics['accuracy']:.4f}")
    print(f"Macro F1 Score            : {metrics['f1_macro']:.4f}")
    print(f"Weighted F1 Score         : {metrics['f1_weighted']:.4f}")
    print(f"Macro Precision           : {metrics['precision_macro']:.4f}")
    print(f"Macro Recall              : {metrics['recall_macro']:.4f}")
    print(f"MCC (Matthews Corr Coef)  : {metrics['mcc']:.4f}")
    print(f"=======================================================\n")
