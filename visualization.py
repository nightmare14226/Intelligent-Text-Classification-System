"""Create a visual dashboard from experiment results."""

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _annotate_bars(axis: Any, bars: Any, precision: int = 3) -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            f"{value:.{precision}f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_quality(axis: Any, models: list[tuple[str, dict[str, Any]]]) -> None:
    names = [result["model"] for _, result in models]
    accuracy = [result["test"]["accuracy"] for _, result in models]
    f1 = [result["test"]["f1_macro"] for _, result in models]
    positions = np.arange(len(names))
    width = 0.36

    accuracy_bars = axis.bar(
        positions - width / 2, accuracy, width, label="Accuracy"
    )
    f1_bars = axis.bar(positions + width / 2, f1, width, label="Macro F1")
    axis.set_title("Test classification quality")
    axis.set_ylabel("Score (0–1)")
    axis.set_ylim(0, 1.08)
    axis.set_xticks(positions, names, rotation=12, ha="right")
    axis.legend()
    _annotate_bars(axis, accuracy_bars)
    _annotate_bars(axis, f1_bars)


def _plot_runtime(
    axis: Any,
    models: list[tuple[str, dict[str, Any]]],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    names = [result["model"] for _, result in models]
    values = [result[metric] for _, result in models]
    bars = axis.bar(names, values)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=12)
    if len(values) > 1 and max(values) / max(min(values), 1e-9) >= 100:
        axis.set_yscale("log")
        axis.set_ylabel(f"{ylabel} (log scale)")
    _annotate_bars(axis, bars, precision=2)


def _plot_confusion_matrix(
    axis: Any, result: dict[str, Any]
) -> None:
    matrix = np.asarray(result["test"]["confusion_matrix"])
    image = axis.imshow(matrix, cmap="Blues")
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    axis.set_title(f"{result['model']} confusion matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1], ["Negative", "Positive"])
    axis.set_yticks([0, 1], ["Negative", "Positive"])
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _plot_bert_history(axis: Any, result: dict[str, Any]) -> None:
    history = result["history"]
    epochs = [entry["epoch"] for entry in history]
    losses = [entry["training_loss"] for entry in history]
    accuracy = [entry["validation"]["accuracy"] for entry in history]
    f1 = [entry["validation"]["f1_macro"] for entry in history]

    axis.plot(epochs, losses, marker="o", label="Training loss")
    axis.plot(epochs, accuracy, marker="o", label="Validation accuracy")
    axis.plot(epochs, f1, marker="o", label="Validation macro F1")
    axis.set_title("BERT learning curves")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Value")
    axis.set_xticks(epochs)
    axis.legend()


def create_result_dashboard(
    results: dict[str, Any],
    output_path: str | Path,
    show: bool = True,
) -> Path:
    """Save a PNG dashboard and optionally display it in a window."""
    models = [
        (key, results[key])
        for key in ("tfidf", "bert")
        if key in results
    ]
    if not models:
        raise ValueError("No model results are available to visualize.")

    panel_count = 3 + len(models)
    bert_result = results.get("bert")
    if bert_result and bert_result.get("history"):
        panel_count += 1

    columns = 3 if panel_count > 4 else 2
    rows = math.ceil(panel_count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 4.6 * rows),
        squeeze=False,
    )
    panels = list(axes.flat)

    _plot_quality(panels[0], models)
    _plot_runtime(
        panels[1],
        models,
        "training_seconds",
        "Training time",
        "Seconds",
    )
    _plot_runtime(
        panels[2],
        models,
        "milliseconds_per_sample",
        "Inference latency",
        "Milliseconds per sample",
    )

    next_panel = 3
    for _, result in models:
        _plot_confusion_matrix(panels[next_panel], result)
        next_panel += 1

    if bert_result and bert_result.get("history"):
        _plot_bert_history(panels[next_panel], bert_result)
        next_panel += 1

    for unused_axis in panels[next_panel:]:
        figure.delaxes(unused_axis)

    split_sizes = results.get("split_sizes", {})
    formatted_sizes = {
        name: f"{split_sizes[name]:,}" if name in split_sizes else "?"
        for name in ("train", "validation", "test")
    }
    subtitle = (
        f"IMDB — train {formatted_sizes['train']}, "
        f"validation {formatted_sizes['validation']}, "
        f"test {formatted_sizes['test']}"
    )
    figure.suptitle(
        f"TF-IDF vs BERT experiment results\n{subtitle}",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return path
