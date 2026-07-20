"""Command-line runner for TF-IDF / BERT / DeBERTa comparisons."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bert_experiment import run_bert_experiment
from dataloader import DatasetSplits, load_imdb_splits
from deberta_experiment import DEFAULT_MODEL as DEFAULT_DEBERTA_MODEL
from deberta_experiment import run_deberta_experiment
from evaluation import save_results
from tfidf_experiment import run_tfidf_experiment
from visualization import create_result_dashboard

MODEL_CHOICES = (
    "both",
    "tfidf",
    "bert",
    "deberta",
    "bert-deberta",
    "all",
)


def _optional_size(value: int) -> int | None:
    return None if value == 0 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TF-IDF, BERT, and DeBERTa on identical IMDB splits."
        )
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default="both")
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bert-model", default="distilbert-base-uncased")
    parser.add_argument("--deberta-model", default=DEFAULT_DEBERTA_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="TensorBoard log root (default: OUTPUT_DIR/tensorboard).",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard event logging.",
    )
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument(
        "--no-show-plots",
        action="store_true",
        help="Save the result dashboard without opening its window.",
    )
    return parser.parse_args()


def _selected_models(choice: str) -> set[str]:
    if choice == "both":
        return {"tfidf", "bert"}
    if choice == "bert-deberta":
        return {"bert", "deberta"}
    if choice == "all":
        return {"tfidf", "bert", "deberta"}
    return {choice}


def _save_error_analysis(
    splits: DatasetSplits,
    predictions_by_model: dict[str, list[int]],
    output_path: Path,
) -> None:
    model_keys = list(predictions_by_model)
    if len(model_keys) < 2:
        return

    fieldnames = ["label", "text"]
    for key in model_keys:
        fieldnames.extend([f"{key}_prediction", f"{key}_correct"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (text, label) in enumerate(
            zip(splits.test["text"], splits.test["label"])
        ):
            preds = {
                key: predictions_by_model[key][index] for key in model_keys
            }
            disagree = len(set(preds.values())) > 1
            any_wrong = any(prediction != label for prediction in preds.values())
            if not (disagree or any_wrong):
                continue
            row: dict[str, Any] = {"label": label, "text": text}
            for key, prediction in preds.items():
                row[f"{key}_prediction"] = prediction
                row[f"{key}_correct"] = prediction == label
            writer.writerow(row)


def _print_summary(results: dict[str, Any]) -> None:
    print("\nTest-set comparison")
    print("-" * 72)
    print(
        f"{'Model':40} {'Accuracy':>9} {'F1':>9} "
        f"{'Train (s)':>10} {'ms/item':>9}"
    )
    for key in ("tfidf", "bert", "deberta"):
        if key not in results:
            continue
        result = results[key]
        print(
            f"{result['model'][:40]:40} "
            f"{result['test']['accuracy']:9.4f} "
            f"{result['test']['f1_macro']:9.4f} "
            f"{result['training_seconds']:10.2f} "
            f"{result['milliseconds_per_sample']:9.3f}"
        )


def main() -> None:
    args = parse_args()
    selected = _selected_models(args.model)
    output_dir = Path(args.output_dir)
    started_at = datetime.now(timezone.utc)
    tensorboard_root = Path(
        args.tensorboard_dir or output_dir / "tensorboard"
    )
    tensorboard_run_dir = (
        None
        if args.no_tensorboard
        else tensorboard_root / started_at.strftime("%Y%m%d-%H%M%S")
    )
    splits = load_imdb_splits(
        train_size=_optional_size(args.train_size),
        validation_size=_optional_size(args.validation_size),
        test_size=_optional_size(args.test_size),
        seed=args.seed,
    )
    results: dict[str, Any] = {
        "created_at": started_at.isoformat(),
        "dataset": "stanfordnlp/imdb",
        "seed": args.seed,
        "tensorboard_run_dir": (
            str(tensorboard_run_dir) if tensorboard_run_dir else None
        ),
        "split_sizes": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
    }
    predictions_by_model: dict[str, list[int]] = {}

    if "tfidf" in selected:
        print("Running TF-IDF experiment...")
        results["tfidf"], predictions_by_model["tfidf"] = run_tfidf_experiment(
            splits.train,
            splits.validation,
            splits.test,
            max_features=args.max_features,
            tensorboard_log_dir=(
                str(tensorboard_run_dir / "tfidf")
                if tensorboard_run_dir
                else None
            ),
        )
        save_results(results, output_dir / "comparison.json")

    if "bert" in selected:
        print("Running BERT experiment...")
        results["bert"], predictions_by_model["bert"] = run_bert_experiment(
            splits.train,
            splits.validation,
            splits.test,
            model_name=args.bert_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
            tensorboard_log_dir=(
                str(tensorboard_run_dir / "bert")
                if tensorboard_run_dir
                else None
            ),
            log_every_steps=max(1, args.log_every_steps),
        )
        save_results(results, output_dir / "comparison.json")

    if "deberta" in selected:
        print("Running DeBERTa experiment...")
        results["deberta"], predictions_by_model["deberta"] = (
            run_deberta_experiment(
                splits.train,
                splits.validation,
                splits.test,
                model_name=args.deberta_model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                seed=args.seed,
                tensorboard_log_dir=(
                    str(tensorboard_run_dir / "deberta")
                    if tensorboard_run_dir
                    else None
                ),
                log_every_steps=max(1, args.log_every_steps),
            )
        )
        save_results(results, output_dir / "comparison.json")

    if len(predictions_by_model) >= 2:
        _save_error_analysis(
            splits,
            predictions_by_model,
            output_dir / "error_analysis.csv",
        )

    dashboard_path = create_result_dashboard(
        results,
        output_dir / "comparison_dashboard.png",
        show=not args.no_show_plots,
    )
    _print_summary(results)
    print(f"\nDetailed results saved to {output_dir / 'comparison.json'}")
    print(f"Visual dashboard saved to {dashboard_path}")
    if tensorboard_run_dir:
        print(
            "Open TensorBoard with:\n"
            f'tensorboard --logdir "{tensorboard_root}"'
        )


if __name__ == "__main__":
    main()
