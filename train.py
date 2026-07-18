"""Command-line runner for the TF-IDF versus BERT comparison."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bert_experiment import run_bert_experiment
from dataloader import DatasetSplits, load_imdb_splits
from evaluation import save_results
from tfidf_experiment import run_tfidf_experiment


def _optional_size(value: int) -> int | None:
    return None if value == 0 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF and BERT on identical IMDB splits."
    )
    parser.add_argument(
        "--model", choices=["both", "tfidf", "bert"], default="both"
    )
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bert-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def _save_error_analysis(
    splits: DatasetSplits,
    tfidf_predictions: list[int],
    bert_predictions: list[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "label",
                "tfidf_prediction",
                "bert_prediction",
                "tfidf_correct",
                "bert_correct",
                "text",
            ],
        )
        writer.writeheader()
        for text, label, tfidf, bert in zip(
            splits.test["text"],
            splits.test["label"],
            tfidf_predictions,
            bert_predictions,
        ):
            if tfidf != bert or tfidf != label:
                writer.writerow(
                    {
                        "label": label,
                        "tfidf_prediction": tfidf,
                        "bert_prediction": bert,
                        "tfidf_correct": tfidf == label,
                        "bert_correct": bert == label,
                        "text": text,
                    }
                )


def _print_summary(results: dict[str, Any]) -> None:
    print("\nTest-set comparison")
    print("-" * 72)
    print(
        f"{'Model':32} {'Accuracy':>9} {'F1':>9} "
        f"{'Train (s)':>10} {'ms/item':>9}"
    )
    for key in ("tfidf", "bert"):
        if key not in results:
            continue
        result = results[key]
        print(
            f"{result['model'][:32]:32} "
            f"{result['test']['accuracy']:9.4f} "
            f"{result['test']['f1_macro']:9.4f} "
            f"{result['training_seconds']:10.2f} "
            f"{result['milliseconds_per_sample']:9.3f}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    splits = load_imdb_splits(
        train_size=_optional_size(args.train_size),
        validation_size=_optional_size(args.validation_size),
        test_size=_optional_size(args.test_size),
        seed=args.seed,
    )
    results: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "stanfordnlp/imdb",
        "seed": args.seed,
        "split_sizes": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
    }
    tfidf_predictions: list[int] | None = None
    bert_predictions: list[int] | None = None

    if args.model in ("both", "tfidf"):
        print("Running TF-IDF experiment...")
        results["tfidf"], tfidf_predictions = run_tfidf_experiment(
            splits.train,
            splits.validation,
            splits.test,
            max_features=args.max_features,
        )
        save_results(results, output_dir / "comparison.json")

    if args.model in ("both", "bert"):
        print("Running BERT experiment...")
        results["bert"], bert_predictions = run_bert_experiment(
            splits.train,
            splits.validation,
            splits.test,
            model_name=args.bert_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
        )
        save_results(results, output_dir / "comparison.json")

    if tfidf_predictions is not None and bert_predictions is not None:
        _save_error_analysis(
            splits,
            tfidf_predictions,
            bert_predictions,
            output_dir / "error_analysis.csv",
        )

    _print_summary(results)
    print(f"\nDetailed results saved to {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
