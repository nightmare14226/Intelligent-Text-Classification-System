"""Fine-tune DeBERTa and introduce its model structure.

DeBERTa (Decoding-enhanced BERT with Disentangled Attention) improves on BERT
with three ideas:

1. Disentangled attention — content and relative-position signals are kept in
   separate vectors and combined only inside the attention score.
2. Enhanced mask decoder — absolute positions are injected only at the final
   decoding / MLM stage rather than being added to every token embedding.
3. (DeBERTa-v3) replaced MLM with replaced-token detection (RTD) for stronger
   pretraining.

This script reuses the same training loop as BERT so metrics stay comparable.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Any

from datasets import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification

from bert_experiment import run_bert_experiment

DEFAULT_MODEL = "microsoft/deberta-v3-base"


def introduce_deberta_structure(model_name: str = DEFAULT_MODEL) -> None:
    """Print architecture, config highlights, and parameter budget."""
    print("=" * 72)
    print(f"DeBERTa structure: {model_name}")
    print("=" * 72)
    print(
        """
Key differences vs BERT / DistilBERT
------------------------------------
- Content vectors  (H) and relative position vectors (P) are disentangled.
- Attention score uses content-to-content, content-to-position, and
  position-to-content terms instead of a single absolute-position sum.
- Absolute position embeddings are applied late (enhanced mask decoder),
  not fused into the input embeddings like BERT.
- DeBERTa-v3 is pretrained with ELECTRA-style replaced-token detection.
"""
    )

    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    print("=== CONFIG HIGHLIGHTS ===")
    for key in (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
        "vocab_size",
        "relative_attention",
        "pos_att_type",
        "position_biased_input",
    ):
        if hasattr(config, key):
            print(f"{key:28}: {getattr(config, key)}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    print("\n=== ARCHITECTURE ===")
    print(model)

    groups: OrderedDict[str, int] = OrderedDict()
    for name, parameter in model.named_parameters():
        parts = name.split(".")
        if name.startswith("deberta.embeddings"):
            key = "deberta.embeddings"
        elif "encoder.layer" in name:
            # deberta.encoder.layer.N....
            layer_index = parts[parts.index("layer") + 1]
            key = f"deberta.encoder.layer[{layer_index}]"
        elif name.startswith("deberta.encoder"):
            key = "deberta.encoder.other"
        elif name.startswith("deberta"):
            key = "deberta.other"
        else:
            key = parts[0]
        groups[key] = groups.get(key, 0) + parameter.numel()

    total = sum(parameter.numel() for parameter in model.parameters())
    print("\n=== PARAMETER BUDGET ===")
    for key, value in groups.items():
        print(f"{key:34} {value:>12,}  {100 * value / total:5.1f}%")
    print("-" * 56)
    print(f"{'TOTAL':34} {total:>12,}  100.0%")
    print(
        "\nNote: the classification head (classifier / pooler) is randomly "
        "initialized and trained during fine-tuning."
    )


def run_deberta_experiment(
    train_data: Dataset,
    validation_data: Dataset,
    test_data: Dataset,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    seed: int = 42,
    tensorboard_log_dir: str | None = None,
    log_every_steps: int = 10,
) -> tuple[dict[str, Any], list[int]]:
    """Fine-tune DeBERTa with the same protocol used for BERT."""
    return run_bert_experiment(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        seed=seed,
        tensorboard_log_dir=tensorboard_log_dir,
        log_every_steps=log_every_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect DeBERTa structure and/or fine-tune it on IMDB."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Only print the model structure; do not fine-tune.",
    )
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensorboard-dir",
        default="results/tensorboard/deberta",
    )
    args = parser.parse_args()

    introduce_deberta_structure(args.model)
    if args.structure_only:
        return

    from dataloader import load_imdb_splits
    from evaluation import save_results

    def optional_size(value: int) -> int | None:
        return None if value == 0 else value

    splits = load_imdb_splits(
        train_size=optional_size(args.train_size),
        validation_size=optional_size(args.validation_size),
        test_size=optional_size(args.test_size),
        seed=args.seed,
    )
    print("\nFine-tuning DeBERTa on IMDB...")
    results, _ = run_deberta_experiment(
        splits.train,
        splits.validation,
        splits.test,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        tensorboard_log_dir=args.tensorboard_dir,
    )
    save_results({"deberta": results}, "results/deberta_only.json")
    print(
        f"Test accuracy={results['test']['accuracy']:.4f}  "
        f"F1={results['test']['f1_macro']:.4f}"
    )
    print("Saved results/deberta_only.json")


if __name__ == "__main__":
    main()
