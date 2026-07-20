"""Plot BERT vs DeBERTa attention heatmaps for one validation sample."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataloader import load_imdb_splits
from deberta_experiment import DEFAULT_MODEL as DEFAULT_DEBERTA_MODEL
from visualization import create_attention_heatmap

LABEL_NAMES = ("Negative", "Positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize attention heatmaps for BERT and DeBERTa on one "
            "IMDB validation sample."
        )
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bert-model", default="distilbert-base-uncased")
    parser.add_argument("--deberta-model", default=DEFAULT_DEBERTA_MODEL)
    parser.add_argument(
        "--max-length",
        type=int,
        default=48,
        help="Token budget for a readable heatmap (not for training).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Transformer layer index for the heatmap (-1 = last layer).",
    )
    parser.add_argument(
        "--output",
        default="results/attention_heatmap.png",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening a window.",
    )
    return parser.parse_args()


def _resolve_layer(layer: int, num_layers: int) -> int:
    if layer < 0:
        layer = num_layers + layer
    if not 0 <= layer < num_layers:
        raise ValueError(
            f"Layer {layer} is out of range for {num_layers} layers."
        )
    return layer


def _mean_attention(
    attentions: tuple[torch.Tensor, ...],
    layer: int,
    sequence_length: int,
) -> np.ndarray:
    # Shape: (batch, heads, seq, seq) -> mean over heads for batch 0.
    matrix = attentions[layer][0].mean(dim=0).detach().cpu().numpy()
    return matrix[:sequence_length, :sequence_length]


def _extract_attention_panel(
    model_name: str,
    model_label: str,
    text: str,
    max_length: int,
    layer: int,
    device: torch.device,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
    sequence_length = len(tokens)

    with torch.inference_mode():
        output = model(**encoded, output_attentions=True)

    if output.attentions is None:
        raise RuntimeError(
            f"{model_name} did not return attentions. "
            "Try a different checkpoint."
        )

    resolved_layer = _resolve_layer(layer, len(output.attentions))
    attention = _mean_attention(
        output.attentions, resolved_layer, sequence_length
    )
    prediction = LABEL_NAMES[int(output.logits.argmax(dim=-1).item())]

    return {
        "model_label": model_label,
        "tokens": tokens,
        "attention": attention,
        "prediction": prediction,
        "layer": resolved_layer,
        "checkpoint": model_name,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validation_size = None if args.validation_size == 0 else args.validation_size
    splits = load_imdb_splits(
        train_size=100,
        validation_size=validation_size,
        test_size=100,
        seed=args.seed,
    )
    if not 0 <= args.sample_index < len(splits.validation):
        raise IndexError(
            f"sample-index {args.sample_index} is outside validation "
            f"size {len(splits.validation)}."
        )

    sample = splits.validation[args.sample_index]
    text = sample["text"]
    true_label = LABEL_NAMES[int(sample["label"])]

    print(f"Device: {device}")
    print(f"Validation sample index: {args.sample_index}")
    print(f"True label: {true_label}")
    print(f"Text preview: {text[:200]}...")

    panels = [
        _extract_attention_panel(
            args.bert_model,
            "BERT",
            text,
            args.max_length,
            args.layer,
            device,
        ),
        _extract_attention_panel(
            args.deberta_model,
            "DeBERTa",
            text,
            args.max_length,
            args.layer,
            device,
        ),
    ]

    for panel in panels:
        print(
            f"{panel['model_label']} ({panel['checkpoint']}): "
            f"layer={panel['layer']}, pred={panel['prediction']}, "
            f"tokens={len(panel['tokens'])}"
        )

    output_path = create_attention_heatmap(
        panels,
        Path(args.output),
        sample_text=text,
        true_label=true_label,
        show=not args.no_show,
    )
    print(f"Saved attention heatmap to {output_path}")


if __name__ == "__main__":
    main()
