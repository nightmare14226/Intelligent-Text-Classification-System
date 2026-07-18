"""Fine-tune a Transformer classifier and report comparable metrics."""

import copy
import random
import time
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from evaluation import classification_metrics


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokenize(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    tokenized = dataset.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=max_length
        ),
        batched=True,
        remove_columns=["text"],
    )
    return tokenized.rename_column("label", "labels")


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], list[int], float]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    started = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            labels.extend(batch["labels"].tolist())
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())

    elapsed = time.perf_counter() - started
    return classification_metrics(labels, predictions), predictions, elapsed


def run_bert_experiment(
    train_data: Dataset,
    validation_data: Dataset,
    test_data: Dataset,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    seed: int = 42,
) -> tuple[dict[str, Any], list[int]]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        _tokenize(train_data, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        _tokenize(validation_data, tokenizer, max_length),
        batch_size=batch_size,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        _tokenize(test_data, tokenizer, max_length),
        batch_size=batch_size,
        collate_fn=collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    training_started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += output.loss.item()
            progress.set_postfix(loss=f"{output.loss.item():.4f}")

        validation_metrics, _, _ = _evaluate(
            model, validation_loader, device
        )
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": total_loss / len(train_loader),
                "validation": validation_metrics,
            }
        )
        if validation_metrics["f1_macro"] > best_f1:
            best_f1 = validation_metrics["f1_macro"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    training_seconds = time.perf_counter() - training_started
    if best_state is not None:
        model.load_state_dict(best_state)

    validation_metrics, _, _ = _evaluate(model, validation_loader, device)
    test_metrics, test_predictions, inference_seconds = _evaluate(
        model, test_loader, device
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    size_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    results = {
        "model": model_name,
        "device": str(device),
        "epochs": epochs,
        "parameter_count": parameter_count,
        "model_size_mb": size_bytes / (1024**2),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "milliseconds_per_sample": 1_000 * inference_seconds / len(test_data),
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
    }
    return results, test_predictions
