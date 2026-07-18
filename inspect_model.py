"""Log a model's computation graph so TensorBoard can display it."""

import argparse
import warnings

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def log_model_graph(
    model_name: str = "distilbert-base-uncased",
    log_dir: str = "results/tensorboard/model-graph",
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # torchscript=True makes the forward pass return plain tuples, which
    # the graph tracer requires instead of Hugging Face output objects.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, torchscript=True
    )
    model.eval()

    example = tokenizer(
        "This movie was absolutely wonderful!",
        return_tensors="pt",
        padding="max_length",
        max_length=32,
        truncation=True,
    )

    writer = SummaryWriter(log_dir=log_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        writer.add_graph(
            model, [example["input_ids"], example["attention_mask"]]
        )
    writer.close()
    print(f"Model graph for {model_name} written to {log_dir}")
    print('View it with: tensorboard --logdir "results/tensorboard"')
    print("Then open http://localhost:6006 and select the GRAPHS tab.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write a model graph for TensorBoard's Graphs tab."
    )
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument(
        "--log-dir", default="results/tensorboard/model-graph"
    )
    arguments = parser.parse_args()
    log_model_graph(arguments.model, arguments.log_dir)
