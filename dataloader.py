"""Shared, deterministic data loading for all model experiments."""

from dataclasses import dataclass

from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class DatasetSplits:
    train: Dataset
    validation: Dataset
    test: Dataset


def _sample(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    if size is None or size >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(size))


def load_imdb_splits(
    train_size: int | None = None,
    validation_size: int | None = None,
    test_size: int | None = None,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetSplits:
    """Load IMDB and create one split shared by TF-IDF and BERT.

    Sizes limit each split for quick experiments. Use ``None`` for all
    available examples.
    """
    dataset = load_dataset("imdb")
    train_validation = dataset["train"].train_test_split(
        test_size=validation_fraction,
        seed=seed,
        stratify_by_column="label",
    )

    return DatasetSplits(
        train=_sample(train_validation["train"], train_size, seed),
        validation=_sample(
            train_validation["test"], validation_size, seed + 1
        ),
        test=_sample(dataset["test"], test_size, seed + 2),
    )

