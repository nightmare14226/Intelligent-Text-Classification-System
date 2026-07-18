# BERT vs TF-IDF Text Classification

This project compares a TF-IDF logistic-regression baseline with a fine-tuned
Transformer on the same deterministic IMDB train, validation, and test splits.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python check_environment.py
```

## Run experiments

Start with TF-IDF to verify the pipeline:

```powershell
python main.py --model tfidf
```

Run the default comparison (5,000 train, 1,000 validation, 2,000 test):

```powershell
python main.py --model both
```

Use all available data by setting each size to zero:

```powershell
python main.py --model both --train-size 0 --validation-size 0 --test-size 0
```

For a quick BERT smoke test:

```powershell
python main.py --model bert --train-size 200 --validation-size 100 --test-size 100 --epochs 1
```

If GPU memory is limited, reduce `--batch-size` or `--max-length`. To use full
BERT instead of the faster default DistilBERT:

```powershell
python main.py --bert-model bert-base-uncased --batch-size 4
```

## Outputs

- `results/comparison.json`: accuracy, macro precision/recall/F1, confusion
  matrices, runtime, inference latency, and model size.
- `results/error_analysis.csv`: examples on which the models disagree or make
  an error.

For a fair report, repeat runs across multiple seeds and compare data sizes
(for example 500, 2,000, 5,000, and the full training set). GPU and CPU latency
should be reported separately.
