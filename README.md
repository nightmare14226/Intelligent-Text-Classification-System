# Text Classification: TF-IDF vs BERT vs DeBERTa

This project compares a TF-IDF logistic-regression baseline with fine-tuned
Transformers (DistilBERT / BERT and DeBERTa) on the same deterministic IMDB
train, validation, and test splits.

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

After training, a visual comparison dashboard opens automatically and is saved
to `results/comparison_dashboard.png`. It includes test accuracy and macro F1,
training time, inference latency, confusion matrices, and BERT learning curves.
Use `--no-show-plots` to save the image without opening a window.

TensorBoard logging is enabled by default. While the experiment is running,
open another terminal in the project directory, activate the virtual
environment, and start TensorBoard:

```powershell
.\venv\Scripts\Activate.ps1
tensorboard --logdir results/tensorboard
```

Then open `http://localhost:6006`. BERT reports batch loss and learning rate
during training, plus epoch-level validation metrics. Both models report final
validation, test, and performance metrics. Use `--log-every-steps 5` to change
the BERT batch logging interval, `--tensorboard-dir PATH` for another log root,
or `--no-tensorboard` to disable logging.

Use all available data by setting each size to zero:

```powershell
python main.py --model both --train-size 0 --validation-size 0 --test-size 0
```

Compare BERT and DeBERTa on identical splits:

```powershell
python main.py --model bert-deberta --no-show-plots
```

Inspect DeBERTa's architecture before fine-tuning:

```powershell
python deberta_experiment.py --structure-only
```

Fine-tune DeBERTa alone:

```powershell
python deberta_experiment.py
# or
python main.py --model deberta --no-show-plots
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

Default DeBERTa checkpoint is `microsoft/deberta-v3-base`. Override with
`--deberta-model`.

## Outputs

- `results/comparison.json`: accuracy, macro precision/recall/F1, confusion
  matrices, runtime, inference latency, and model size.
- `results/error_analysis.csv`: examples on which the models disagree or make
  an error.
- `results/comparison_dashboard.png`: visual model-quality, efficiency, error,
  and learning-curve comparison.
- `results/tensorboard/`: timestamped TensorBoard event logs for each model.

For a fair report, repeat runs across multiple seeds and compare data sizes
(for example 500, 2,000, 5,000, and the full training set). GPU and CPU latency
should be reported separately.
