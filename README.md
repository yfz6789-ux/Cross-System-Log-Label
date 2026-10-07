# Cross-System Log Labelling

An HTTP access-log labelling pipeline that classifies requests as `normal` or `anomaly` using labelled source examples, target request context, and local inference through Ollama.

| Direction | Source → target |
|---|---|
| `b_to_a` | Biblio-US17 → AIT Santos |
| `srbh_to_b` | SR-BH 2020 → Biblio-US17 |
| `srbh_to_a` | SR-BH 2020 → AIT Santos |

## Setup

Use Python 3.11 or later. Run commands from the project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull deepseek-r1:32b
OLLAMA_HOST=127.0.0.1:11500 OLLAMA_NUM_PARALLEL=6 ollama serve
```

## Run

With the Ollama server running, use another terminal:

```bash
PYTHONPATH=src .venv/bin/python src/benchmark.py --direction b_to_a --output output
PARALLEL=1 bash ./run_all.sh
```

The batch runner runs these three directions by default. Provide input files in `data/improved/`, including `SR-BH 2020.csv`; datasets are not included in this repository. SR-BH sources use 50,000 sampled rows with seed 42 by default. Use `--help` for available options.

## Data and results

[Data preparation](data/README.md) describes input formats. [Method](docs/METHOD.md) describes request context, sampling, and evaluation.

Each run writes results to `output/<direction>/`, including configuration, cached responses, row and text predictions, and a summary. Experiment outputs are not included in this repository. Identical request texts share a prediction. Confidence is uncalibrated.

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Tests use synthetic data and require no inference server.
