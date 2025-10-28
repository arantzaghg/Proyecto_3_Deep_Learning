# Proyecto 3 — Deep Learning for Trading

A reproducible pipeline for training deep learning models (MLP/CNN), generating trading signals, and backtesting a portfolio. The repo also includes utilities for feature engineering, data normalization, MLflow experiment tracking, plotting, and data-drift checks.

> Tested on macOS (Apple Silicon), Linux, and Windows with Python 3.10+.

---

## Features

* End‑to‑end workflow: data prep → modeling → signals → backtest → metrics → plots
* MLP and CNN reference architectures
* Technical indicators and normalization utilities
* Portfolio simulation and performance metrics (Sharpe, Sortino, drawdowns, etc.)
* MLflow-ready experiment folders in mlruns/
* Data drift utilities and a simple drift report
* Companion dashboard notebook for ad‑hoc analysis

---

## Repository structure


Proyecto_3_Deep_Learning/

├─ 003 Advanced Trading Strategies Deep Learning.pdf   # project report

├─ mlruns/                                             # MLflow runs (auto‑created)

├─ dashboard.ipynb                                     # quick exploratory dashboard

├─ main.py                                             # end‑to‑end run entrypoint

├─ run_models.py                                       # train/evaluate models from CLI

├─ backtesting.py                                      # backtesting utilities

├─ portfolio_value.py                                  # portfolio value computation

├─ get_signals.py                                      # signal generation helpers

├─ indicators.py                                       # technical indicators

├─ normalization.py                                    # scaling/normalization helpers

├─ data_utils.py                                       # data I/O & dataset helpers

├─ metrics.py                                          # performance metrics

├─ plots.py                                            # visualization utilities

├─ models.py                                           # model registry / wrappers

├─ MLP_model.py                                        # MLP reference model

├─ CNN_model.py                                        # CNN reference model

├─ data_drift.py                                       # drift detection helpers

├─ run_drift.py                                        # CLI to run a drift report

├─ drift_simple_report/                                # example drift outputs

├─ requirements.txt                                    # pinned Python deps

└─ README.md


---

## Requirements

* *Python* 3.10 or newer
* *pip* ≥ 22 or *uv/poetry/conda* (optional)
* On Apple Silicon (M1/M2/M3), a recent *pip* and *virtualenv* are recommended.

> If you plan to use GPU acceleration, install a build of PyTorch/TensorFlow that matches your hardware; otherwise, the CPU versions in requirements.txt are fine.

---

## Setup (recommended: virtual environment)

bash
## 1) Clone
git clone https://github.com/arantzaghg/Proyecto_3_Deep_Learning.git
cd Proyecto_3_Deep_Learning

## 2) Create & activate a virtual env (choose one)
python3 -m venv .venv && source .venv/bin/activate      # macOS/Linux (bash/zsh)
## OR (conda)
## conda create -n dl-trading python=3.10 -y && conda activate dl-trading

## 3) Install project dependencies
pip install --upgrade pip
pip install -r requirements.txt


---

## Data

Place your raw price data (e.g., CSV/Parquet) in a folder of your choice (e.g., data/). By default, the helper modules expect a tabular time series with a timestamp index or a datetime column plus OHLCV columns; adjust paths/columns in data_utils.py and get_signals.py as needed.

> Tip: keep a consistent timezone and frequency (e.g., daily or minute bars) across all steps.

---

## Quickstart

### 1) Train & evaluate models

The easiest entrypoint is run_models.py, which trains/evaluates the registered models and logs outputs (metrics, plots, and artifacts). Example:

bash
python run_models.py \
  --data-path ./data/prices.csv \
  --target-column target \
  --models MLP CNN \
  --test-split 0.2 \
  --seed 42


*Common flags* (check run_models.py for the authoritative list):

* --data-path: Input dataset path
* --target-column: Target to predict (e.g., future return label)
* --models: One or multiple model names (e.g., MLP, CNN)
* --epochs, --batch-size, --lr: Training hyperparameters
* --test-split, --val-split, --seed: Experiment splits & reproducibility
* --mlflow: Enable MLflow logging if available (defaults handled internally)

### 2) End‑to‑end run

main.py typically wires the full pipeline (features → train → signals → backtest → report). For example:

bash
python main.py \
  --data-path ./data/prices.csv \
  --config ./config.yaml


(If --config is not present, use the available CLI flags in main.py.)

### 3) Generate trading signals

bash
python -m get_signals \
  --model-checkpoint ./artifacts/best_mlp.pt \
  --data-path ./data/prices.csv \
  --out ./artifacts/signals.csv


### 4) Backtest

bash
python -m backtesting \
  --signals ./artifacts/signals.csv \
  --initial-cash 100000 \
  --transaction-cost 0.0005 \
  --out ./artifacts/backtest.json


### 5) Plot results

bash
python -m plots \
  --backtest ./artifacts/backtest.json \
  --save ./artifacts/plots/


---

## Data drift report

Use run_drift.py to compute a quick drift summary between a *reference* (train) period and a *current* (production) period. Outputs are written to drift_simple_report/ by default.

bash
python run_drift.py \
  --reference ./data/train.csv \
  --current   ./data/live.csv \
  --out       ./drift_simple_report


If you need custom behavior, check data_drift.py and adjust thresholds/features.

---

## MLflow tracking (optional)

If MLflow is installed, runs will be saved under mlruns/. To launch the UI locally:

bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000


Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Reproducibility

* Set --seed (or the corresponding config option) to fix splits and initializations.
* For exact determinism across hardware/backends, also fix NumPy/PyTorch seeds in your config; note some GPU ops remain nondeterministic.

---

## Development

bash
Run linters/formatters if you add them later
ruff check . && ruff format .
black .

Run unit tests (if/when added)
pytest -q


---

## Troubleshooting

* *Apple Silicon (M‑series):* If you need accelerated PyTorch builds, install via the official instructions for macOS/Metal. Otherwise the CPU wheel from requirements.txt works.
* *Matplotlib backend errors on macOS:* try pip install pyqt5 or set MPLBACKEND=Agg when running headless.
* *Pandas date parsing:* ensure your CSV has a parseable datetime column (ISO‑8601) or set dayfirst/format explicitly in data_utils.py.

---

## License

This project is licensed under the *MIT License*. See LICENSE for details.

---

## Acknowledgments

* Course project materials and references included in the repository PDF.

---

## Citation

If you use this project in academic work, please consider citing the repository.

text
@software{proyecto3_deeplearning,
  author  = {arantzaghg and collaborators},
  title   = {Proyecto 3 — Deep Learning for Trading},
  year    = {2025},
  url     = {https://github.com/arantzaghg/Proyecto_3_Deep_Learning}
}
