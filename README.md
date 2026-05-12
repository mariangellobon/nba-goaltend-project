# NBA goaltend — backboard IMU ML

Portfolio repository for supervised learning on **backboard-mounted accelerometers** applied to NBA-style goaltend detection. This codebase is intentionally **lightweight on Git**: raw LabChart exports stay on disk locally; commits contain **training and evaluation code** only.

Two modeling threads share the same data layout (`GOALTEND_DATA_DIR`) but differ in framing and methodology.

## Approaches at a glance

| Track | Location | Idea |
|--------|----------|------|
| **Close-call spectrogram / shape models** | `src/goaltend_close_call/` | Peak-windowed traces; direction-change spectrograms and/or handcrafted time-domain shape features; **stratified K-fold** over labeled ambiguous “close calls” with optional segmented data augmenting each fold; **sklearn** classifiers (logistic regression, gradient boosting, or random forest). |
| **Segmented traces — ROCKET + TabPFN** | `src/goaltend_tabpfn/` | Full six-channel variable-length snippets from class folders → **aeon ROCKET** embeddings → **[TabPFN](https://priorlabs.ai/tabpfn)** (requires Hugging Face license acceptance + token). Supports **leave-one-out**, stratified holdout, and optional **`test/`** inference. |

Outputs (OOF CSVs, TabPFN run logs, dumped `fitted_tabpfn_pipeline.joblib`) go under **`outputs/`**, ignored by git.

## Repository layout

| Path | Role |
|------|------|
| `src/goaltend_close_call/` | Packaging, IO, spectrogram/shape pipelines, sklearn classifiers |
| `src/goaltend_tabpfn/` | `goaltend_classify.py`, `goaltend_holdout.py`, `tabpfn_analysis.py` (ROCKET + TabPFN + error/confidence plots) |
| `notebooks/` | Exploratory work (spectrograms, sensor plots) |
| `syncing_video_data/` | Small utilities to overlay IMU traces on video and scrub frames |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel

# Close-call pipelines + notebooks optional extras
pip install -r requirements.txt

# TabPFN track (heavy: PyTorch, aeon — install when you run that branch)
pip install -e ".[tabpfn]"
```

### Local data folder (required to run experiments)

Git does **not** contain CSVs or label files. After cloning, create **`data/`** at the repo root (or point `GOALTEND_DATA_DIR` anywhere) with:

- **Segmented folders** (names preserved by the loaders):  
  `Ball on Rim - Segmented/`, `Blocks - Segmented/`, `Hand on Backboard - Segmented/`, `Goaltends - Segmented/` (optional: `Other Data - Segmented/`)
- **Close-call track:** `Close Calls/` plus `close_calls_labels.csv` (columns described in loader docstrings — see `src/goaltend_close_call/close_call_labels.py`)

Override location:

```bash
export GOALTEND_DATA_DIR=/absolute/path/to/your/data
export GOALTEND_OUTPUT_DIR=/absolute/path/to/write/artifacts   # default: <repo>/outputs
```

## Run — close-call models

From repo root:

```bash
python -m goaltend_close_call.close_call_model
python -m goaltend_close_call.close_call_shape_model
python -m goaltend_close_call.sensor_io   # smoke test on discovered CSVs
```

See **Environment variables** in the sections below or in module docstrings for CV splits, logistic vs boosted trees, window length, and STFT size.

## Run — ROCKET + TabPFN (`goaltend_tabpfn`)

TabPFN model weights are **gated on Hugging Face**:

1. Accept the licence at [Prior-Labs/TabPFN-v2-clf](https://huggingface.co/Prior-Labs/TabPFN-v2-clf)  
2. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  
3. `export HF_TOKEN=...` (or let scripts prompt interactively)

```bash
# Leave-one-out CV + optional run report → outputs/tabpfn_runs/<timestamp>/ ...
python -m goaltend_tabpfn.goaltend_classify

# Holdout shorthand (built into classify CLI)
python -m goaltend_tabpfn.goaltend_classify --holdout

# Dedicated holdout + save pipeline to outputs/fitted_tabpfn_pipeline.joblib
python -m goaltend_tabpfn.goaltend_holdout

# After training holdout module: unseen CSVs in data/test/
python -m goaltend_tabpfn.goaltend_holdout --test
```

**Inspect wrong or confidence-bucketed calls** (re-runs LOO or holdout, writes `filtered.csv`, `summary.txt`, `figures/*.png` under `outputs/tabpfn_analysis/`):

```bash
# Every misclassification (LOO)
python -m goaltend_tabpfn.tabpfn_analysis --split loo --view wrong

# Wrong but model was >= 0.75 sure of its predicted class
python -m goaltend_tabpfn.tabpfn_analysis --split loo --view confident_wrong --threshold 0.75

# Ambiguous predictions (max class prob < 0.75); add --wrong-only for errors only
python -m goaltend_tabpfn.tabpfn_analysis --split loo --view low_confidence --threshold 0.75
python -m goaltend_tabpfn.tabpfn_analysis --split loo --view low_confidence --threshold 0.75 --wrong-only

# Same views on the stratified holdout *test* split
python -m goaltend_tabpfn.tabpfn_analysis --split holdout --view confident_wrong --random-state 42
```

Use **CPU**, **CUDA**, or **Apple MPS** as supported by your installed Torch (scripts default toward MPS where available).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `GOALTEND_DATA_DIR` | `<repo>/data` | Dataset root (`* - Segmented/`, close calls + labels when using that pipeline) |
| `GOALTEND_OUTPUT_DIR` | `<repo>/outputs` | Writable artifacts (predictions, TabPFN reports, dumped TabPFN pipelines) |
| `GOALTEND_CC_CV_SPLITS` | `5` | Stratified folds over usable close-call clips |
| `GOALTEND_TRAIN_CLOSE_ONLY` | `1` | `1` train each fold from labeled close calls only; `0` include segmented augmentation |
| `GOALTEND_MODEL` | `logistic` | `logistic`, `hgb`, or `rf` (close-call spectrogram/stack) |
| `GOALTEND_LOGISTIC_C`, `GOALTEND_WIN_SEC`, `GOALTEND_NPERSEG`, … | see `close_call_model.py` | Tuning knobs for the spectrogram path |
| `HF_TOKEN` | — | Hugging Face read token for TabPFN weights |

## Notebooks

Start from `notebooks/goaltend_spectrogram_analysis.ipynb`. The opening cells expect the kernel cwd under `notebooks/` and typical `PYTHONPATH` / `GOALTEND_DATA_DIR` conventions documented in-repo.

Keep **Executed** notebooks with large outputs out of commits (patterns in `.gitignore`).

## Sensor conventions

- Non-goaltend segmented CSVs: physical sensors **1 and 2** as `(a1, a2)`.
- `Goaltends - Segmented/`: sensors **1 and 3** aliased into the same pairing (see `sensor_io.load_recording_csv`).

## Licence

MIT — see [LICENSE](LICENSE).
