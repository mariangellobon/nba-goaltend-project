# Goaltend close-call — backboard accelerometer analysis

Python tools to load tri-axial backboard accelerometer CSVs, extract **scale-aware** features (direction-change spectrograms and optional time-domain shape features), and train **goaltend vs legal** classifiers. Includes a pipeline for **marginal “close call”** trials held out from segmented training data.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/goaltend_close_call/` | Installable package (`sensor_io`, feature extractors, models) |
| `data/` | Segmented class folders, `Close Calls/`, and `close_calls_labels.csv` |
| `notebooks/` | Exploratory analysis (e.g. spectrograms, PCA) |
| `outputs/` | Generated prediction CSVs (gitignored except `.gitkeep`) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -e ".[notebooks]"   # omit [notebooks] if you only run models
```

If you prefer not to install the package, you can run with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m goaltend_close_call.close_call_model
```

## Run models

From the **repository root** (parent of `src/`):

```bash
python -m goaltend_close_call.close_call_model
python -m goaltend_close_call.close_call_shape_model
python -m goaltend_close_call.sensor_io   # smoke test: loads sample CSVs from data/
```

Predictions are written to `outputs/close_calls_predictions.csv` and `outputs/close_calls_shape_predictions.csv`.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `GOALTEND_DATA_DIR` | `<repo>/data` | Root containing `* - Segmented`, `Close Calls/`, `close_calls_labels.csv` |
| `GOALTEND_OUTPUT_DIR` | `<repo>/outputs` | Where prediction CSVs are written |

## Notebooks

Open `notebooks/goaltend_spectrogram_analysis.ipynb`. The first cell adds `src/` to `sys.path` and sets `DATA_ROOT` to `../data` when the kernel’s working directory is `notebooks/`.

## Sensor conventions

- **Non-goaltend** segmented CSVs: physical sensors **1 and 2** as `(a1, a2)`.
- **Goaltends - Segmented** files: physical sensors **1 and 3** as `(a1, a2)` (see `sensor_io.load_recording_csv`).

## License

MIT — see [LICENSE](LICENSE).
