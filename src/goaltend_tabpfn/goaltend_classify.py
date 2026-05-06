"""
goaltend_classify.py
--------------------
ROCKET + TabPFN time-series classification for goaltend detection.

Dataset layout (expected under ``GOALTEND_DATA_DIR``, default ``<repo>/data``):
    Ball on Rim - Segmented/        → label "legal"
    Blocks - Segmented/             → label "legal"
    Hand on Backboard - Segmented/  → label "legal"
    Goaltends - Segmented/          → label "goaltend"

Each CSV file is one sample with 6 accelerometer channels and a variable
number of timesteps. Shorter samples are zero-padded at the end so that
the final dataset tensor is shape (N, 6, T_max).

Authentication
--------------
TabPFN weights are gated on HuggingFace. Before running:
  1. Accept the license at https://huggingface.co/Prior-Labs/TabPFN-v2-clf
  2. Create a read token at https://huggingface.co/settings/tokens
  3. Either set the environment variable HF_TOKEN=<your_token>
     or let the script prompt you interactively.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

from goaltend_close_call.paths import data_root, outputs_dir

# Labeled segmented folders live here (override with GOALTEND_DATA_DIR).
DATA_DIR = data_root()


# ── HuggingFace authentication (runs once at import time) ─────────────────────

def _hf_login() -> None:
    """Log in to HuggingFace so TabPFN can download its gated weights.

    Reads HF_TOKEN from the environment if set, otherwise falls back to an
    interactive prompt via huggingface_hub.login().
    """
    try:
        from huggingface_hub import login, whoami
        token = os.environ.get("HF_TOKEN")
        try:
            whoami()   # already logged in (cached token on disk)
        except Exception:
            if token:
                login(token=token, add_to_git_credential=False)
            else:
                print(
                    "HuggingFace login required to download TabPFN weights.\n"
                    "Accept the license at: https://huggingface.co/Prior-Labs/TabPFN-v2-clf\n"
                    "Then create a token at: https://huggingface.co/settings/tokens\n"
                )
                login(add_to_git_credential=False)
    except ImportError:
        raise ImportError(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )

_hf_login()

# ── configuration ─────────────────────────────────────────────────────────────

N_KERNELS = 500
N_GROUPS  = 1    # independent ROCKET+TabPFN ensembles averaged per iteration

# Folder name → binary label
FOLDER_LABELS: dict[str, str] = {
    "Ball on Rim - Segmented":       "legal",
    "Blocks - Segmented":            "legal",
    "Hand on Backboard - Segmented": "legal",
    "Goaltends - Segmented":         "goaltend",
}

# The 6 accelerometer channel column names as they appear in the CSV headers
ACCEL_COLS = [
    "Latest: X Acceleration 1 (m/s²)",
    "Latest: Y Acceleration 1 (m/s²)",
    "Latest: Z Acceleration 1 (m/s²)",
    "Latest: Z Acceleration 2 (m/s²)",
    "Latest: Y Acceleration 2 (m/s²)",
    "Latest: X Acceleration 2 (m/s²)",
]


# ── dataset loader ────────────────────────────────────────────────────────────

def load_goaltend_dataset(data_dir: Path = DATA_DIR) -> tuple[list[np.ndarray], np.ndarray, list[Path]]:
    """Load all CSV samples as a list of variable-length arrays (no pre-padding).

    Padding is deferred to each LOO fold so that T_max is computed only from
    training samples, preventing the test sample's length from leaking into
    the training representation.

    Returns
    -------
    raw_samples : list of np.ndarray, each shape (6, T_i)
        One unpadded array per sample; T_i varies across samples.
    y : np.ndarray of str, shape (N,)
        "legal" or "goaltend" for each sample.
    paths : list[Path]
        CSV file path for each sample (same order as raw_samples and y).
    """
    raw_samples: list[np.ndarray] = []
    labels:      list[str]        = []
    paths:       list[Path]       = []

    for folder_name, label in FOLDER_LABELS.items():
        folder = data_dir / folder_name
        if not folder.exists():
            print(f"  [warn] folder not found, skipping: {folder}")
            continue
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            print(f"  [warn] no CSV files in: {folder}")
            continue
        for csv_path in csv_files:
            df  = pd.read_csv(csv_path, usecols=ACCEL_COLS)
            arr = df[ACCEL_COLS].values.T.astype(np.float32)  # (6, T_i)
            raw_samples.append(arr)
            labels.append(label)
            paths.append(csv_path)

    if not raw_samples:
        raise ValueError(f"No CSV files found under {data_dir!r}. Check DATA_DIR and FOLDER_LABELS.")

    y      = np.array(labels)
    N      = len(raw_samples)
    T_min  = min(s.shape[1] for s in raw_samples)
    T_max  = max(s.shape[1] for s in raw_samples)
    unique, counts = np.unique(y, return_counts=True)

    print(f"Dataset loaded: {N} samples | 6 channels | T_min={T_min}  T_max={T_max}")
    print(f"  Class counts: { {k: v for k, v in zip(unique, counts)} }")
    return raw_samples, y, paths


def pad_to(samples: list[np.ndarray], T: int) -> np.ndarray:
    """Zero-pad or truncate a list of (6, T_i) arrays to a fixed length T and stack them.

    Shorter samples are zero-padded at the end; longer samples are truncated.
    Truncation only happens to the test sample when it is the longest in the
    dataset and is held out — training T_max then drops below its length.

    Returns np.ndarray of shape (N, 6, T).
    """
    N, M = len(samples), samples[0].shape[0]
    X    = np.zeros((N, M, T), dtype=np.float32)
    for i, s in enumerate(samples):
        t_copy = min(s.shape[1], T)
        X[i, :, :t_copy] = s[:, :t_copy]
    return X


def zscore_normalize(X: np.ndarray) -> np.ndarray:
    """Per-sample, per-channel z-score normalization.

    Subtracts the mean and divides by the std of each channel independently
    for each sample. Makes the model invariant to absolute acceleration
    magnitude so it generalizes across different sensors and mounting positions.

    Input/output shape: (N, 6, T)
    """
    out = X.copy()
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            ch  = X[i, c, :]
            out[i, c, :] = (ch - ch.mean()) / (ch.std() + 1e-8)
    return out


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_sample(csv_path: Path, label: str, pred: str | None = None) -> None:
    """Plot all 6 accelerometer channels for one CSV sample.

    Parameters
    ----------
    csv_path : path to the raw CSV file
    label    : true class label ("legal" or "goaltend")
    pred     : predicted label to show in the title (optional)
    """
    import matplotlib.pyplot as plt

    TIME_COL = "Latest: Time (s)"
    df = pd.read_csv(csv_path, usecols=[TIME_COL] + ACCEL_COLS)
    t  = df[TIME_COL].values
    # Shift time so it starts at 0
    t  = t - t[0]

    short_names = ["X1", "Y1", "Z1", "Z2", "Y2", "X2"]
    colors      = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    for ax, col, name, color in zip(axes, ACCEL_COLS, short_names, colors):
        ax.plot(t, df[col].values, color=color, linewidth=0.8)
        ax.set_ylabel(f"{name}\n(m/s²)", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")

    status = ""
    if pred is not None:
        status = f"  |  pred: {pred}  {'✓' if pred == label else '✗'}"
    fig.suptitle(
        f"{csv_path.name}\nTrue label: {label}{status}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_goaltend(
    *,
    data_dir:  Path = DATA_DIR,
    n_groups:  int  = N_GROUPS,
    n_kernels: int  = N_KERNELS,
    device:    str  = "auto",
) -> dict:
    """Run ROCKET + TabPFN leave-one-out cross-validation on the goaltend dataset.

    Each iteration trains on all samples except one and tests on that single
    held-out sample, repeating for every sample in the dataset.

    Parameters
    ----------
    data_dir  : root directory containing the class sub-folders
    n_groups  : number of independent ROCKET+TabPFN ensembles averaged per iteration
    n_kernels : number of ROCKET convolutional kernels per group
    device    : TabPFN compute device ("auto", "cpu", "cuda", "mps")

    Returns
    -------
    dict with accuracy statistics and timing.
    """
    from aeon.transformations.collection.convolution_based import Rocket
    from tabpfn import TabPFNClassifier

    raw_samples, y, paths = load_goaltend_dataset(data_dir)

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Need at least 2 classes to classify, but only found: {unique_classes}. "
            "Add goaltend samples to 'Goaltends - Segmented/'."
        )

    loo   = LeaveOneOut()
    folds = list(loo.split(raw_samples))   # N folds, each test set is exactly 1 sample
    N     = len(folds)

    # "goaltend" is the positive class for ROC AUC
    pos_label  = "goaltend"

    correct:       list[bool]  = []
    all_true:      list[str]   = []   # true label for every held-out sample
    all_pos_proba: list[float] = []   # P(goaltend) for every held-out sample
    fold_times:    list[float] = []
    predictions:   list[str]   = []   # predicted label for every sample

    for i, (tr_idx, te_idx) in enumerate(tqdm(folds, desc="LOO")):
        t_fold = time.perf_counter()

        tr_raw = [raw_samples[j] for j in tr_idx]
        te_raw = [raw_samples[j] for j in te_idx]
        y_tr   = y[tr_idx]
        y_te   = y[te_idx]

        # Use the test sample's natural length as the canonical length.
        # Training samples are zero-padded (if shorter) or truncated (if longer)
        # to match. The test sample is never zero-padded.
        T_te = te_raw[0].shape[1]
        X_tr = zscore_normalize(pad_to(tr_raw, T_te))
        X_te = zscore_normalize(np.array([te_raw[0]], dtype=np.float32))

        all_probas: list[np.ndarray] = []
        classes_   = None

        for g in range(n_groups):
            seed = i * n_groups + g

            rocket = Rocket(n_kernels=n_kernels, random_state=seed, n_jobs=-1)
            rocket.fit(X_tr)
            X_tr_feat = rocket.transform(X_tr)
            X_te_feat = rocket.transform(X_te)
            del rocket

            clf = TabPFNClassifier(
                n_estimators=8,
                device=device,
                random_state=seed,
                ignore_pretraining_limits=True,
            )
            clf.fit(X_tr_feat, y_tr)
            probas = clf.predict_proba(X_te_feat)
            if classes_ is None:
                classes_ = clf.classes_
            all_probas.append(probas)
            del clf, X_tr_feat, X_te_feat, probas

        avg_probas = np.mean(all_probas, axis=0)          # (1, n_classes)
        y_pred     = classes_[np.argmax(avg_probas, axis=1)]
        hit        = bool(y_pred[0] == y_te[0])
        elapsed    = time.perf_counter() - t_fold

        # Probability of the positive class for this held-out sample
        if pos_label in classes_:
            pos_idx = list(classes_).index(pos_label)
            pos_proba = float(avg_probas[0, pos_idx])
        else:
            # positive class not seen in training split; assign 0
            pos_proba = 0.0

        correct.append(hit)
        all_true.append(y_te[0])
        all_pos_proba.append(pos_proba)
        fold_times.append(elapsed)
        predictions.append(y_pred[0])
        tqdm.write(
            f"  sample {i+1:>3}/{N}  true={y_te[0]}  pred={y_pred[0]}"
            f"  p(goaltend)={pos_proba:.3f}  {'OK' if hit else 'WRONG'}  ({elapsed:.0f}s)"
        )

        del all_probas, avg_probas, X_tr, X_te, y_tr, y_te, classes_
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    acc = float(np.mean(correct))

    # ROC AUC requires both classes to appear in the true labels
    unique_true = np.unique(all_true)
    if len(unique_true) >= 2:
        auc = float(roc_auc_score(
            [1 if t == pos_label else 0 for t in all_true],
            all_pos_proba,
        ))
    else:
        auc = float("nan")
        print(f"  [warn] ROC AUC is undefined: only class '{unique_true[0]}' present in labels.")

    result = {
        "n_samples":       N,
        "n_groups":        n_groups,
        "n_kernels":       n_kernels,
        "n_features":      2 * n_kernels,
        "n_correct":       sum(correct),
        "acc":             acc,
        "roc_auc":         auc,
        "time_total_s":    round(sum(fold_times), 1),
        "time_per_iter_s": round(float(np.mean(fold_times)), 1),
    }
    time_per_iter = float(np.mean(fold_times))
    time_total    = sum(fold_times)

    wrong_indices = [i for i, ok in enumerate(correct) if not ok]

    print(f"\nAccuracy      : {acc:.4f}  ({sum(correct)}/{N} correct)")
    print(f"ROC AUC       : {auc:.4f}")
    print(f"Avg time/iter : {time_per_iter:.1f}s")
    print(f"Total time    : {time_total:.1f}s")

    if wrong_indices:
        print(f"\nWrong predictions ({len(wrong_indices)}):")
        for i in wrong_indices:
            print(f"  sample {i+1:>3}  true={all_true[i]}  pred={predictions[i]}  {paths[i].name}")
    else:
        print("\nAll samples predicted correctly.")

    print("Full results:", result)
    return result, paths, predictions


# ── 80/20 holdout validation ─────────────────────────────────────────────────

def evaluate_goaltend_holdout(
    *,
    data_dir:   Path  = DATA_DIR,
    n_groups:   int   = N_GROUPS,
    n_kernels:  int   = N_KERNELS,
    test_size:  float = 0.2,
    random_state: int = 42,
    device:     str   = "auto",
) -> dict:
    """Single stratified train/test split (default 80/20) with ROCKET + TabPFN.

    Parameters
    ----------
    test_size     : fraction of data held out for testing (default 0.2 = 20 %)
    random_state  : seed for the split (change to get a different split)

    Returns
    -------
    dict with accuracy, ROC AUC, and timing.
    """
    from aeon.transformations.collection.convolution_based import Rocket
    from sklearn.model_selection import train_test_split
    from tabpfn import TabPFNClassifier

    raw_samples, y, paths = load_goaltend_dataset(data_dir)

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Need at least 2 classes to classify, but only found: {unique_classes}."
        )

    indices = np.arange(len(raw_samples))
    tr_idx, te_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"\n80/20 holdout split  (seed={random_state})")
    print(f"  Train : {len(tr_idx)} samples  {dict(zip(*np.unique(y[tr_idx], return_counts=True)))}")
    print(f"  Test  : {len(te_idx)} samples  {dict(zip(*np.unique(y[te_idx], return_counts=True)))}\n")

    tr_raw = [raw_samples[i] for i in tr_idx]
    te_raw = [raw_samples[i] for i in te_idx]
    y_tr   = y[tr_idx]
    y_te   = y[te_idx]

    T_tr = max(s.shape[1] for s in tr_raw)
    X_tr = pad_to(tr_raw, T_tr)
    X_te = pad_to(te_raw, T_tr)

    t_start    = time.perf_counter()
    all_probas: list[np.ndarray] = []
    classes_   = None
    pos_label  = "goaltend"

    for g in range(n_groups):
        seed   = g
        rocket = Rocket(n_kernels=n_kernels, random_state=seed, n_jobs=-1)
        rocket.fit(X_tr)
        X_tr_feat = rocket.transform(X_tr)
        X_te_feat = rocket.transform(X_te)
        del rocket

        clf = TabPFNClassifier(
            n_estimators=8,
            device=device,
            random_state=seed,
            ignore_pretraining_limits=True,
        )
        clf.fit(X_tr_feat, y_tr)
        probas = clf.predict_proba(X_te_feat)
        if classes_ is None:
            classes_ = clf.classes_
        all_probas.append(probas)
        del clf, X_tr_feat, X_te_feat, probas
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    elapsed    = time.perf_counter() - t_start
    avg_probas = np.mean(all_probas, axis=0)
    y_pred     = classes_[np.argmax(avg_probas, axis=1)]
    acc        = float(np.mean(y_pred == y_te))

    if pos_label in classes_:
        pos_idx    = list(classes_).index(pos_label)
        pos_probas = avg_probas[:, pos_idx]
        auc        = float(roc_auc_score(
            [1 if t == pos_label else 0 for t in y_te],
            pos_probas,
        ))
    else:
        auc = float("nan")

    n_correct = int(np.sum(y_pred == y_te))

    # Print wrong predictions
    wrong = [(te_idx[i], paths[te_idx[i]], y_te[i], y_pred[i])
             for i in range(len(y_te)) if y_pred[i] != y_te[i]]

    print(f"Accuracy  : {acc:.4f}  ({n_correct}/{len(te_idx)} correct)")
    print(f"ROC AUC   : {auc:.4f}")
    print(f"Time      : {elapsed:.1f}s")

    if wrong:
        print(f"\nWrong predictions ({len(wrong)}):")
        for idx, path, true_label, pred_label in wrong:
            print(f"  sample {idx+1:>3}  true={true_label}  pred={pred_label}  {path.name}")
    else:
        print("\nAll test samples predicted correctly.")

    result = {
        "method":       f"holdout_{int((1-test_size)*100)}/{int(test_size*100)}",
        "random_state": random_state,
        "n_train":      len(tr_idx),
        "n_test":       len(te_idx),
        "n_groups":     n_groups,
        "n_kernels":    n_kernels,
        "n_features":   2 * n_kernels,
        "n_correct":    n_correct,
        "acc":          acc,
        "roc_auc":      auc,
        "time_s":       round(elapsed, 1),
    }
    return result, paths, y_te, y_pred, te_idx


# ── run report ───────────────────────────────────────────────────────────────

def save_run_report(
    result: dict,
    paths: list[Path],
    y_all: np.ndarray,
    predictions: list[str],
    device: str,
    *,
    reports_root: Path | None = None,
) -> Path:
    """Save a timestamped report folder with a summary text file and wrong-prediction plots.

    Creates under ``GOALTEND_OUTPUT_DIR/tabpfn_runs`` (unless ``reports_root`` is set):

        <ts>/summary.txt
        <ts>/wrong_<N>_true-<label>_pred-<label>.png  (one per mistake)

    Returns the path to the report folder.
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend so figures save without a display
    import matplotlib.pyplot as plt
    from datetime import datetime

    ts         = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base       = (outputs_dir() / "tabpfn_runs") if reports_root is None else reports_root
    report_dir = base / ts
    report_dir.mkdir(parents=True, exist_ok=True)

    # ── summary.txt ──────────────────────────────────────────────────────────
    wrong = [(i, paths[i], y_all[i], predictions[i])
             for i in range(len(predictions))
             if predictions[i] != y_all[i]]

    lines = [
        "=" * 52,
        "  GOALTEND CLASSIFIER  –  RUN REPORT",
        "=" * 52,
        f"  Timestamp        : {ts}",
        "",
        "── Setup ──────────────────────────────────────────",
        f"  n_groups         : {result['n_groups']}",
        f"  n_kernels        : {result['n_kernels']}",
        f"  n_features       : {result['n_features']}",
        f"  device           : {device}",
        f"  validation       : leave-one-out ({result['n_samples']} samples)",
        "",
        "── Results ────────────────────────────────────────",
        f"  Accuracy         : {result['acc']:.4f}  ({result['n_correct']}/{result['n_samples']} correct)",
        f"  ROC AUC          : {result['roc_auc']:.4f}",
        f"  Avg time / iter  : {result['time_per_iter_s']:.1f}s",
        f"  Total time       : {result['time_total_s']:.1f}s",
        "",
        "── Wrong predictions ───────────────────────────────",
    ]

    if wrong:
        for i, path, true_label, pred_label in wrong:
            lines.append(f"  sample {i+1:>3}  true={true_label:<10}  pred={pred_label}  {path.name}")
    else:
        lines.append("  None – all samples predicted correctly.")

    lines.append("=" * 52)

    summary_path = report_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport saved → {report_dir}")

    # ── wrong-prediction plots ────────────────────────────────────────────────
    TIME_COL    = "Latest: Time (s)"
    short_names = ["X1", "Y1", "Z1", "Z2", "Y2", "X2"]
    colors      = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]

    for i, path, true_label, pred_label in wrong:
        df = pd.read_csv(path, usecols=[TIME_COL] + ACCEL_COLS)
        t  = df[TIME_COL].values - df[TIME_COL].values[0]

        fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
        for ax, col, name, color in zip(axes, ACCEL_COLS, short_names, colors):
            ax.plot(t, df[col].values, color=color, linewidth=0.8)
            ax.set_ylabel(f"{name}\n(m/s²)", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"Sample {i+1}  |  {path.name}\nTrue: {true_label}   Pred: {pred_label}  ✗",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()

        fname = f"wrong_{i+1:03d}_true-{true_label}_pred-{pred_label}.png"
        fig.savefig(report_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  saved {fname}")

    return report_dir


# ── standalone plot (no training needed) ─────────────────────────────────────
# Run this directly to inspect any sample without re-training:
#   python3 goaltend_classify.py --plot 2

def plot_only(sample_number: int) -> None:
    """Load the dataset and plot one sample by its 1-based index."""
    _, y_all, paths = load_goaltend_dataset(DATA_DIR)  # raw_samples unused here
    idx = sample_number - 1
    if idx < 0 or idx >= len(paths):
        raise ValueError(f"Sample number must be between 1 and {len(paths)}, got {sample_number}.")
    print(f"Plotting sample {sample_number}: {paths[idx].name}  label={y_all[idx]}")
    plot_sample(paths[idx], label=y_all[idx])


# ── single-sample prediction ──────────────────────────────────────────────────
# Train on everything except the named file, predict that file, print probabilities.
#   python3 goaltend_classify.py --predict block_light_1_event_01

def predict_one(
    filename:  str,
    n_groups:  int = N_GROUPS,
    n_kernels: int = N_KERNELS,
    device:    str = "cpu",
) -> None:
    """Train on all samples except the named file and predict that one sample.

    Parameters
    ----------
    filename  : CSV stem or full name, e.g. "block_light_1_event_01" or
                "block_light_1_event_01.csv"
    """
    from aeon.transformations.collection.convolution_based import Rocket
    from tabpfn import TabPFNClassifier

    raw_samples, y, paths = load_goaltend_dataset(DATA_DIR)

    # Find the target sample by filename (stem match, case-insensitive)
    stem = filename.replace(".csv", "").lower()
    matches = [i for i, p in enumerate(paths) if p.stem.lower() == stem]
    if not matches:
        available = [p.stem for p in paths]
        raise ValueError(
            f"File '{filename}' not found in the dataset.\n"
            f"Available stems: {available}"
        )
    te_idx = matches[0]
    tr_idx = [i for i in range(len(raw_samples)) if i != te_idx]

    print(f"\nTarget  : {paths[te_idx].name}")
    print(f"Label   : {y[te_idx]}")
    print(f"Training on {len(tr_idx)} samples...\n")

    # Pad using training T_max only
    tr_raw = [raw_samples[i] for i in tr_idx]
    te_raw = [raw_samples[te_idx]]
    T_tr   = max(s.shape[1] for s in tr_raw)
    X_tr   = pad_to(tr_raw, T_tr)
    X_te   = pad_to(te_raw, T_tr)
    y_tr   = y[tr_idx]

    all_probas: list[np.ndarray] = []
    classes_   = None

    for g in range(n_groups):
        seed   = g
        rocket = Rocket(n_kernels=n_kernels, random_state=seed, n_jobs=-1)
        rocket.fit(X_tr)
        X_tr_feat = rocket.transform(X_tr)
        X_te_feat = rocket.transform(X_te)
        del rocket

        clf = TabPFNClassifier(
            n_estimators=8,
            device=device,
            random_state=seed,
            ignore_pretraining_limits=True,
        )
        clf.fit(X_tr_feat, y_tr)
        probas = clf.predict_proba(X_te_feat)
        if classes_ is None:
            classes_ = clf.classes_
        all_probas.append(probas)
        del clf, X_tr_feat, X_te_feat, probas

    avg_probas = np.mean(all_probas, axis=0)[0]   # shape (n_classes,)
    pred_label = classes_[np.argmax(avg_probas)]
    correct    = pred_label == y[te_idx]

    print("── Prediction ──────────────────────────────────")
    for cls, prob in sorted(zip(classes_, avg_probas), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<12} {prob:.4f}  {bar}")
    print(f"\n  Predicted : {pred_label}  {'✓ correct' if correct else '✗ wrong'}")
    print(f"  True label: {y[te_idx]}")
    print("────────────────────────────────────────────────")

    plot_sample(paths[te_idx], label=y[te_idx], pred=pred_label)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Quick plot mode: python3 goaltend_classify.py --plot <N>
    if "--plot" in sys.argv:
        n = int(sys.argv[sys.argv.index("--plot") + 1])
        plot_only(n)
        sys.exit(0)

    # 80/20 holdout: python3 goaltend_classify.py --holdout
    if "--holdout" in sys.argv:
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using {dev.upper()}")
        evaluate_goaltend_holdout(device=dev)
        sys.exit(0)

    # Single-sample predict: python3 goaltend_classify.py --predict block_light_1_event_01
    if "--predict" in sys.argv:
        fname = sys.argv[sys.argv.index("--predict") + 1]
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        predict_one(fname, device=dev)
        sys.exit(0)

    import torch
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("MPS not available, falling back to CPU")

    result, paths, predictions = evaluate_goaltend(
        data_dir  = DATA_DIR,
        n_groups  = N_GROUPS,
        n_kernels = N_KERNELS,
        device    = device,
    )

    _, y_all, _ = load_goaltend_dataset(DATA_DIR)   # labels only needed here
    save_run_report(result, paths, y_all, predictions, device=device)

# 80/20 holdout: python3 goaltend_classify.py --holdout