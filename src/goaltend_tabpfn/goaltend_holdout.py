"""
goaltend_holdout.py
-------------------
Stratified 80/20 train/test split for goaltend classification.

Usage
-----
Train and evaluate on labelled data:
    python3 goaltend_holdout.py

Predict on unseen data in a "test" folder (no labels needed):
    python3 goaltend_holdout.py --test

Dataset layout
--------------
    Ball on Rim - Segmented/        → label "legal"
    Blocks - Segmented/             → label "legal"
    Hand on Backboard - Segmented/  → label "legal"
    Goaltends - Segmented/          → label "goaltend"
    test/                           → unseen samples (used with --test flag)

Authentication
--------------
Accept the TabPFN licence at https://huggingface.co/Prior-Labs/TabPFN-v2-clf
then set HF_TOKEN=<your_token> or let the script prompt you.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from goaltend_close_call.paths import data_root, outputs_dir

# ── configuration ─────────────────────────────────────────────────────────────

DATA_DIR = data_root()
TEST_DIR = DATA_DIR / "test"           # folder for unseen samples (--test mode)
FIT_PATH = outputs_dir() / "fitted_tabpfn_pipeline.joblib"

N_KERNELS    = 1000
RANDOM_STATE = 42                      # controls the train/test split
TEST_SIZE    = 0.20                    # 20 % held out for testing

FOLDER_LABELS: dict[str, str] = {
    "Ball on Rim - Segmented":       "legal",
    "Blocks - Segmented":            "legal",
    "Hand on Backboard - Segmented": "legal",
    "Goaltends - Segmented":         "goaltend",
}

ACCEL_COLS = [
    "Latest: X Acceleration 1 (m/s²)",
    "Latest: Y Acceleration 1 (m/s²)",
    "Latest: Z Acceleration 1 (m/s²)",
    "Latest: Z Acceleration 2 (m/s²)",
    "Latest: Y Acceleration 2 (m/s²)",
    "Latest: X Acceleration 2 (m/s²)",
]

# ── HuggingFace authentication ─────────────────────────────────────────────────

def _hf_login() -> None:
    try:
        from huggingface_hub import login, whoami
        token = os.environ.get("HF_TOKEN")
        try:
            whoami()
        except Exception:
            if token:
                login(token=token, add_to_git_credential=False)
            else:
                print(
                    "HuggingFace login required.\n"
                    "Accept licence : https://huggingface.co/Prior-Labs/TabPFN-v2-clf\n"
                    "Create token   : https://huggingface.co/settings/tokens\n"
                )
                login(add_to_git_credential=False)
    except ImportError:
        raise ImportError("Run: pip install huggingface_hub")

_hf_login()

# ── data loading ───────────────────────────────────────────────────────────────

def load_labelled(data_dir: Path = DATA_DIR) -> tuple[list[np.ndarray], np.ndarray, list[Path]]:
    """Load all labelled CSV files as variable-length (6, T_i) arrays."""
    raw, labels, paths = [], [], []
    for folder_name, label in FOLDER_LABELS.items():
        folder = data_dir / folder_name
        if not folder.exists():
            print(f"  [warn] folder not found, skipping: {folder}")
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            df  = pd.read_csv(csv_path, usecols=ACCEL_COLS)
            raw.append(df[ACCEL_COLS].values.T.astype(np.float32))
            labels.append(label)
            paths.append(csv_path)

    if not raw:
        raise ValueError(f"No CSV files found under {data_dir}")

    y = np.array(labels)
    unique, counts = np.unique(y, return_counts=True)
    print(f"Loaded {len(raw)} labelled samples")
    print(f"  Class counts : { {k: v for k, v in zip(unique, counts)} }")
    return raw, y, paths


def load_unlabelled(test_dir: Path = TEST_DIR) -> tuple[list[np.ndarray], list[Path]]:
    """Load CSV files from the test folder (no labels).

    Automatically renames sensor-3 columns to sensor-2 so that files recorded
    with a different sensor index still match the trained model's feature names.
        e.g. "Latest: X Acceleration 3 (m/s²)" → "Latest: X Acceleration 2 (m/s²)"
    """
    sensor3_to_sensor2 = {
        "Latest: X Acceleration 3 (m/s²)": "Latest: X Acceleration 2 (m/s²)",
        "Latest: Y Acceleration 3 (m/s²)": "Latest: Y Acceleration 2 (m/s²)",
        "Latest: Z Acceleration 3 (m/s²)": "Latest: Z Acceleration 2 (m/s²)",
    }

    raw, paths = [], []
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")
    for csv_path in sorted(test_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df = df.rename(columns=sensor3_to_sensor2)
        missing = [c for c in ACCEL_COLS if c not in df.columns]
        if missing:
            print(f"  [warn] skipping {csv_path.name} — missing columns: {missing}")
            continue
        raw.append(df[ACCEL_COLS].values.T.astype(np.float32))
        paths.append(csv_path)
    if not raw:
        raise ValueError(f"No usable CSV files found in {test_dir}")
    print(f"Loaded {len(raw)} unseen test samples from {test_dir}")
    return raw, paths


def pad_to(samples: list[np.ndarray], T: int) -> np.ndarray:
    """Zero-pad or truncate samples to length T and stack into (N, 6, T)."""
    N, M = len(samples), samples[0].shape[0]
    X    = np.zeros((N, M, T), dtype=np.float32)
    for i, s in enumerate(samples):
        t = min(s.shape[1], T)
        X[i, :, :t] = s[:, :t]
    return X


def zscore_normalize(X: np.ndarray) -> np.ndarray:
    """Per-sample, per-channel z-score normalization.

    Subtracts the mean and divides by the std of each channel independently
    for each sample. This makes the model invariant to absolute acceleration
    magnitude, which is critical when test data comes from a different sensor,
    person, or mounting position than the training data.

    Input/output shape: (N, 6, T)
    """
    out = X.copy()
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            ch   = X[i, c, :]
            mean = ch.mean()
            std  = ch.std()
            out[i, c, :] = (ch - mean) / (std + 1e-8)   # +1e-8 avoids div-by-zero
    return out

# ── ROCKET feature extraction ──────────────────────────────────────────────────

def rocket_features(
    X_tr_raw: list[np.ndarray],
    X_te_raw: list[np.ndarray],
    n_kernels: int = N_KERNELS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ROCKET on training samples, transform both train and test.

    Training samples set the canonical length T_tr.
    Test samples are zero-padded or truncated to T_tr.
    """
    from aeon.transformations.collection.convolution_based import Rocket

    T_tr  = max(s.shape[1] for s in X_tr_raw)
    X_tr  = zscore_normalize(pad_to(X_tr_raw, T_tr))
    X_te  = zscore_normalize(pad_to(X_te_raw, T_tr))

    rocket = Rocket(n_kernels=n_kernels, random_state=seed, n_jobs=-1)
    rocket.fit(X_tr)
    return rocket.transform(X_tr), rocket.transform(X_te), rocket

# ── 80/20 train/test evaluation ────────────────────────────────────────────────

def run_holdout() -> None:
    from tabpfn import TabPFNClassifier
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device : {device.upper()}\n")

    raw, y, paths = load_labelled()

    # Stratified split — preserves goaltend/legal ratio in both sets
    indices = np.arange(len(raw))
    tr_idx, te_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    tr_raw = [raw[i] for i in tr_idx]
    te_raw = [raw[i] for i in te_idx]
    y_train = y[tr_idx]
    y_test  = y[te_idx]

    u_tr, c_tr = np.unique(y_train, return_counts=True)
    u_te, c_te = np.unique(y_test,  return_counts=True)
    print(f"Train : {len(tr_idx)} samples  { {k: v for k, v in zip(u_tr, c_tr)} }")
    print(f"Test  : {len(te_idx)} samples  { {k: v for k, v in zip(u_te, c_te)} }")
    print()

    # ROCKET features
    print("Extracting ROCKET features...")
    X_train, X_test, _ = rocket_features(tr_raw, te_raw, n_kernels=N_KERNELS)

    # TabPFN
    print("Fitting TabPFN...")
    clf = TabPFNClassifier(
        n_estimators=8,
        device=device,
        random_state=RANDOM_STATE,
        ignore_pretraining_limits=True,
    )
    clf.fit(X_train, y_train)

    # Predictions
    prediction_probabilities = clf.predict_proba(X_test)
    predictions              = clf.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    classes = list(clf.classes_)
    pos_idx = classes.index("goaltend")
    auc = roc_auc_score(
        [1 if t == "goaltend" else 0 for t in y_test],
        prediction_probabilities[:, pos_idx],
    )

    print("\n── Results ─────────────────────────────────────")
    print(f"  Accuracy : {acc:.4f}  ({int(acc * len(y_test))}/{len(y_test)} correct)")
    print(f"  ROC AUC  : {auc:.4f}")

    wrong = [(te_idx[i], paths[te_idx[i]], y_test[i], predictions[i])
             for i in range(len(y_test)) if predictions[i] != y_test[i]]
    if wrong:
        print(f"\n  Wrong predictions ({len(wrong)}):")
        for idx, path, true_l, pred_l in wrong:
            print(f"    sample {idx+1:>3}  true={true_l:<10}  pred={pred_l}  {path.name}")
    else:
        print("\n  All test samples predicted correctly.")
    print("────────────────────────────────────────────────")

    # Save fitted objects for use on unseen data later
    import joblib

    FIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "clf": clf,
            "T_tr": max(s.shape[1] for s in tr_raw),
            "n_kernels": N_KERNELS,
            "classes": classes,
        },
        FIT_PATH,
    )
    print(f"\nFitted pipeline saved → {FIT_PATH}")
    print("Run with --test to predict unseen samples using this pipeline.")

# ── predict unseen samples from test/ folder ──────────────────────────────────

def run_test() -> None:
    import joblib
    from aeon.transformations.collection.convolution_based import Rocket

    if not FIT_PATH.exists():
        raise FileNotFoundError(
            "No fitted pipeline found. Run without --test first to train."
        )

    print(f"Loading pipeline from {FIT_PATH}...")
    pipe  = joblib.load(FIT_PATH)
    clf   = pipe["clf"]
    T_tr  = pipe["T_tr"]
    n_kernels = pipe["n_kernels"]
    classes   = pipe["classes"]

    te_raw, paths = load_unlabelled()

    # Reproduce the exact same training split to refit ROCKET with the same kernels
    raw_all, y_all, _ = load_labelled()
    tr_idx, _ = train_test_split(
        np.arange(len(raw_all)),
        test_size=TEST_SIZE,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )
    tr_raw = [raw_all[i] for i in tr_idx]
    T_tr   = max(s.shape[1] for s in tr_raw)
    X_tr   = zscore_normalize(pad_to(tr_raw, T_tr))
    X_te   = zscore_normalize(pad_to(te_raw, T_tr))

    rocket = Rocket(n_kernels=n_kernels, random_state=0, n_jobs=-1)
    rocket.fit(X_tr)
    X_te_feat = rocket.transform(X_te)

    probas      = clf.predict_proba(X_te_feat)
    predictions = clf.predict(X_te_feat)
    pos_idx     = classes.index("goaltend")

    print("\n── Predictions on unseen test samples ──────────")
    for i, (path, pred) in enumerate(zip(paths, predictions)):
        p_goaltend = probas[i, pos_idx]
        p_legal    = 1 - p_goaltend
        bar_g = "█" * int(p_goaltend * 20)
        bar_l = "█" * int(p_legal    * 20)
        print(f"  {path.name}")
        print(f"    goaltend  {p_goaltend:.4f}  {bar_g}")
        print(f"    legal     {p_legal:.4f}  {bar_l}")
        print(f"    → PREDICTION: {pred}\n")
    print("────────────────────────────────────────────────")

    # ── plot X1, Y1, Z1 for each test event ──────────────────────────────────
    import matplotlib.pyplot as plt

    TIME_COL = "Latest: Time (s)"
    PLOT_COLS = {
        "X1": "Latest: X Acceleration 1 (m/s²)",
        "Y1": "Latest: Y Acceleration 1 (m/s²)",
        "Z1": "Latest: Z Acceleration 1 (m/s²)",
    }
    COLORS = {"X1": "#e6194b", "Y1": "#3cb44b", "Z1": "#4363d8"}

    sensor3_to_sensor2 = {
        "Latest: X Acceleration 3 (m/s²)": "Latest: X Acceleration 2 (m/s²)",
        "Latest: Y Acceleration 3 (m/s²)": "Latest: Y Acceleration 2 (m/s²)",
        "Latest: Z Acceleration 3 (m/s²)": "Latest: Z Acceleration 2 (m/s²)",
    }

    for i, (path, pred) in enumerate(zip(paths, predictions)):
        p_goaltend = probas[i, pos_idx]
        df = pd.read_csv(path).rename(columns=sensor3_to_sensor2)
        t  = df[TIME_COL].values - df[TIME_COL].values[0]

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        for ax, (name, col) in zip(axes, PLOT_COLS.items()):
            ax.plot(t, df[col].values, color=COLORS[name], linewidth=0.9)
            ax.set_ylabel(f"{name}\n(m/s²)", fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"{path.name}\nPrediction: {pred}   p(goaltend)={p_goaltend:.4f}",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_test()
    else:
        run_holdout()
