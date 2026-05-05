"""
Goaltend vs legal using **time-domain shape** features only (no STFT/FFT).

Training pool (default): usable **labeled close calls only** (env ``GOALTEND_TRAIN_CLOSE_ONLY=0``
to include segmented folders). Uses **physical sensor 1** only (``SENSOR_1_ONLY``).

Evaluation: stratified K-fold on close calls; optional union with all segmented rows per fold.

Writes ``outputs/close_calls_shape_oof_predictions.csv``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .close_call_cv import stratified_kfold_eval_close_calls
from .close_call_labels import load_usable_close_call_binary_labels
from .close_call_model import segmented_stratified_kfold_accuracy
from .paths import data_root, outputs_dir
from .sensor_io import discover_segmented_folders, estimate_fs, load_recording_csv, crop_peak_window
from .shape_time_features import extract_shape_features

DATA_ROOT = data_root()
OUTPUT_DIR = outputs_dir()
WIN_SEC = 1.0
SENSOR_1_ONLY = True
USE_SYNTHETIC_TRAINING = False
LABELS_PATH = DATA_ROOT / "close_calls_labels.csv"
CLOSE_DIR = DATA_ROOT / "Close Calls"

CV_N_SPLITS = int(os.environ.get("GOALTEND_CC_CV_SPLITS", "5"))
CV_RANDOM_STATE = 42

TRAIN_CLOSE_ONLY = os.environ.get("GOALTEND_TRAIN_CLOSE_ONLY", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=1,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)


def _features_for_file(csv_path: Path) -> dict:
    t, a1, a2 = load_recording_csv(csv_path, sensor_1_only=SENSOR_1_ONLY)
    fs = estimate_fs(t)
    t, a1, a2 = crop_peak_window(
        t, a1, a2, win_sec=WIN_SEC, fs=fs, sensor_1_only=SENSOR_1_ONLY
    )
    return extract_shape_features(t, a1, a2, fs=fs, sensor_1_only=SENSOR_1_ONLY)


def _extract_shape_for_synthetic(t, a1, a2, fs: float) -> dict:
    return extract_shape_features(t, a1, a2, fs=fs, sensor_1_only=SENSOR_1_ONLY)


def build_base_binary() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for folder, label in discover_segmented_folders(DATA_ROOT):
        if folder.name == "Other Data - Segmented":
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            feat = _features_for_file(csv_path)
            feat["y"] = "goaltend" if label == "goaltends" else "legal"
            rows.append(feat)
    df = pd.DataFrame(rows)
    feat_cols = sorted([c for c in df.columns if c != "y"])
    return df, feat_cols


def build_usable_close_calls_df(feat_cols: list[str]) -> pd.DataFrame:
    manifest = load_usable_close_call_binary_labels(LABELS_PATH, CLOSE_DIR)
    rows = []
    for _, r in manifest.iterrows():
        feat = _features_for_file(Path(r["path"]))
        feat["y"] = r["y"]
        feat["filename"] = r["filename"]
        rows.append(feat)
    df = pd.DataFrame(rows)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


def run(rf_params: dict | None = None) -> dict:
    params = {**RF_PARAMS, **(rf_params or {})}
    df_base, feat_cols = build_base_binary()
    df_cc = build_usable_close_calls_df(feat_cols)

    if TRAIN_CLOSE_ONLY:
        cv_summary_seg = {
            "cv_n_splits_requested": CV_N_SPLITS,
            "cv_n_splits_effective": 0,
            "cv_fold_accuracies": [],
            "cv_mean_accuracy": float("nan"),
            "cv_std_accuracy": float("nan"),
            "segmented_cv_skipped": True,
        }
    else:
        cv_summary_seg = segmented_stratified_kfold_accuracy(
            df_base, feat_cols, rf_params=params
        )

    df_train = df_base
    n_synth = 0
    if USE_SYNTHETIC_TRAINING and not TRAIN_CLOSE_ONLY:
        from .synthetic_close_training import (
            SYNTHETIC_AUGMENTS_PER_SOURCE_FILE,
            build_synthetic_feature_rows,
        )

        df_syn = build_synthetic_feature_rows(
            DATA_ROOT,
            _extract_shape_for_synthetic,
            augments_per_file=SYNTHETIC_AUGMENTS_PER_SOURCE_FILE,
            win_sec=WIN_SEC,
            sensor_1_only=SENSOR_1_ONLY,
            rng=np.random.default_rng(42),
        )
        for c in feat_cols:
            if c not in df_syn.columns:
                df_syn[c] = 0.0
        df_syn = df_syn[feat_cols + ["y"]]
        df_train = pd.concat([df_base, df_syn], ignore_index=True)
        n_synth = len(df_syn)

    seg_side = df_base.iloc[0:0] if TRAIN_CLOSE_ONLY else df_train

    cc_cv = stratified_kfold_eval_close_calls(
        seg_side,
        df_cc,
        feat_cols,
        rf_params=params,
        include_segmented=not TRAIN_CLOSE_ONLY,
        n_splits=CV_N_SPLITS,
        random_state=CV_RANDOM_STATE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "close_calls_shape_oof_predictions.csv"
    cc_cv["oof_predictions_df"].to_csv(out_path, index=False)

    return {
        "output_csv": str(out_path),
        "params": params,
        "labeled_close_accuracy_oof": cc_cv["oof_accuracy_close_calls"],
        "close_call_fold_accuracies": cc_cv["fold_accuracies_close_calls_only"],
        "oof_predictions_df": cc_cv["oof_predictions_df"],
        "feature_columns": feat_cols,
        "n_real_train": len(df_base),
        "n_synthetic_train": n_synth,
        "n_close_calls_usable": len(df_cc),
        **cv_summary_seg,
        "close_call_cv_n_splits_requested": cc_cv["cv_n_splits_requested"],
        "close_call_cv_n_splits_effective": cc_cv["cv_n_splits_effective"],
        "train_close_calls_only": TRAIN_CLOSE_ONLY,
        "include_segmented_in_train": cc_cv["include_segmented_in_train"],
    }


if __name__ == "__main__":
    r = run()
    print("Params:", r["params"])
    print(
        "Training mode:",
        "labeled close calls only"
        if r["train_close_calls_only"]
        else "segmented (+ optional synth) + close-call folds",
    )
    print("Training rows — real segmented:", r["n_real_train"], "synthetic:", r["n_synthetic_train"])
    print("Usable labeled close calls:", r["n_close_calls_usable"])
    print(
        "Close-call CV splits (effective):",
        r["close_call_cv_n_splits_effective"],
        "(requested",
        r["close_call_cv_n_splits_requested"],
        ")",
    )
    print(
        "Per-fold accuracy (close-call test sets):",
        [round(x, 4) for x in r["close_call_fold_accuracies"]],
    )
    print(
        "OOF accuracy on close calls (no leakage):",
        round(r["labeled_close_accuracy_oof"], 4),
    )
    if r.get("segmented_cv_skipped"):
        print("Stratified CV on segmented: skipped (training uses close calls only)")
    print("Wrote:", r["output_csv"])
    print("\n" + Path(r["output_csv"]).read_text())
