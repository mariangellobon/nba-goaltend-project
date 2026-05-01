"""
Goaltend vs legal using **time-domain shape** features only (no STFT/FFT).

Training: labeled *Segmented* folders, optionally plus synthetic augmentations
(``USE_SYNTHETIC_TRAINING``). Uses **physical sensor 1** only (``SENSOR_1_ONLY``).
Evaluation: all 11 rows in close_calls_labels.csv (unseen holdout).

Writes ``outputs/close_calls_shape_predictions.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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


def build_close_call_table(feat_cols: list[str]) -> pd.DataFrame:
    lab = pd.read_csv(LABELS_PATH)
    lab["ground_truth"] = lab["ground_truth"].str.lower().str.strip()
    rows = []
    for _, r in lab.iterrows():
        p = CLOSE_DIR / r["filename"]
        feat = _features_for_file(p)
        feat["filename"] = r["filename"]
        feat["ground_truth"] = r["ground_truth"]
        rows.append(feat)
    df = pd.DataFrame(rows)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


def run(rf_params: dict | None = None) -> dict:
    params = {**RF_PARAMS, **(rf_params or {})}
    df_base, feat_cols = build_base_binary()
    df_all_cc = build_close_call_table(feat_cols)

    df_train = df_base
    n_synth = 0
    if USE_SYNTHETIC_TRAINING:
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

    X_train = df_train[feat_cols].values
    y_train = df_train["y"].values
    sc = StandardScaler()
    X_train_z = sc.fit_transform(X_train)
    clf = RandomForestClassifier(**params)
    clf.fit(X_train_z, y_train)

    cls = list(clf.classes_)
    i_goal = cls.index("goaltend") if "goaltend" in cls else 0
    i_leg = cls.index("legal") if "legal" in cls else 1

    out_rows = []
    for _, r in df_all_cc.iterrows():
        x = r[feat_cols].values.astype(np.float64, copy=False).reshape(1, -1)
        xz = sc.transform(x)
        proba = clf.predict_proba(xz)[0]
        p_goal = float(proba[i_goal])
        p_leg = float(proba[i_leg])
        pred = str(clf.predict(xz)[0])
        out_rows.append(
            {
                "filename": r["filename"],
                "ground_truth": r["ground_truth"],
                "predicted": pred,
                "P_goaltend": p_goal,
                "P_legal": p_leg,
            }
        )
    out_df = pd.DataFrame(out_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "close_calls_shape_predictions.csv"
    out_df.to_csv(out_path, index=False)

    labeled_mask = out_df["ground_truth"].isin(["legal", "goaltend"])
    acc_labeled_holdout = float(
        (out_df.loc[labeled_mask, "predicted"] == out_df.loc[labeled_mask, "ground_truth"]).mean()
    )

    return {
        "output_csv": str(out_path),
        "params": params,
        "labeled_close_accuracy_holdout": acc_labeled_holdout,
        "predictions_df": out_df,
        "feature_columns": feat_cols,
        "n_real_train": len(df_base),
        "n_synthetic_train": n_synth,
    }


if __name__ == "__main__":
    r = run()
    print("Params:", r["params"])
    print("Training rows — real segmented:", r["n_real_train"], "synthetic:", r["n_synthetic_train"])
    print(
        "Accuracy on 9 labeled close calls (holdout):",
        r["labeled_close_accuracy_holdout"],
    )
    print("Wrote:", r["output_csv"])
    print("\n" + Path(r["output_csv"]).read_text())
