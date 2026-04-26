"""
Goaltend vs legal model for close-call trials.

- Training: **only** labeled *Segmented* folders (101 binary rows); close-call CSVs are **not** in training.
- Synthetic augmentations: off (``USE_SYNTHETIC_TRAINING``).
- Sensors: **two** accelerometer triples per ``sensor_io`` (Blocks etc.: 1+2; Goaltends folder: 1+3).
- Predictions: **all** rows in ``close_calls_labels.csv`` (11 files), with human ``ground_truth`` for comparison where present.

Writes ``outputs/close_calls_predictions.csv`` (see ``paths.outputs_dir``).

Also reports **stratified k-fold accuracy** on the segmented table only (see ``segmented_stratified_kfold_accuracy``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .paths import data_root, outputs_dir
from .sensor_io import discover_segmented_folders, estimate_fs, load_recording_csv, crop_peak_window
from .spectrogram_features import extract_features

DATA_ROOT = data_root()
OUTPUT_DIR = outputs_dir()
WIN_SEC = 1.0
NPERSEG = 256
SENSOR_1_ONLY = False
LABELS_PATH = DATA_ROOT / "close_calls_labels.csv"
CLOSE_DIR = DATA_ROOT / "Close Calls"

USE_SYNTHETIC_TRAINING = False

CV_N_SPLITS = 5
CV_RANDOM_STATE = 42

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
    return extract_features(
        t, a1, a2, fs=fs, nperseg=NPERSEG, sensor_1_only=SENSOR_1_ONLY
    )


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
    """Feature rows for every close-call file listed in ``close_calls_labels.csv``."""
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


def segmented_stratified_kfold_accuracy(
    df_base: pd.DataFrame,
    feat_cols: list[str],
    *,
    rf_params: dict | None = None,
    n_splits: int = CV_N_SPLITS,
    random_state: int = CV_RANDOM_STATE,
) -> dict[str, float | int | list[float]]:
    """
    Stratified k-fold accuracy on the segmented table only (no close calls).
    Uses ``Pipeline(StandardScaler, RandomForest)`` so scaling is fit per training fold.
    ``n_splits`` is lowered if the minority class count is smaller than ``n_splits``.
    """
    params = {**RF_PARAMS, **(rf_params or {})}
    cv_clf_params = {**params, "n_jobs": 1}
    X = df_base[feat_cols].values
    y = df_base["y"].values
    min_class = int(pd.Series(y).value_counts().min())
    k = int(min(n_splits, min_class))
    if k < 2:
        raise ValueError(
            f"Need at least 2 samples per class for CV; got min class count {min_class}"
        )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**cv_clf_params)),
        ]
    )
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return {
        "cv_n_splits_requested": n_splits,
        "cv_n_splits_effective": k,
        "cv_fold_accuracies": [float(s) for s in scores],
        "cv_mean_accuracy": float(np.mean(scores)),
        "cv_std_accuracy": float(np.std(scores)),
    }


def run(rf_params: dict | None = None) -> dict:
    params = {**RF_PARAMS, **(rf_params or {})}
    df_base, feat_cols = build_base_binary()
    df_all_cc = build_close_call_table(feat_cols)

    cv_summary = segmented_stratified_kfold_accuracy(
        df_base, feat_cols, rf_params=params
    )

    df_train = df_base
    n_synth = 0
    if USE_SYNTHETIC_TRAINING:
        from .synthetic_close_training import (
            SYNTHETIC_AUGMENTS_PER_SOURCE_FILE,
            build_synthetic_feature_rows,
        )

        def _extract_synth(t, a1, a2, fs: float) -> dict:
            return extract_features(
                t, a1, a2, fs=fs, nperseg=NPERSEG, sensor_1_only=SENSOR_1_ONLY
            )

        df_syn = build_synthetic_feature_rows(
            DATA_ROOT,
            _extract_synth,
            augments_per_file=SYNTHETIC_AUGMENTS_PER_SOURCE_FILE,
            win_sec=WIN_SEC,
            sensor_1_only=SENSOR_1_ONLY,
            rng=np.random.default_rng(42),
        )
        for c in feat_cols:
            if c not in df_syn.columns:
                df_syn[c] = 0.0
        df_syn = df_syn[feat_cols + ["y"]]
        df_train = pd.concat([df_train, df_syn], ignore_index=True)
        n_synth = len(df_syn)

    X_train = df_train[feat_cols].values
    y_train = df_train["y"].values
    sc = StandardScaler()
    X_train_z = sc.fit_transform(X_train)
    clf = RandomForestClassifier(**params)
    clf.fit(X_train_z, y_train)
    train_pred = clf.predict(X_train_z)
    train_accuracy = float(np.mean(train_pred == y_train))

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
    out_path = OUTPUT_DIR / "close_calls_predictions.csv"
    out_df.to_csv(out_path, index=False)

    labeled_mask = out_df["ground_truth"].isin(["legal", "goaltend"])
    acc_labeled_holdout = float(
        (out_df.loc[labeled_mask, "predicted"] == out_df.loc[labeled_mask, "ground_truth"]).mean()
    )

    out = {
        "output_csv": str(out_path),
        "params": params,
        "predictions_df": out_df,
        "n_segmented_train": len(df_base),
        "n_synthetic_train": n_synth,
        "n_close_calls_scored": len(out_df),
        "train_accuracy": train_accuracy,
        "labeled_close_accuracy_holdout": acc_labeled_holdout,
    }
    out.update(cv_summary)
    return out


if __name__ == "__main__":
    r = run()
    print("Params:", r["params"])
    print(
        "Training rows — segmented:",
        r["n_segmented_train"],
        "synthetic:",
        r["n_synthetic_train"],
    )
    print("Close-call rows scored:", r["n_close_calls_scored"])
    print("Training accuracy (segmented only):", round(r["train_accuracy"], 4))
    print(
        "Accuracy on 9 labeled close calls (true holdout vs segmented-only train):",
        round(r["labeled_close_accuracy_holdout"], 4),
    )
    print(
        "Stratified CV on segmented only —",
        f"k={r['cv_n_splits_effective']} (requested {r['cv_n_splits_requested']}),",
        "fold accs:",
        [round(x, 4) for x in r["cv_fold_accuracies"]],
    )
    print(
        "CV mean ± std:",
        round(r["cv_mean_accuracy"], 4),
        "±",
        round(r["cv_std_accuracy"], 4),
    )
    print("Wrote:", r["output_csv"])
    print("\n" + Path(r["output_csv"]).read_text())
