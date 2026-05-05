"""
Stratified K-fold evaluation on close-call clips: each fold trains on either (a) all
segmented rows plus close-call training folds, or (b) **only** close-call training folds
when ``include_segmented=False``. The held-out fold is always close-call clips only (no leakage).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def stratified_kfold_eval_close_calls(
    df_seg: pd.DataFrame,
    df_cc: pd.DataFrame,
    feat_cols: list[str],
    *,
    rf_params: dict | None = None,
    make_pipeline: Callable[[], Pipeline] | None = None,
    include_segmented: bool = True,
    n_splits: int = 5,
    random_state: int = 42,
    y_seg_col: str = "y",
    y_cc_col: str = "y",
    filename_col: str = "filename",
) -> dict:
    """
    ``df_seg``: segmented rows with ``feat_cols`` + ``y_seg_col`` (ignored when
    ``include_segmented`` is False).
    ``df_cc``: one row per usable close call with ``feat_cols`` + ``y_cc_col`` + ``filename_col``.

    Each fold trains on CC rows outside the test fold, optionally union ``df_seg``, then
    evaluates accuracy only on the held-out CC fold. Splits always use **StratifiedKFold**
    on ``y_cc`` so legal/goaltend proportions stay similar across folds.
    """
    if df_cc.empty:
        raise ValueError("No usable labeled close calls for CV.")

    X_cc = df_cc[feat_cols].values.astype(np.float64, copy=False)
    y_cc = df_cc[y_cc_col].values
    fn_cc = df_cc[filename_col].values

    if include_segmented:
        X_seg = df_seg[feat_cols].values.astype(np.float64, copy=False)
        y_seg = df_seg[y_seg_col].values
    else:
        z = np.zeros((0, len(feat_cols)), dtype=np.float64)
        X_seg = z
        y_seg = np.array([], dtype=object)

    min_class = int(pd.Series(y_cc).value_counts().min())
    k = int(min(n_splits, len(df_cc), max(min_class, 2)))
    if k < 2:
        raise ValueError(
            f"Need at least 2 splits and 2 samples per class for stratified CC CV; "
            f"got {len(df_cc)} close-call rows, min class count {min_class}."
        )

    fold_acc: list[float] = []
    oof_pred = np.empty(len(df_cc), dtype=object)
    oof_fold = np.full(len(df_cc), -1, dtype=np.int32)
    p_goal = np.zeros(len(df_cc), dtype=np.float64)
    p_leg = np.zeros(len(df_cc), dtype=np.float64)

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    def _default_make_pipeline() -> Pipeline:
        params = rf_params or {}
        cv_clf_params = {**params, "n_jobs": 1}
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(**cv_clf_params)),
            ]
        )

    _factory = make_pipeline or _default_make_pipeline

    for fold_id, (train_idx, test_idx) in enumerate(
        cv.split(np.zeros(len(df_cc)), y_cc)
    ):
        pipe = _factory()
        cc_train = df_cc.iloc[train_idx]
        cc_test = df_cc.iloc[test_idx]

        X_tr_cc = cc_train[feat_cols].values.astype(np.float64)
        y_tr_cc = cc_train[y_cc_col].values
        if include_segmented:
            X_train = np.vstack([X_seg, X_tr_cc])
            y_train = np.concatenate([y_seg, y_tr_cc])
        else:
            X_train = X_tr_cc
            y_train = y_tr_cc

        pipe.fit(X_train, y_train)

        X_te = cc_test[feat_cols].values.astype(np.float64)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)
        cls = list(pipe.named_steps["clf"].classes_)
        i_goal = cls.index("goaltend") if "goaltend" in cls else 0
        i_leg = cls.index("legal") if "legal" in cls else 1 - i_goal
        acc = float(np.mean(pred == cc_test[y_cc_col].values))
        fold_acc.append(acc)

        for i_local, j_global in enumerate(test_idx):
            oof_pred[j_global] = pred[i_local]
            oof_fold[j_global] = fold_id
            p_goal[j_global] = float(proba[i_local, i_goal])
            p_leg[j_global] = float(proba[i_local, i_leg])

    oof_correct = oof_pred == y_cc
    oof_accuracy = float(np.mean(oof_correct))

    out_df = pd.DataFrame(
        {
            filename_col: fn_cc,
            "fold_id": oof_fold,
            y_cc_col: y_cc,
            "predicted": oof_pred,
            "P_goaltend": p_goal,
            "P_legal": p_leg,
            "correct": oof_correct,
        }
    )

    return {
        "cv_n_splits_requested": n_splits,
        "cv_n_splits_effective": k,
        "fold_accuracies_close_calls_only": fold_acc,
        "oof_accuracy_close_calls": oof_accuracy,
        "oof_predictions_df": out_df,
        "n_close_calls_evaluated": len(df_cc),
        "n_segmented_train_side": len(df_seg) if include_segmented else 0,
        "include_segmented_in_train": include_segmented,
    }


def write_oof_csv(out_df: pd.DataFrame, path: Path, extra_cols: pd.DataFrame | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist=True)
    out = out_df.copy()
    if extra_cols is not None:
        out = pd.concat([out.reset_index(drop=True), extra_cols.reset_index(drop=True)], axis=1)
    out.to_csv(path, index=False)
    return path
