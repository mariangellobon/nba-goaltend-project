"""
Load close-call label tables and map Eyeballed Contact / ground_truth to legal | goaltend.

Rows without a usable binary label (empty eyeballed, cant_tell, unknown text) are excluded
from training and from stratified CV evaluation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd

BinaryLabel = Literal["legal", "goaltend"]

def _strip_cell(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none") else s


def _normalize_header(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for c in out.columns:
        key = str(c).strip().lower().replace(" ", "_")
        if key in ("eyeballed_contact", "eyeballed"):
            rename[c] = "eyeballed_contact"
        elif key == "ground_truth":
            rename[c] = "ground_truth"
        elif key == "filename":
            rename[c] = "filename"
    out = out.rename(columns=rename)
    return out


def eyeballed_to_binary(text: str) -> tuple[BinaryLabel | None, str | None]:
    """
    Map free-text eyeballed label to legal/goaltend.

    Returns (label, None) or (None, reason) with reason in {cant_tell, ambiguous, empty}.
    """
    s = _strip_cell(text).lower()
    if not s:
        return None, "empty"
    if "so close" in s and "tell" in s:
        return None, "cant_tell"
    if re.search(r"\bcant\s*tell\b", s) or re.search(r"can\x27t\s*tell\b", s):
        return None, "cant_tell"
    # Definite goaltend wording (includes "goaltend?", "obvi goaltend", "close goaltend")
    if re.search(r"goaltend", s):
        return "goaltend", None
    # Blocks / legal contacts
    if re.search(r"\bblock", s) or "obvi block" in s or "obvious block" in s:
        return "legal", None
    return None, "ambiguous"


def row_to_binary_label(
    ground_truth: str | float | None,
    eyeballed_contact: str | float | None,
) -> tuple[BinaryLabel | None, str | None]:
    """
    Prefer explicit ``ground_truth`` when set; otherwise parse ``eyeballed_contact``.
    Returns (binary_label, skip_reason or None).
    """
    gt = _strip_cell(ground_truth).lower()
    if gt:
        if gt in ("legal", "leg", "block"):
            return "legal", None
        if gt in ("goaltend", "goal_tend"):
            return "goaltend", None
        if gt in ("cant_tell", "cant tell", "can't tell"):
            return None, "cant_tell"
        parsed = eyeballed_to_binary(gt)
        if parsed[0] is not None:
            return parsed
        return None, "bad_ground_truth"

    return eyeballed_to_binary(_strip_cell(eyeballed_contact))


def load_labels_csv(path: Path | str) -> pd.DataFrame:
    """Read ``close_calls_labels.csv`` with flexible column names."""
    path = Path(path)
    df = _normalize_header(pd.read_csv(path))
    if "filename" not in df.columns:
        raise ValueError(f"{path} must contain a filename column")
    if "eyeballed_contact" not in df.columns:
        df["eyeballed_contact"] = ""
    if "ground_truth" not in df.columns:
        df["ground_truth"] = ""
    df["filename"] = df["filename"].astype(str).str.strip()
    return df


def load_usable_close_call_binary_labels(
    labels_path: Path | str,
    close_calls_dir: Path | str,
) -> pd.DataFrame:
    """
    Rows with binary legal/goaltend labels and an existing CSV under ``close_calls_dir``.

    Columns: filename, y (legal|goaltend), ground_truth_raw, eyeballed_contact, skip_reason (NaN if usable).
    """
    root = Path(close_calls_dir)
    lab = load_labels_csv(labels_path)
    rows = []
    for _, r in lab.iterrows():
        fn = r["filename"]
        if not fn.endswith(".csv"):
            fn = f"{fn}.csv"
        p = root / fn
        y_bin, skip = row_to_binary_label(r.get("ground_truth"), r.get("eyeballed_contact"))
        rows.append(
            {
                "filename": fn,
                "path": str(p),
                "y": y_bin,
                "ground_truth_raw": _strip_cell(r.get("ground_truth")),
                "eyeballed_contact": _strip_cell(r.get("eyeballed_contact")),
                "skip_reason": skip if y_bin is None else None,
                "file_exists": p.is_file(),
            }
        )
    out = pd.DataFrame(rows)
    usable = out["y"].notna() & out["file_exists"]
    return out.loc[usable].reset_index(drop=True)
