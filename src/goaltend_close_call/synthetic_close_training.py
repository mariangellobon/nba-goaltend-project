"""
Synthetic training examples for the close-call pipeline.

All synthetic rows are generated **only** from labeled *Segmented* CSVs (same pool as
``build_base_binary``). Real close-call CSVs are augmented only via segmented sources,
not via held-out close-call evaluation clips (CV trains on a subset of labeled close
calls per fold; synthetic rows do not copy raw close-call files).

Augmentations (sensor 1 only when ``sensor_1_only``):
- Random choice: standard peak-centered crop vs **jittered** crop (window center shifted
  near the peak) to mimic timing / alignment uncertainty on marginal calls.
- Optional small **Gaussian noise** on acceleration (scale relative to ||a1|| std in-window).
- Optional small **random 3D rotation** of each sample vector (mounting / frame drift).

Tune counts via ``SYNTHETIC_AUGMENTS_PER_SOURCE_FILE`` and flags in ``close_call_model``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .sensor_io import crop_peak_window, discover_segmented_folders, estimate_fs, load_recording_csv

SYNTHETIC_AUGMENTS_PER_SOURCE_FILE = 4
SYNTHETIC_NOISE_SCALE = 0.028
SYNTHETIC_JITTER_MAX_FRAC = 0.14
SYNTHETIC_ROTATION_PROB = 0.28
SYNTHETIC_ROTATION_MAX_DEG = 10.0
SYNTHETIC_STANDARD_CROP_PROB = 0.55


def iter_segmented_labeled_paths(data_root: str | Path) -> Iterator[tuple[Path, str]]:
    """(csv_path, y) with y in {\"goaltend\", \"legal\"}."""
    root = Path(data_root)
    for folder, label in discover_segmented_folders(root):
        if folder.name == "Other Data - Segmented":
            continue
        y = "goaltend" if label == "goaltends" else "legal"
        for csv_path in sorted(folder.glob("*.csv")):
            yield csv_path, y


def crop_jittered_peak_window(
    t: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    *,
    win_sec: float,
    fs: float,
    sensor_1_only: bool,
    rng: np.random.Generator,
    jitter_max_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same length window as ``crop_peak_window`` but center jittered around the magnitude peak."""
    n = len(t)
    win_samples = int(round(win_sec * fs))
    if win_samples >= n or win_sec <= 0:
        return t, a1, a2
    if sensor_1_only:
        mag = np.linalg.norm(a1, axis=1)
    else:
        mag = np.linalg.norm(a1, axis=1) + np.linalg.norm(a2, axis=1)
    peak = int(np.argmax(mag))
    j = max(1, int(round(jitter_max_frac * win_samples)))
    shift = int(rng.integers(-j, j + 1))
    center = int(np.clip(peak + shift, 0, n - 1))
    half = win_samples // 2
    start = max(0, center - half)
    end = start + win_samples
    if end > n:
        end = n
        start = max(0, end - win_samples)
    sl = slice(start, end)
    return t[sl], a1[sl], a2[sl]


def _add_acc_noise(
    a1: np.ndarray,
    a2: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float,
    sensor_1_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    ref = float(np.std(np.linalg.norm(a1, axis=1)) + 1e-9)
    sigma = noise_scale * ref
    a1n = a1 + rng.normal(0.0, sigma, size=a1.shape).astype(np.float64)
    if sensor_1_only:
        return a1n, a2
    a2n = a2 + rng.normal(0.0, sigma, size=a2.shape).astype(np.float64)
    return a1n, a2n


def _rotate_accel_small(
    a1: np.ndarray,
    a2: np.ndarray,
    rng: np.random.Generator,
    max_deg: float,
    sensor_1_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    ang = rng.uniform(-max_deg, max_deg)
    r = Rotation.from_rotvec(np.deg2rad(ang) * axis)
    a1r = r.apply(a1)
    if sensor_1_only:
        return a1r.astype(np.float64), a2
    a2r = r.apply(a2)
    return a1r.astype(np.float64), a2r.astype(np.float64)


def augment_one_window(
    t: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    *,
    fs: float,
    win_sec: float,
    sensor_1_only: bool,
    rng: np.random.Generator,
    jitter_max_frac: float,
    noise_scale: float,
    rotation_prob: float,
    rotation_max_deg: float,
    standard_crop_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng.random() < standard_crop_prob:
        t, a1, a2 = crop_peak_window(
            t, a1, a2, win_sec=win_sec, fs=fs, sensor_1_only=sensor_1_only
        )
    else:
        t, a1, a2 = crop_jittered_peak_window(
            t,
            a1,
            a2,
            win_sec=win_sec,
            fs=fs,
            sensor_1_only=sensor_1_only,
            rng=rng,
            jitter_max_frac=jitter_max_frac,
        )
    if noise_scale > 0:
        a1, a2 = _add_acc_noise(a1, a2, rng, noise_scale, sensor_1_only)
    if rotation_prob > 0 and rng.random() < rotation_prob:
        a1, a2 = _rotate_accel_small(
            a1, a2, rng, rotation_max_deg, sensor_1_only=sensor_1_only
        )
    return t, a1, a2


def build_synthetic_feature_rows(
    data_root: str | Path,
    extract_fn,
    *,
    augments_per_file: int = SYNTHETIC_AUGMENTS_PER_SOURCE_FILE,
    win_sec: float,
    sensor_1_only: bool,
    rng: np.random.Generator | None = None,
    jitter_max_frac: float = SYNTHETIC_JITTER_MAX_FRAC,
    noise_scale: float = SYNTHETIC_NOISE_SCALE,
    rotation_prob: float = SYNTHETIC_ROTATION_PROB,
    rotation_max_deg: float = SYNTHETIC_ROTATION_MAX_DEG,
    standard_crop_prob: float = SYNTHETIC_STANDARD_CROP_PROB,
) -> pd.DataFrame:
    """
    ``extract_fn(t, a1, a2, fs)`` must return a feature dict (no label column).
    """
    rng = rng or np.random.default_rng(42)
    rows: list[dict] = []
    for csv_path, y in iter_segmented_labeled_paths(data_root):
        for _ in range(augments_per_file):
            t, a1, a2 = load_recording_csv(csv_path, sensor_1_only=sensor_1_only)
            fs = estimate_fs(t)
            t, a1, a2 = augment_one_window(
                t,
                a1,
                a2,
                fs=fs,
                win_sec=win_sec,
                sensor_1_only=sensor_1_only,
                rng=rng,
                jitter_max_frac=jitter_max_frac,
                noise_scale=noise_scale,
                rotation_prob=rotation_prob,
                rotation_max_deg=rotation_max_deg,
                standard_crop_prob=standard_crop_prob,
            )
            feat = extract_fn(t, a1, a2, fs)
            feat["y"] = y
            rows.append(feat)
    return pd.DataFrame(rows)
