"""
Load backboard accelerometer CSVs: two-sensor format (sensor2 axes Z,Y,X in file order).

**Default (legal classes, etc.):** returns tri-axial data for **physical sensors 1 and 2**
as ``(a1, a2)``.

**Goaltends folder:** returns **physical sensors 1 and 3** as ``(a1, a2)`` so the pipeline
still sees two triples; only the mount selection changes.

Axes are always aligned to XYZ via column names.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

AXIS_ORDER = {"X": 0, "Y": 1, "Z": 2}
COL_PAT = re.compile(r"Latest:\s*([XYZ])\s*Acceleration\s*(\d+)", re.I)
# Some exports omit the sensor index on one axis, e.g. "Latest: Y Acceleration (m/s²)" for sensor 1.
COL_PAT_BARE = re.compile(r"Latest:\s*([XYZ])\s*Acceleration\s*\(m/s", re.I)


def _latest_accel_columns(df: pd.DataFrame) -> dict[tuple[int, int], str]:
    """Map (sensor_index_0based, axis_0based) -> column name."""
    mapping: dict[tuple[int, int], str] = {}
    cols = [str(c).strip() for c in df.columns]
    for c in cols:
        m = COL_PAT.match(c)
        if not m:
            continue
        axis = AXIS_ORDER[m.group(1).upper()]
        sens = int(m.group(2)) - 1
        mapping[(sens, axis)] = c
    for c in cols:
        if COL_PAT.match(c):
            continue
        mb = COL_PAT_BARE.match(c)
        if mb:
            axis = AXIS_ORDER[mb.group(1).upper()]
            if (0, axis) not in mapping:
                mapping[(0, axis)] = c
    return mapping


def _stack_sensor_xyz(m: dict[tuple[int, int], str], df: pd.DataFrame, sens_0based: int) -> np.ndarray:
    """Build Nx3 array for one physical sensor (0-based index: 0=sensor1, 1=sensor2, 2=sensor3)."""
    if (sens_0based, 0) not in m or (sens_0based, 1) not in m or (sens_0based, 2) not in m:
        raise ValueError(f"Sensor {sens_0based + 1} XYZ not found")
    return np.column_stack([df[m[sens_0based, ax]].values.astype(np.float64) for ax in range(3)])


def _goaltend_folder_uses_sensors_13(path: Path) -> bool:
    """Goaltend exports use physical sensors 1+3 as the two channels for modeling."""
    return "goaltend" in path.parent.name.lower()


def load_recording_csv(
    path: str | Path, *, sensor_1_only: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (time_s, a1_Nx3, a2_Nx3) in XYZ column order.

    When ``sensor_1_only`` is True, only **physical sensor 1** is read into ``a1``;
    ``a2`` is a zero array of the same shape (callers should pass ``sensor_1_only``
    into ``crop_peak_window`` / feature extractors so ``a2`` is ignored).

    Otherwise: for files under a *goaltends* folder, ``a1``/``a2`` are **physical sensors 1 and 3**.
    Otherwise they are **physical sensors 1 and 2**.

    Uses Latest accelerometer columns only (ignores FFT columns).
    """
    path = Path(path)
    df = pd.read_csv(path)
    time_cols = [
        c
        for c in df.columns
        if "time" in str(c).lower() and "fft" not in str(c).lower() and "frequency" not in str(c).lower()
    ]
    if not time_cols:
        raise ValueError(f"No time column in {path}")
    t = df[time_cols[0]].values.astype(np.float64)
    m = _latest_accel_columns(df)

    if sensor_1_only:
        if (0, 0) not in m or (0, 1) not in m or (0, 2) not in m:
            raise ValueError(f"Sensor 1 XYZ not found in {path}")
        a1 = _stack_sensor_xyz(m, df, 0)
        a2 = np.zeros_like(a1)
        return t, a1, a2

    if _goaltend_folder_uses_sensors_13(path):
        try:
            a1 = _stack_sensor_xyz(m, df, 0)
            a2 = _stack_sensor_xyz(m, df, 2)
        except ValueError as e:
            raise ValueError(
                f"Goaltend file needs Latest X/Y/Z Acceleration 1 and 3: {path}"
            ) from e
    else:
        if (0, 0) not in m or (0, 1) not in m or (0, 2) not in m:
            raise ValueError(f"Sensor 1 XYZ not found in {path}: {list(df.columns)[:8]}")
        if (1, 0) not in m or (1, 1) not in m or (1, 2) not in m:
            raise ValueError(f"Sensor 2 XYZ not found in {path}")
        a1 = _stack_sensor_xyz(m, df, 0)
        a2 = _stack_sensor_xyz(m, df, 1)

    return t, a1, a2


def estimate_fs(t: np.ndarray) -> float:
    if len(t) < 2:
        return 600.0
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        dt = (t[-1] - t[0]) / max(len(t) - 1, 1)
    return 1.0 / dt


def crop_peak_window(
    t: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    win_sec: float,
    fs: Optional[float] = None,
    *,
    sensor_1_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Center a fixed-duration window on the sample where vector magnitude is maximal.
    If ``sensor_1_only``, uses ``||a1||`` only; otherwise ``||a1|| + ||a2||``.
    If the recording is shorter than win_sec, returns the full recording unchanged.
    """
    n = len(t)
    if n < 2:
        return t, a1, a2
    if fs is None:
        fs = estimate_fs(t)
    if sensor_1_only:
        mag = np.linalg.norm(a1, axis=1)
    else:
        mag = np.linalg.norm(a1, axis=1) + np.linalg.norm(a2, axis=1)
    win_samples = int(round(win_sec * fs))
    if win_samples >= n or win_sec <= 0:
        return t, a1, a2
    peak = int(np.argmax(mag))
    half = win_samples // 2
    start = max(0, peak - half)
    end = start + win_samples
    if end > n:
        end = n
        start = max(0, end - win_samples)
    sl = slice(start, end)
    return t[sl], a1[sl], a2[sl]


def discover_segmented_folders(data_root: str | Path) -> list[tuple[Path, str]]:
    """Return (folder_path, canonical_label) for each *Segmented folder."""
    root = Path(data_root)
    out: list[tuple[Path, str]] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or "Segmented" not in p.name:
            continue
        name = p.name.replace(" - Segmented", "").strip()
        key = name.lower().replace(" ", "_")
        out.append((p, key))
    return out


if __name__ == "__main__":
    import sys

    from .paths import data_root

    root = data_root()
    legal = next((root / "Blocks - Segmented").glob("*.csv"), None)
    goal = next((root / "Goaltends - Segmented").glob("*.csv"), None)
    if legal is None or goal is None:
        print("Could not find sample CSVs under data/Blocks or data/Goaltends.", file=sys.stderr)
        sys.exit(1)
    t1, a1, a2 = load_recording_csv(legal)
    print(f"Legal sample: {legal.name}")
    print(f"  shape a1,a2: {a1.shape}, {a2.shape}  (physical sensors 1 and 2)")
    tg, g1, g2 = load_recording_csv(goal)
    print(f"Goaltend sample: {goal.name}")
    print(f"  shape g1,g2: {g1.shape}, {g2.shape}  (physical sensors 1 and 3)")
    print("sensor_io OK")
