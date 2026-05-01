"""
Time-domain shape features for acceleration windows (one or two sensors).

No frequency-domain analysis (no STFT/FFT/Welch). Uses normalized magnitude
envelopes for peak structure (scale of the hit within the window is factored out)
and direction-based summaries so the model is not a raw "how hard" classifier.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def _unit_rows(a: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, nrm, out=np.zeros_like(a), where=nrm > 1e-12)


def _peak_structure(trace: np.ndarray) -> dict[str, float]:
    """
    trace: nonnegative, peak-normalized to max 1 (shape of envelope).
    """
    x = np.asarray(trace, dtype=np.float64).ravel()
    n = len(x)
    if n < 5:
        return {
            "n_peaks": 0.0,
            "sec_to_pri_ratio": 0.0,
            "third_to_pri_ratio": 0.0,
            "top2_peak_time_sep": 0.0,
            "first_peak_rel_pos": 0.5,
            "double_peak_prom": 0.0,
        }
    rng = float(np.max(x) - np.min(x))
    prom = max(0.04 * max(rng, 1e-6), 0.02)
    dist = max(3, n // 40)
    peaks, props = find_peaks(x, prominence=prom, height=0.12, distance=dist)
    heights = props.get("prominences", np.zeros(0))
    if len(peaks) == 0:
        p0 = int(np.argmax(x))
        return {
            "n_peaks": 1.0,
            "sec_to_pri_ratio": 0.0,
            "third_to_pri_ratio": 0.0,
            "top2_peak_time_sep": 0.0,
            "first_peak_rel_pos": float(p0 / max(n - 1, 1)),
            "double_peak_prom": 0.0,
        }
    ph = x[peaks]
    order = np.argsort(-ph)
    sorted_peaks = peaks[order]
    sorted_h = ph[order]
    pri = float(sorted_h[0])
    sec_r = float(sorted_h[1] / (pri + 1e-12)) if len(sorted_h) > 1 else 0.0
    third_r = float(sorted_h[2] / (pri + 1e-12)) if len(sorted_h) > 2 else 0.0
    if len(sorted_peaks) >= 2:
        i1, i2 = sorted((int(sorted_peaks[0]), int(sorted_peaks[1])))
        sep = abs(i2 - i1) / max(n - 1, 1)
    else:
        sep = 0.0
    first_peak_rel_pos = float(sorted_peaks[0] / max(n - 1, 1))
    proms = props.get("prominences", np.array([0.0]))
    if len(proms) >= 2:
        sp = np.sort(proms)[::-1]
        double_peak_prom = float(sp[1] / (sp[0] + 1e-12))
    else:
        double_peak_prom = 0.0
    return {
        "n_peaks": float(len(peaks)),
        "sec_to_pri_ratio": sec_r,
        "third_to_pri_ratio": third_r,
        "top2_peak_time_sep": sep,
        "first_peak_rel_pos": first_peak_rel_pos,
        "double_peak_prom": double_peak_prom,
    }


def _xcorr_max_zero_mean(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, float]:
    """Max normalized Pearson corr over integer lags; no FFT."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(len(a), len(b))
    if n < 8:
        return 0.0, 0.0
    a = a[:n] - np.mean(a[:n])
    b = b[:n] - np.mean(b[:n])
    sa = np.std(a)
    sb = np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0, 0.0
    a = a / sa
    b = b / sb
    ml = min(max_lag, n // 4)
    best_r = -2.0
    best_lag = 0.0
    for lag in range(-ml, ml + 1):
        if lag == 0:
            seg_a, seg_b = a, b
        elif lag > 0:
            seg_a, seg_b = a[lag:], b[:-lag]
        else:
            k = -lag
            seg_a, seg_b = a[:-k], b[k:]
        m = len(seg_a)
        if m < 5:
            continue
        r = float(np.dot(seg_a, seg_b) / m)
        if r > best_r:
            best_r = r
            best_lag = float(lag)
    return best_r if best_r > -2 else 0.0, best_lag / max(ml, 1)


def _half_symmetry(trace: np.ndarray) -> float:
    """1 - normalized L1 diff between first and second half (0=asymmetric, 1=symmetric)."""
    x = np.asarray(trace, dtype=np.float64).ravel()
    n = len(x)
    if n < 10:
        return 0.0
    h = n // 2
    a, b = x[:h], x[n - h :]
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    d = float(np.mean(np.abs(a - b[::-1])))
    scale = float(np.mean(np.abs(x)) + 1e-9)
    return float(np.clip(1.0 - d / scale, 0.0, 1.0))


def extract_shape_features(
    t: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    fs: float | None = None,
    *,
    sensor_1_only: bool = False,
) -> dict[str, float]:
    """
    Shape-only summaries on normalized magnitude envelopes and direction traces.
    When ``sensor_1_only``, only physical sensor 1 (``a1``) is used; ``a2`` is ignored.
    ``fs`` is used for max xcorr lag when two sensors are used.
    """
    if fs is None:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 600.0
        fs = 1.0 / max(dt, 1e-9)
    max_lag = int(round(0.05 * fs))
    max_lag = max(max_lag, 5)

    m1 = np.linalg.norm(a1, axis=1)

    def norm_peak(m: np.ndarray) -> np.ndarray:
        mx = float(np.max(m))
        if mx < 1e-12:
            return np.zeros_like(m, dtype=np.float64)
        return (m / mx).astype(np.float64)

    m1n = norm_peak(m1)
    u1 = _unit_rows(a1)
    d1 = np.linalg.norm(np.diff(u1, axis=0), axis=1)

    feats: dict[str, float] = {}
    for k, v in _peak_structure(m1n).items():
        feats[f"mag1_{k}"] = v
    feats["mag_half_sym_mag1"] = _half_symmetry(m1n)
    feats["mean_dirchg_rate_s1"] = float(np.mean(d1)) if len(d1) else 0.0
    feats["std_dirchg_rate_s1"] = float(np.std(d1)) if len(d1) > 1 else 0.0
    nu = max(len(u1) - 1, 1)
    feats["path_len_dir_u1"] = float(np.sum(d1) / nu) if len(d1) else 0.0

    if len(m1n) > 2:
        dd = np.diff(m1n.astype(np.float64))
        d2 = np.diff(dd)
        feats["mag1_d2_mean"] = float(np.mean(np.abs(d2)))
    else:
        feats["mag1_d2_mean"] = 0.0

    if sensor_1_only:
        return {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in feats.items()}

    m2 = np.linalg.norm(a2, axis=1)
    msum = m1 + m2
    m2n = norm_peak(m2)
    mxsum = float(np.max(msum))
    msumn = (msum / mxsum) if mxsum > 1e-12 else np.zeros_like(msum)

    for prefix, tr in [("mag2", m2n), ("magsum", msumn)]:
        for k, v in _peak_structure(tr).items():
            feats[f"{prefix}_{k}"] = v
    feats["mag_half_sym_magsum"] = _half_symmetry(msumn)

    u2 = _unit_rows(a2)
    dots = np.sum(u1 * u2, axis=1)
    feats["mean_dot_u1_u2"] = float(np.mean(dots)) if len(dots) else 0.0
    feats["std_dot_u1_u2"] = float(np.std(dots)) if len(dots) > 1 else 0.0

    d2 = np.linalg.norm(np.diff(u2, axis=0), axis=1)
    k = min(len(d1), len(d2))
    if k >= 4 and np.std(d1[:k]) > 1e-12 and np.std(d2[:k]) > 1e-12:
        feats["corr_dir_rate_s1_s2"] = float(np.corrcoef(d1[:k], d2[:k])[0, 1])
    else:
        feats["corr_dir_rate_s1_s2"] = 0.0

    r, lag_n = _xcorr_max_zero_mean(m1, m2, max_lag=max_lag)
    feats["env_xcorr_max"] = r
    feats["env_xcorr_best_lag_norm"] = lag_n

    if len(msumn) > 2:
        d = np.diff(msumn.astype(np.float64))
        d2m = np.diff(d)
        feats["magsum_d2_mean"] = float(np.mean(np.abs(d2m)))
    else:
        feats["magsum_d2_mean"] = 0.0

    return {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in feats.items()}
