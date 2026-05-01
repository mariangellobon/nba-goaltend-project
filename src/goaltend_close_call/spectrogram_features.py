"""Spectrogram-based features that avoid raw acceleration magnitude (force) so light vs hard contact is not a dominant cue."""

from __future__ import annotations

import numpy as np
from scipy import signal


def _stft_power(
    x: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns f, t_stft, Sxx power (not dB)."""
    if noverlap is None:
        noverlap = nperseg * 3 // 4
    f, t_stft, Zxx = signal.stft(
        x,
        fs=fs,
        nperseg=min(nperseg, len(x) // 2 or 1),
        noverlap=min(noverlap, max(0, (len(x) // 2 or 1) - 1)),
        boundary="zeros",
    )
    Sxx = np.abs(Zxx) ** 2
    return f, t_stft, Sxx


def spectral_shape_features(f: np.ndarray, Sxx: np.ndarray) -> dict[str, float]:
    """Aggregate power spectrum over time: centroid, spread, low/mid/high energy fractions."""
    p = np.mean(Sxx, axis=1) + 1e-20
    p = p / p.sum()
    centroid = float(np.sum(f * p))
    spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * p)))
    nyq = f[-1]
    low = float(np.sum(p[f <= 0.05 * nyq]))
    mid = float(np.sum(p[(f > 0.05 * nyq) & (f <= 0.35 * nyq)]))
    high = float(np.sum(p[f > 0.35 * nyq]))
    rolloff_bins = np.where(np.cumsum(p) >= 0.85)[0]
    rolloff = float(f[rolloff_bins[0]]) if len(rolloff_bins) else float(f[-1])
    peak_idx = int(np.argmax(p))
    peak_hz = float(f[peak_idx])
    return {
        "spec_centroid_hz": centroid,
        "spec_spread_hz": spread,
        "spec_low_frac": low,
        "spec_mid_frac": mid,
        "spec_high_frac": high,
        "spec_rolloff_hz": rolloff,
        "spec_peak_hz": peak_hz,
    }


def _unit_direction_rows(a: np.ndarray) -> np.ndarray:
    """Per-sample acceleration direction (unit vectors); invariant to scaling of ||a||."""
    nrm = np.linalg.norm(a, axis=1, keepdims=True)
    out = np.divide(a, nrm, out=np.zeros_like(a), where=nrm > 1e-12)
    return out


def _direction_change_series(u: np.ndarray) -> np.ndarray:
    """
    Euclidean norm of successive differences of direction vectors.
    Scale-invariant w.r.t. original acceleration magnitude; captures how fast
    the force vector direction changes (related to angular motion of the vector).
    """
    if len(u) < 2:
        return np.array([0.0], dtype=np.float64)
    du = np.diff(u, axis=0)
    return np.linalg.norm(du, axis=1)


def scale_free_time_features(a1: np.ndarray, a2: np.ndarray) -> dict[str, float]:
    """Correlation between direction-change traces only (no RMS / jerk of raw acc)."""
    u1 = _unit_direction_rows(a1)
    u2 = _unit_direction_rows(a2)
    d1 = _direction_change_series(u1)
    d2 = _direction_change_series(u2)
    n = min(len(d1), len(d2))
    if n < 3:
        corr = 0.0
    else:
        d1 = d1[:n]
        d2 = d2[:n]
        if np.std(d1) < 1e-12 or np.std(d2) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(d1, d2)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
    return {"dirchg_corr_s1_s2": corr}


def extract_features(
    t: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    fs: float | None = None,
    nperseg: int = 256,
    *,
    sensor_1_only: bool = False,
) -> dict[str, float]:
    """
    Spectral shape on **direction-change** scalars (from unit-normalized acceleration),
    plus (when not ``sensor_1_only``) correlation of change-rate traces between sensors.
    Does not use RMS, raw jerk, or STFT on ||a||, so overall impact strength (light vs hard)
    is not encoded as a direct amplitude feature.
    """
    if fs is None:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 600.0
        fs = 1.0 / max(dt, 1e-9)

    u1 = _unit_direction_rows(a1)
    s1 = _direction_change_series(u1)

    feats: dict[str, float] = {}
    if sensor_1_only:
        series_list = [("dirchg_s1", s1)]
    else:
        u2 = _unit_direction_rows(a2)
        s2 = _direction_change_series(u2)
        ssum = s1[: min(len(s1), len(s2))] + s2[: min(len(s1), len(s2))]
        feats.update(scale_free_time_features(a1, a2))
        series_list = [("dirchg_s1", s1), ("dirchg_s2", s2), ("dirchg_sum", ssum)]

    for name, sig in series_list:
        if len(sig) < 2:
            sig = np.array([0.0, 0.0], dtype=np.float64)
        sig_z = sig - np.mean(sig)
        f, _, Sxx = _stft_power(sig_z, fs, nperseg=nperseg)
        for k, v in spectral_shape_features(f, Sxx).items():
            feats[f"{name}_{k}"] = v

    return {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in feats.items()}
