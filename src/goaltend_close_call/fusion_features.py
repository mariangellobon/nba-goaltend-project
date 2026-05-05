"""Combine spectrogram-on-direction-change features with time-domain shape features."""

from __future__ import annotations

from .shape_time_features import extract_shape_features
from .spectrogram_features import extract_features


def extract_fusion_features(
    t,
    a1,
    a2,
    fs: float | None = None,
    nperseg: int = 256,
    *,
    sensor_1_only: bool = False,
) -> dict[str, float]:
    """
    Spectrogram summaries on direction-change series plus prefixed ``shape_*`` envelopes /
    cross-sensor cues from ``extract_shape_features``.
    """
    spec = extract_features(
        t, a1, a2, fs=fs, nperseg=nperseg, sensor_1_only=sensor_1_only
    )
    shp = extract_shape_features(
        t, a1, a2, fs=fs, sensor_1_only=sensor_1_only
    )
    out = dict(spec)
    for k, v in shp.items():
        out[f"shape_{k}"] = float(v)
    return out
