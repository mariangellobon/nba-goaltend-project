"""
TabPFN error and confidence analysis — tables + IMU trace plots.

Runs the same ROCKET + TabPFN protocol as ``goaltend_classify`` (LOO or holdout test set),
then filters rows and saves PNGs plus CSV summaries under ``GOALTEND_OUTPUT_DIR``.

Examples
--------
All misclassified LOO samples (plots + summary CSV)::

    python -m goaltend_tabpfn.tabpfn_analysis --split loo --view wrong

Wrong predictions where the model assigned ≥0.75 probability to its (wrong) class::

    python -m goaltend_tabpfn.tabpfn_analysis --split loo --view confident_wrong --threshold 0.75

Samples with low max-class probability (ambiguous); add ``--wrong-only`` to restrict to errors::

    python -m goaltend_tabpfn.tabpfn_analysis --split loo --view low_confidence --threshold 0.75

Holdout test split (same random seed as classify holdout by default)::

    python -m goaltend_tabpfn.tabpfn_analysis --split holdout --view confident_wrong
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from goaltend_close_call.paths import outputs_dir

from goaltend_tabpfn.goaltend_classify import (
    ACCEL_COLS,
    DATA_DIR,
    N_GROUPS,
    N_KERNELS,
    holdout_predictions_table,
    loo_predictions_table,
)


TIME_COL = "Latest: Time (s)"


def _pick_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def save_trace_figure(
    csv_path: Path,
    *,
    y_true: str,
    y_pred: str,
    p_legal: float,
    p_goaltend: float,
    p_predicted: float,
    out_path: Path,
    subtitle: str = "",
) -> None:
    """Save a 6-panel accelerometer plot (non-interactive)."""
    df = pd.read_csv(csv_path, usecols=[TIME_COL] + ACCEL_COLS)
    t = df[TIME_COL].values - df[TIME_COL].values[0]

    short_names = ["X1", "Y1", "Z1", "Z2", "Y2", "X2"]
    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    for ax, col, name, color in zip(axes, ACCEL_COLS, short_names, colors):
        ax.plot(t, df[col].values, color=color, linewidth=0.8)
        ax.set_ylabel(f"{name}\n(m/s²)", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")

    mark = "[match]" if y_pred == y_true else "[mismatch]"
    title = (
        f"{csv_path.name}\n"
        f"true={y_true}   pred={y_pred}  {mark}\n"
        f"p(legal)={p_legal:.3f}  p(goaltend)={p_goaltend:.3f}  "
        f"p(predicted class)={p_predicted:.3f}"
    )
    if subtitle:
        title = subtitle + "\n" + title
    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def filter_rows(
    df: pd.DataFrame,
    view: str,
    *,
    threshold: float,
    wrong_only: bool,
) -> pd.DataFrame:
    if view == "wrong":
        return df.loc[~df["correct"]].copy()
    if view == "confident_wrong":
        return df.loc[(~df["correct"]) & (df["p_predicted"] >= threshold)].copy()
    if view == "low_confidence":
        sub = df.loc[df["p_predicted"] < threshold].copy()
        if wrong_only:
            sub = sub.loc[~sub["correct"]].copy()
        return sub
    raise ValueError(f"Unknown view: {view!r}")


def run(
    *,
    split: str,
    view: str,
    threshold: float,
    wrong_only: bool,
    out_dir: Path | None,
    save_plots: bool,
    save_full_csv: bool,
    n_kernels: int,
    n_groups: int,
    test_size: float,
    random_state: int,
    data_dir: Path,
) -> Path:
    device = _pick_device()
    print(f"Device: {device.upper()}")

    if split == "loo":
        df = loo_predictions_table(
            data_dir=data_dir,
            n_groups=n_groups,
            n_kernels=n_kernels,
            device=device,
        )
        meta = {"split": "loo", "n_rows": len(df)}
    else:
        df, meta = holdout_predictions_table(
            data_dir=data_dir,
            n_groups=n_groups,
            n_kernels=n_kernels,
            test_size=test_size,
            random_state=random_state,
            device=device,
        )
        meta["split"] = "holdout"

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = out_dir or (outputs_dir() / "tabpfn_analysis" / f"{view}_{split}_{ts}")
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)

    if save_full_csv:
        df.to_csv(base / "full_predictions.csv", index=False)

    filtered = filter_rows(df, view, threshold=threshold, wrong_only=wrong_only)
    filtered.to_csv(base / "filtered.csv", index=False)

    lines = [
        f"split={split}  view={view}  threshold={threshold}  wrong_only={wrong_only}",
        f"n_full={len(df)}  n_filtered={len(filtered)}",
        str(meta),
        "",
    ]
    for _, row in filtered.iterrows():
        lines.append(
            f"  {row['filename']:<48}  true={row['y_true']:<10} pred={row['y_pred']:<10}  "
            f"p_pred={row['p_predicted']:.3f}  p_gt={row['p_goaltend']:.3f}  p_leg={row['p_legal']:.3f}"
        )
    (base / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {base / 'filtered.csv'} ({len(filtered)} rows)")
    print(f"Summary → {base / 'summary.txt'}")

    if save_plots and len(filtered) == 0:
        print("No rows to plot.")
    elif save_plots:
        for plot_i, (_, row) in enumerate(filtered.iterrows()):
            stem = Path(row["filename"]).stem
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:80]
            fname = f"{plot_i:04d}_{safe}.png"
            p_leg = float(row["p_legal"]) if pd.notna(row["p_legal"]) else float("nan")
            save_trace_figure(
                Path(row["path"]),
                y_true=str(row["y_true"]),
                y_pred=str(row["y_pred"]),
                p_legal=p_leg,
                p_goaltend=float(row["p_goaltend"]),
                p_predicted=float(row["p_predicted"]),
                out_path=base / "figures" / fname,
                subtitle=f"{view}  ({split})",
            )
        print(f"Figures → {base / 'figures'}")

    return base


def main() -> None:
    p = argparse.ArgumentParser(description="TabPFN wrong / confidence analysis + plots")
    p.add_argument("--split", choices=["loo", "holdout"], default="loo")
    p.add_argument(
        "--view",
        choices=["wrong", "confident_wrong", "low_confidence"],
        required=True,
        help="wrong: all errors; confident_wrong: errors with p(predicted)≥threshold; "
        "low_confidence: max-class prob < threshold (see --wrong-only)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="For confident_wrong: min p(predicted). For low_confidence: max p(predicted) strictly below this.",
    )
    p.add_argument(
        "--wrong-only",
        action="store_true",
        help="With low_confidence: keep only misclassified rows.",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Output folder (default under outputs/tabpfn_analysis/)")
    p.add_argument("--no-plots", action="store_true", help="Only write CSV + summary text")
    p.add_argument("--no-full-csv", action="store_true", help="Skip writing full (unfiltered) prediction table")
    p.add_argument("--n-kernels", type=int, default=N_KERNELS)
    p.add_argument("--n-groups", type=int, default=N_GROUPS)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = p.parse_args()

    run(
        split=args.split,
        view=args.view,
        threshold=args.threshold,
        wrong_only=args.wrong_only,
        out_dir=args.out_dir,
        save_plots=not args.no_plots,
        save_full_csv=not args.no_full_csv,
        n_kernels=args.n_kernels,
        n_groups=args.n_groups,
        test_size=args.test_size,
        random_state=args.random_state,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
