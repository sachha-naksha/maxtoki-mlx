"""Aggregated Δt bar plot across the 6 PDK4 / IRS2 / NR4A3 inhibit + overexpress
8k evenly-spaced YM2-context runs. Unlike combine_by_donor_fig.py, this pools
all OM6 + OM9 cells together — one bar per (gene, direction) — to read out
the perturbation effect on mean Δt without sample-specific splitting.
Output: editable SVG + PNG + CSV.
"""
from __future__ import annotations

from pathlib import Path

import datasets  # noqa: F401  (used implicitly via load_from_disk)
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

plt.rcParams.update({
    "text.usetex": False,
    "svg.fonttype": "none",
})

FIGURE_PARAMS = {
    "dpi": 300,
    "bbox_inches": "tight",
    "format": "svg",
    "transparent": True,
}

OUT_ROOT = Path("/workspaces/maxToki/out")

RUNS = [
    ("PDK4 i",   "pdk4_217m_inhibit_evenly_seq8k",      "inhibit"),
    ("PDK4 oe",  "pdk4_217m_overexpress_evenly_seq8k",  "overexpress"),
    ("IRS2 i",   "irs2_217m_inhibit_evenly_seq8k",      "inhibit"),
    ("IRS2 oe",  "irs2_217m_overexpress_evenly_seq8k",  "overexpress"),
    ("NR4A3 i",  "nr4a3_217m_inhibit_evenly_seq8k",     "inhibit"),
    ("NR4A3 oe", "nr4a3_217m_overexpress_evenly_seq8k", "overexpress"),
]

DONORS = {"OM6", "OM9"}
COLORS = {"inhibit": "#c0392b", "overexpress": "#2c7fb8"}


def aggregated_stats(run_dir: Path) -> tuple[float, float, int]:
    z = np.load(run_dir / "scores.npz", allow_pickle=True)
    delta_t = np.asarray(z["delta_t"]).ravel()
    ds = load_from_disk(str(run_dir / "baseline.dataset"))
    group = np.array([str(r["group"]) for r in ds])
    sel = np.isin(group, list(DONORS))
    vals = delta_t[sel].astype(float)
    n = int(sel.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(vals.mean())
    sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem, n


def main() -> None:
    rows = []
    for label, run_subdir, direction in RUNS:
        mean, sem, n = aggregated_stats(OUT_ROOT / run_subdir)
        rows.append((label, mean, sem, n, direction))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bar_h = 0.55

    ys = list(range(len(RUNS)))
    ys = list(reversed(ys))

    yticks: list[float] = []
    ylabels: list[str] = []
    for y, (label, mean, sem, n, direction) in zip(ys, rows):
        ax.barh(
            y, mean, height=bar_h, color=COLORS[direction],
            edgecolor="black", linewidth=0.4,
            xerr=sem, ecolor="black",
            error_kw={"elinewidth": 0.7, "capsize": 2},
        )
        yticks.append(y)
        ylabels.append(f"{label}  (n={n})")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Mean Δt  (perturbed − baseline pseudotime)")
    ax.set_title(
        "Zero-shot perturbation Δt — 217M, 8k seq, YM2 evenly-spaced ctx\n"
        "OM6 + OM9 cells pooled  (PDK4 / IRS2 / NR4A3)",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["inhibit"], label="inhibit"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["overexpress"], label="overexpress"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    out_svg = OUT_ROOT / "combined_aggregated_8k_evenly_3gene.svg"
    fig.savefig(out_svg, **FIGURE_PARAMS)
    out_png = OUT_ROOT / "combined_aggregated_8k_evenly_3gene.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")

    csv_path = OUT_ROOT / "mean_delta_t_8k_evenly_3gene_aggregated.csv"
    with open(csv_path, "w") as f:
        f.write("gene_label,mean_delta_t,sem,n,direction\n")
        for label, mean, sem, n, direction in rows:
            f.write(f"{label},{mean:.6f},{sem:.6f},{n},{direction}\n")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
