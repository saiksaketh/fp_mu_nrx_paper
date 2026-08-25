#!/usr/bin/env python3
"""Generate IEEE figures: weight-grid match (why FP4 beats INT4) and
per-layer magnitude-pruning sparsity.

Reads campaign checkpoints listed in scripts/run_eval_campaign.py and the
matching magglobal50 prune reports. Writes PDFs + stats.json next to this
script. See README.md for usage.
"""
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(HERE, "figures")
STATS_PATH = os.path.join(HERE, "stats.json")

# Campaign QAT-only pickles (same map as scripts/run_eval_campaign.py).
QAT_WEIGHTS = {
    "float": os.path.join(REPO, "weights/nrx_large_weights"),
    "int4": os.path.join(REPO, "weights/quantized_weights/nrx_large_qat_4bit_pc_weights"),
    "fp4": os.path.join(
        REPO, "weights/quantized_weights/nrx_large_qat_fp4_e2m1_pc_weights"
    ),
}

# Mask reports that produced the campaign pruned checkpoints.
PRUNE_REPORTS = {
    "int4": os.path.join(
        REPO,
        "weights/pruned_weights/nrx_large_qat_4bit_pc_magglobal50/"
        "nrx_large_qat_4bit_pc_magglobal50_report.json",
    ),
    "int8": os.path.join(
        REPO,
        "weights/pruned_weights/nrx_large_qat_8bit_pc_magglobal50/"
        "nrx_large_qat_8bit_pc_magglobal50_report.json",
    ),
    "fp4": os.path.join(
        REPO,
        "weights/pruned_weights/nrx_large_qat_fp4_e2m1_pc_magglobal50/"
        "nrx_large_qat_fp4_e2m1_pc_magglobal50_report.json",
    ),
}

# Same 15-level grids used in utils/quantized_conv2d.py.
FP4_E2M1 = np.array(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
)
INT4_NARROW = np.arange(-7, 8, dtype=float)  # narrow_range=True, 4-bit
FP4_MAX = 6.0
INT4_MAX = 7.0

IEEE_COL = 3.5
IEEE_DOUBLE = 7.16
IEEE_H = 2.85
IEEE_SQUARE = 3.5

COLORS = {
    "fp32": "#000000",
    "int4": "#9467bd",
    "fp4": "#d62728",
    "int8": "#2ca02c",
}

# Layer roles for sparsity bars. Key = (block, cin, cout).
ROLE_ORDER = [
    ("init", 18, 128, "Input CNN\n18→128"),
    ("init", 128, 128, "Input CNN\n128→128"),
    ("init", 128, 56, "Input CNN\n→ state 56"),
    ("agg", 56, 64, "User mix\n56→64"),
    ("agg", 64, 56, "User mix\n64→56"),
    ("upd", 114, 128, "Update CNN\n114→128"),
    ("upd", 128, 128, "Update CNN\n128→128"),
    ("upd", 128, 56, "Update CNN\n→ state 56"),
]


def _ieee_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.4,
            "axes.linewidth": 0.6,
            "legend.framealpha": 0.9,
            "pdf.fonttype": 42,
        }
    )


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _per_channel_normalize(arr):
    """Match QAT abs-max scaling: divide by per-output-channel max-abs."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 4:
        # Conv kernel (kh, kw, Cin, Cout); reduce spatial+in channels.
        scale = np.max(np.abs(a), axis=(0, 1, 2), keepdims=True)
    elif a.ndim == 2:
        # Dense (Cin, Cout); reduce input axis (same as QuantizedDense axis=0).
        scale = np.max(np.abs(a), axis=0, keepdims=True)
    else:
        return None
    scale = np.maximum(scale, 1e-12)
    return (a / scale).ravel()


def collect_normalized_kernels(weights):
    """Pointwise (1x1) + dense kernels only --- the prune-eligible set."""
    chunks = []
    n_pw = n_dense = 0
    for w in weights:
        a = np.asarray(w)
        if a.ndim == 4 and a.shape[0] == 1 and a.shape[1] == 1:
            chunks.append(_per_channel_normalize(a))
            n_pw += a.size
        elif a.ndim == 2 and min(a.shape) > 1:
            chunks.append(_per_channel_normalize(a))
            n_dense += a.size
    x = np.concatenate(chunks)
    x = x[np.isfinite(x)]
    x = np.clip(x, -1.0, 1.0)
    return x, {"pointwise": n_pw, "dense": n_dense, "total": int(x.size)}


def excess_kurtosis(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    m2 = np.mean(x * x)
    if m2 <= 0:
        return float("nan")
    m4 = np.mean(x ** 4)
    return float(m4 / (m2 ** 2) - 3.0)


def histogram_stats(x):
    absx = np.abs(x)
    first_int4 = 1.0 / INT4_MAX
    first_fp4 = 0.5 / FP4_MAX
    int4_pos = INT4_NARROW[INT4_NARROW >= 0] / INT4_MAX
    fp4_pos = FP4_E2M1[FP4_E2M1 >= 0] / FP4_MAX
    return {
        "n": int(x.size),
        "excess_kurtosis": excess_kurtosis(x),
        "frac_abs_lt_first_int4": float(np.mean(absx < first_int4)),
        "frac_abs_lt_first_fp4": float(np.mean(absx < first_fp4)),
        "frac_abs_lt_0p25": float(np.mean(absx < 0.25)),
        "frac_abs_lt_0p50": float(np.mean(absx < 0.50)),
        "n_int4_levels_le_0p25": int(np.sum(int4_pos <= 0.25 + 1e-12)),
        "n_fp4_levels_le_0p25": int(np.sum(fp4_pos <= 0.25 + 1e-12)),
        "first_int4_level": first_int4,
        "first_fp4_level": first_fp4,
    }


def _kernel_legend_handles():
    return [
        Patch(facecolor="#4c4c4c", alpha=0.35, edgecolor="#4c4c4c",
              label="FP32 kernels"),
        Line2D([0], [0], color=COLORS["int4"], lw=1.4, label="INT4 levels"),
        Line2D([0], [0], color=COLORS["fp4"], lw=1.4, label="FP4 (E2M1) levels"),
    ]


def _save_fig(fig, out_path, *, square=False):
    # square=True keeps the canvas 1:1 (no tight crop) for column figures.
    kw = {} if square else {"bbox_inches": "tight"}
    fig.savefig(out_path, **kw)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=300, **kw)
    plt.close(fig)


def plot_weight_histogram_signed(x, out_path):
    """Standalone signed histogram (former panel a)."""
    int4_unit = INT4_NARROW / INT4_MAX
    fp4_unit = FP4_E2M1 / FP4_MAX
    fig, ax = plt.subplots(figsize=(IEEE_SQUARE, IEEE_SQUARE), layout="constrained")
    ax.hist(
        x,
        bins=120,
        range=(-1.0, 1.0),
        density=True,
        color="#4c4c4c",
        alpha=0.35,
        histtype="stepfilled",
        zorder=1,
    )
    ax.hist(
        x,
        bins=120,
        range=(-1.0, 1.0),
        density=True,
        color="#4c4c4c",
        histtype="step",
        linewidth=0.8,
        zorder=2,
    )
    ymin, ymax = 0.0, 4.6
    ax.set_ylim(ymin, ymax)
    for v in int4_unit:
        ax.vlines(v, ymin, 0.18 * ymax, colors=COLORS["int4"], linewidth=0.9, zorder=3)
    for v in fp4_unit:
        ax.vlines(
            v, 0.18 * ymax, 0.36 * ymax, colors=COLORS["fp4"], linewidth=0.9, zorder=3
        )
    ax.set_xlim(-1.02, 1.02)
    ax.set_xlabel("Normalized weight")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(handles=_kernel_legend_handles(), loc="upper right", frameon=True)
    ax.set_box_aspect(1)
    ax.set_title("Weight mass concentrates at zero")
    _save_fig(fig, out_path, square=True)


def plot_weight_histogram_zoom(x, out_path):
    """Standalone |w| count histogram near zero (former panel b)."""
    int4_unit = INT4_NARROW / INT4_MAX
    fp4_unit = FP4_E2M1 / FP4_MAX
    first_int4 = 1.0 / INT4_MAX
    absx = np.abs(x)
    frac_int4 = float(np.mean(absx < first_int4))
    frac_body = float(np.mean(absx < 0.25))
    x_max = 0.40

    fig, ax = plt.subplots(figsize=(IEEE_SQUARE, IEEE_SQUARE), layout="constrained")
    counts, _, _ = ax.hist(
        absx,
        bins=40,
        range=(0.0, x_max),
        color="#4c4c4c",
        alpha=0.35,
        histtype="stepfilled",
        zorder=1,
    )
    ax.hist(
        absx,
        bins=40,
        range=(0.0, x_max),
        color="#4c4c4c",
        histtype="step",
        linewidth=0.8,
        zorder=2,
    )
    ymax = float(np.max(counts)) * 1.12
    ax.axvspan(0.0, first_int4, color=COLORS["int4"], alpha=0.12, zorder=0)
    for v in int4_unit:
        if 0.0 <= v <= x_max:
            ax.vlines(v, 0.0, 0.18 * ymax, colors=COLORS["int4"], linewidth=0.9, zorder=3)
    for v in fp4_unit:
        if 0.0 <= v <= x_max:
            ax.vlines(
                v, 0.18 * ymax, 0.36 * ymax, colors=COLORS["fp4"], linewidth=0.9, zorder=3
            )
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel(r"Absolute weight $|w|$")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(handles=_kernel_legend_handles(), loc="upper right", frameon=True)
    ax.set_box_aspect(1)
    _save_fig(fig, out_path, square=True)
    return {
        "frac_int4": frac_int4,
        "frac_body": frac_body,
        "first_int4": first_int4,
        "first_fp4": 0.5 / FP4_MAX,
    }


def _tensor_role(name, shape):
    cin, cout = None, None
    if len(shape) == 4:
        cin, cout = int(shape[2]), int(shape[3])
    elif len(shape) == 2:
        cin, cout = int(shape[0]), int(shape[1])
    if cin is None:
        return None
    if "state_init" in name:
        block = "init"
    elif "aggregate" in name:
        block = "agg"
    elif "update_state" in name:
        block = "upd"
    else:
        return None
    return (block, cin, cout)


def collect_sparsity(report_path):
    data = json.load(open(report_path))
    buckets = defaultdict(list)
    for name, info in data["report"]["per_tensor"].items():
        key = _tensor_role(name, info["shape"])
        if key is None:
            continue
        buckets[key].append(float(info["sparsity"]))
    out = {}
    for block, cin, cout, _label in ROLE_ORDER:
        vals = np.array(buckets.get((block, cin, cout), []), dtype=float)
        out[(block, cin, cout)] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std(ddof=0)) if len(vals) else float("nan"),
            "min": float(vals.min()) if len(vals) else float("nan"),
            "max": float(vals.max()) if len(vals) else float("nan"),
            "n": int(len(vals)),
            "values": [float(v) for v in vals],
        }
    return out, data["meta"]


def plot_sparsity_bars(by_fmt, out_path):
    """Square column figure: horizontal bars, residual rows highlighted."""
    short = {
        (18, 128): "18→128",
        (128, 128): "128→128",
        (128, 56): "→ state 56",
        (56, 64): "56→64",
        (64, 56): "64→56",
        (114, 128): "114→128",
    }
    blocks = [
        ("Input", ROLE_ORDER[0:3]),
        ("User mix", ROLE_ORDER[3:5]),
        ("Update", ROLE_ORDER[5:8]),
    ]
    formats = [
        ("int4", "INT4 QAT", COLORS["int4"], "///"),
        ("fp4", "FP4 QAT", COLORS["fp4"], "xxx"),
        ("int8", "INT8 QAT", COLORS["int8"], "..."),
    ]
    bar_h = 0.25

    # Build y from the bottom so Input sits at the top of the axes.
    rows = []
    block_spans = []
    y = 0.0
    for bname, members in reversed(blocks):
        y0 = y
        for item in reversed(members):
            block, cin, cout, _lab = item
            rows.append(
                {
                    "y": y,
                    "key": (block, cin, cout),
                    "label": short[(cin, cout)],
                    "residual": (cin, cout) == (128, 56),
                }
            )
            y += 1.0
        block_spans.append((bname, y0, y - 1.0))
        y += 0.62

    fig, ax = plt.subplots(figsize=(IEEE_SQUARE, IEEE_SQUARE))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.86, bottom=0.20)

    for bname, y0, y1 in block_spans:
        ax.axhspan(
            y0 - 0.42,
            y1 + 0.42,
            color="#f2f2f2" if bname != "User mix" else "#ffffff",
            zorder=0,
            lw=0,
        )
    for row in rows:
        if row["residual"]:
            ax.axhspan(
                row["y"] - 0.48,
                row["y"] + 0.48,
                color=COLORS["int4"],
                alpha=0.14,
                zorder=0.5,
                lw=0,
            )

    ax.axvline(50.0, color="#444444", ls="--", lw=0.8, zorder=1, label="50% target")

    for i, (key, legend, color, hatch) in enumerate(formats):
        offset = (1 - i) * bar_h
        ys = [r["y"] + offset for r in rows]
        means = []
        xerr = []
        for r in rows:
            rec = by_fmt[key][r["key"]]
            means.append(100.0 * rec["mean"])
            xerr.append(100.0 * rec["std"] if rec["n"] > 1 else 0.0)
        ax.barh(
            ys,
            means,
            height=bar_h,
            xerr=xerr,
            color=color,
            edgecolor="#333333",
            linewidth=0.25,
            hatch=hatch,
            label=legend,
            error_kw={"elinewidth": 0.55, "capsize": 1.2, "ecolor": "#333333"},
            zorder=2,
        )
        if key == "int4":
            for r, m in zip(rows, means):
                if r["residual"]:
                    ax.text(
                        m - 1.4,
                        r["y"] + offset,
                        f"{m:.0f}%",
                        ha="right",
                        va="center",
                        fontsize=6.2,
                        color="white",
                        fontweight="bold",
                        zorder=3,
                    )

    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7)
    ax.set_xlabel("Share of weights set to zero (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(
        min(r["y"] for r in rows) - 0.55,
        max(r["y"] for r in rows) + 0.55,
    )
    ax.grid(True, axis="x", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_title("INT4 over-zeros residual state writes\nunder a global 50% prune")
    for bname, y0, y1 in block_spans:
        trans = ax.get_yaxis_transform()
        if bname in ("Input", "Update"):
            trans = trans + ScaledTranslation(-2 / 25.4, 0, fig.dpi_scale_trans)
        ax.text(
            -0.22,
            0.5 * (y0 + y1),
            bname,
            transform=trans,
            ha="center",
            va="center",
            rotation=90,
            fontsize=6.5,
            fontweight="bold",
            clip_on=False,
        )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, -0.14),
        ncol=2,
        fontsize=6,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.6,
    )
    _save_fig(fig, out_path, square=True)


def _jsonable_roles(role_map):
    out = {}
    for (block, cin, cout), rec in role_map.items():
        out[f"{block}_{cin}_{cout}"] = rec
    return out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    _ieee_style()

    for key, path in QAT_WEIGHTS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    fp32_w = _load_pickle(QAT_WEIGHTS["float"])
    x, counts = collect_normalized_kernels(fp32_w)
    hstats = histogram_stats(x)
    signed_pdf = os.path.join(FIG_DIR, "weight_histogram_signed.pdf")
    zoom_pdf = os.path.join(FIG_DIR, "weight_histogram_zoom.pdf")
    plot_weight_histogram_signed(x, signed_pdf)
    plot_weight_histogram_zoom(x, zoom_pdf)

    by_fmt = {}
    meta = {}
    for key, path in PRUNE_REPORTS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        by_fmt[key], meta[key] = collect_sparsity(path)
    plot_sparsity_bars(by_fmt, os.path.join(FIG_DIR, "sparsity_bars.pdf"))

    # Highlight INT4 over-pruning of update pointwise layers.
    upd_keys = [("upd", 114, 128), ("upd", 128, 128), ("upd", 128, 56)]
    int4_upd_max = max(by_fmt["int4"][k]["max"] for k in upd_keys)
    fp4_upd_max = max(by_fmt["fp4"][k]["max"] for k in upd_keys)
    int4_upd_mean = float(
        np.mean([by_fmt["int4"][k]["mean"] for k in upd_keys])
    )
    fp4_upd_mean = float(
        np.mean([by_fmt["fp4"][k]["mean"] for k in upd_keys])
    )

    stats = {
        "weights": {k: os.path.relpath(p, REPO) for k, p in QAT_WEIGHTS.items()},
        "prune_reports": {
            k: os.path.relpath(p, REPO) for k, p in PRUNE_REPORTS.items()
        },
        "kernel_counts": counts,
        "histogram": hstats,
        "sparsity": {k: _jsonable_roles(v) for k, v in by_fmt.items()},
        "sparsity_highlights": {
            "int4_update_pw_mean": int4_upd_mean,
            "fp4_update_pw_mean": fp4_upd_mean,
            "int4_update_pw_max": int4_upd_max,
            "fp4_update_pw_max": fp4_upd_max,
            "global_target": 0.5,
        },
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print("Wrote", signed_pdf)
    print("Wrote", zoom_pdf)
    print("Wrote", os.path.join(FIG_DIR, "sparsity_bars.pdf"))
    print("Wrote", STATS_PATH)
    print(
        "kurtosis={:.2f}  frac<1/7={:.1f}%  frac<0.25={:.1f}%  "
        "INT4/FP4 levels in |w|<=0.25: {}/{}".format(
            hstats["excess_kurtosis"],
            100 * hstats["frac_abs_lt_first_int4"],
            100 * hstats["frac_abs_lt_0p25"],
            hstats["n_int4_levels_le_0p25"],
            hstats["n_fp4_levels_le_0p25"],
        )
    )
    print(
        "INT4 update-PW max sparsity={:.1f}%  FP4={:.1f}%".format(
            100 * int4_upd_max, 100 * fp4_upd_max
        )
    )


if __name__ == "__main__":
    main()
