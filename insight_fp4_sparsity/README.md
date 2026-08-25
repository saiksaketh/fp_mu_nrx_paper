# FP4 vs INT4 insight figures

Self-contained plot pipeline for the two analysis figures wired into
`sections/04_Experimental_Results.tex`:

1. **Weight histogram vs 4-bit grids** — why FP4 (E2M1) matches the receiver
   kernels better than uniform INT4.
2. **Per-layer sparsity bars** — how a *global* 50% magnitude threshold
   actually lands on StateInit / aggregation / update layers, and why INT4
   is over-pruned on the residual (`→ 56`) writes.

All generation code lives in this folder. Compile the paper from
`fp_mu_nrx_paper/` as usual; the figures are included by relative path.

## Deliverables

| File | Role |
|---|---|
| `figures/weight_histogram_signed.pdf` | Square column figure: signed kernel histogram vs 4-bit grids |
| `figures/weight_histogram_zoom.pdf` | Square column figure: zoomed $|w|$ count histogram |
| `figures/sparsity_bars.pdf` | Square column figure: horizontal sparsity bars, residual rows highlighted |
| `figures/*.png` | 300 dpi previews of the same plots |
| `stats.json` | Exact numbers quoted in the results text |
| `figures_inc.tex` | Pointer only; figure environments live in `sections/04_Experimental_Results.tex` |
| `make_insight_figures.py` | Generator |

## Usage

From the repo root, with `neural_rx_nvidia` active:

```bash
source /home/yellaps1/miniconda3/etc/profile.d/conda.sh && conda activate neural_rx_nvidia
python fp_mu_nrx_paper/insight_fp4_sparsity/make_insight_figures.py
```

The script needs only NumPy, Matplotlib, and the campaign checkpoints
already used by `scripts/run_eval_campaign.py`. It does **not** build the
TensorFlow graph.

### Inputs (campaign canonical paths)

QAT-only pickles (histogram):

- `weights/nrx_large_weights` (FP32 kernels; per-channel abs-max normalized)
- `weights/quantized_weights/nrx_large_qat_4bit_pc_weights` (path recorded, not required for the histogram)
- `weights/quantized_weights/nrx_large_qat_fp4_e2m1_pc_weights`

Prune reports matching the campaign pruned weights (sparsity bars):

- `weights/pruned_weights/nrx_large_qat_4bit_pc_magglobal50/..._report.json`
- `weights/pruned_weights/nrx_large_qat_fp4_e2m1_pc_magglobal50/..._report.json`
- `weights/pruned_weights/nrx_large_qat_8bit_pc_magglobal50/..._report.json`

### What is measured

- Histogram uses **pointwise 1×1 + dense** kernels (404,224 weights), i.e. the
  prune-eligible set. Depthwise and biases are excluded.
- Each output channel is divided by its abs-max, matching QAT scaling in
  `utils/quantized_conv2d.py`. INT4 levels are \(\pm k/7\) (4-bit, narrow
  range); FP4 levels are E2M1 \(\,g/6\).
- `weight_histogram_zoom.pdf` is a **count** histogram of \(|w|\), zoomed
  to \([0, 0.4]\). Same kernels as the signed plot, folded to magnitude.
  The 65\% / 79\% callouts are fractions of *all* weights, including those
  outside the zoom window. Figure environments live in
  `sections/04_Experimental_Results.tex`.
- Sparsity is the **mask** sparsity from the global 50% magnitude reports,
  grouped by layer role into Input / User mix / Update. Residual
  writes ($\to$ state 56) are highlighted. Error bars are the standard
  deviation across the eight unrolled CGNN iterations; StateInit has a
  single copy and is drawn without a whisker.

Recompile `main.tex` after regenerating PDFs. Numbers in the results
prose come from `stats.json`; update those sentences if you change
checkpoints.