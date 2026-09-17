"""Summarise out-of-fold results across configs and draw figures.

Every slice is predicted exactly once (by the model that did not see its scan),
so per-scan statistics use all 23 scans.
Writes results/summary.md, results/summary.csv and results/figures/*.png.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

LABELS = {
    "a0_unet_scratch_bce_legacy_masks": "U-Net (scratch), BCE, 2024 masks (>127)",
    "a_unet_scratch_bce": "U-Net (scratch), BCE",
    "b_unet_scratch_bcedice": "U-Net (scratch), BCE+Dice",
    "c_resnet34_bcedice": "ResNet34 U-Net, BCE+Dice",
    "d_resnet34_cldice": "ResNet34 U-Net, +clDice",
    "e_resnet34_cldice_25d": "ResNet34 U-Net, +clDice, 2.5D",
}

SHORT = {c: c.split("_")[0].upper() for c in LABELS}


def load(results):
    frames, logs = [], []
    for cfg in LABELS:
        for f in sorted(glob.glob(os.path.join(results, cfg, "fold*", "metrics.csv"))):
            d = pd.read_csv(f)
            d["config"] = cfg
            frames.append(d)
            logs.append(dict(config=cfg, **{k: v for k, v in json.load(
                open(f.replace("metrics.csv", "log.json"))).items() if k in
                ("fold", "train_seconds", "infer_ms_per_slice")}))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(logs)


def summarise(df, logs):
    rows = []
    for cfg, d in df.groupby("config", sort=False):
        folds = d.groupby("fold")
        per_scan = d.groupby("scan").agg(dev_pred=("dev_pred", "max"), dev_gt=("dev_gt", "max"),
                                         ratio_pred=("dev_ratio_pred", "max"), ratio_gt=("dev_ratio_gt", "max"))
        err = (d.dev_pred - d.dev_gt).abs()
        row = dict(config=cfg, label=LABELS[cfg], n_folds=d.fold.nunique(), n_slices=len(d))
        for k in ("f1_tol", "cldice", "dice_band", "cl_mean_dist", "cl_hd95"):
            fm = folds[k].mean()
            row[k] = d[k].mean()
            row[k + "_fold_std"] = fm.std(ddof=1) if len(fm) > 1 else np.nan
        row["detect_rate"] = d.detected.mean()
        row["fragmented_rate"] = (d.n_components > 1).mean()
        row["dev_mae_px"] = err.mean()
        row["dev_r_slice"] = d[["dev_pred", "dev_gt"]].corr().iloc[0, 1]
        row["scan_dev_mae_px"] = (per_scan.dev_pred - per_scan.dev_gt).abs().mean()
        row["scan_dev_r"] = per_scan[["dev_pred", "dev_gt"]].corr().iloc[0, 1]
        lg = logs[logs.config == cfg]
        row["train_min_per_fold"] = lg.train_seconds.mean() / 60
        row["infer_ms_per_slice"] = lg.infer_ms_per_slice.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(s):
    lines = ["| Model | F1@2px | clDice | Dice (3-px band) | Centreline dist (px) | HD95 (px) | "
             "Deviation MAE (px) | Scan-level deviation r | Fragmented |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        pm = lambda k, fmt: f"{r[k]:{fmt}} ± {r[k + '_fold_std']:{fmt}}"
        lines.append(f"| {r.label} | {pm('f1_tol', '.3f')} | {pm('cldice', '.3f')} | {pm('dice_band', '.3f')} | "
                     f"{pm('cl_mean_dist', '.2f')} | {pm('cl_hd95', '.2f')} | {r.dev_mae_px:.2f} | "
                     f"{r.scan_dev_r:.2f} | {100 * r.fragmented_rate:.1f}% |")
    lines.append("")
    lines.append("Mean over all out-of-fold slices ± std of the 5 fold means. Scans are never shared "
                 "between training and test folds.")
    return "\n".join(lines)


def figures(df, s, data_dir, results, best):
    fig_dir = os.path.join(results, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. per-fold score by config
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (k, title) in zip(axes, [("f1_tol", "Centreline F1 @ 2 px"), ("cldice", "clDice"),
                                     ("cl_mean_dist", "Mean centreline distance (px)")]):
        fm = df.groupby(["config", "fold"])[k].mean().reset_index()
        order = [c for c in LABELS if c in fm.config.unique()]
        data = [fm[fm.config == c][k].values for c in order]
        ax.boxplot(data, widths=0.5)
        for i, v in enumerate(data):
            ax.scatter(np.full(len(v), i + 1) + np.random.uniform(-0.08, 0.08, len(v)), v, s=14, zorder=3)
        ax.set_xticks(range(1, len(order) + 1), [SHORT[c] for c in order])
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.text(0.5, -0.02, "  ".join(f"{SHORT[c]}: {LABELS[c]}" for c in order), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "ablation_folds.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2. deviation agreement for the best config
    d = df[df.config == best]
    per_scan = d.groupby("scan").agg(p=("dev_pred", "max"), g=("dev_gt", "max"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    ax.scatter(d.dev_gt, d.dev_pred, s=6, alpha=0.4)
    lim = np.nanmax([d.dev_gt.max(), d.dev_pred.max()]) + 1
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set(xlabel="Deviation from annotation (px)", ylabel="Deviation from prediction (px)",
           title=f"Per slice (r = {d[['dev_pred', 'dev_gt']].corr().iloc[0, 1]:.2f})")
    ax = axes[1]
    ax.scatter(per_scan.g, per_scan.p, s=30)
    lim = max(per_scan.g.max(), per_scan.p.max()) + 1
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set(xlabel="Max deviation, annotation (px)", ylabel="Max deviation, prediction (px)",
           title=f"Per scan, n={len(per_scan)} (r = {per_scan.corr().iloc[0, 1]:.2f})")
    ax = axes[2]
    mean, diff = (d.dev_pred + d.dev_gt) / 2, d.dev_pred - d.dev_gt
    ax.scatter(mean, diff, s=6, alpha=0.4)
    md, sd = diff.mean(), diff.std()
    for y, ls in ((md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")):
        ax.axhline(y, color="k", ls=ls, lw=1)
    ax.set(xlabel="Mean of the two (px)", ylabel="Prediction − annotation (px)",
           title=f"Bland–Altman: bias {md:.2f}, LoA ±{1.96 * sd:.2f} px")
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle(LABELS[best])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "deviation_agreement.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3. qualitative: best / median / worst slices by centreline F1
    preds = {}
    for f in range(5):
        p = os.path.join(results, best, f"fold{f}", "preds.npz")
        if os.path.exists(p):
            preds.update(dict(np.load(p)))
    ds = d.sort_values("f1_tol")
    picks = list(ds.tail(3).slice) + list(ds.iloc[len(ds) // 2 - 1:len(ds) // 2 + 2].slice) + list(ds.head(3).slice)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, sl, tag in zip(axes.ravel(), picks, ["best"] * 3 + ["median"] * 3 + ["worst"] * 3):
        f = f"slice_{int(sl):04d}.png"
        img = np.array(Image.open(os.path.join(data_dir, "images", f)))
        gt = np.array(Image.open(os.path.join(data_dir, "centerlines", f))) > 0
        pr = preds[f"slice_{int(sl):04d}"] > 127
        rgb = np.stack([img] * 3, -1).astype(float) * 0.8
        rgb[pr] = rgb[pr] * 0.3 + np.array([255, 60, 60]) * 0.7
        rgb[gt] = [60, 255, 60]
        ys, xs = np.where(gt | pr)
        y0, y1 = max(ys.min() - 15, 0), ys.max() + 15
        x0, x1 = max(xs.min() - 25, 0), xs.max() + 25
        ax.imshow(rgb[y0:y1, x0:x1].astype(np.uint8), interpolation="nearest")
        r = d[d.slice == sl].iloc[0]
        ax.set_title(f"{tag}: slice {int(sl)}, F1@2px {r.f1_tol:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Green: annotated centreline; red: predicted septum")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "qualitative.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    df, logs = load(args.results)
    s = summarise(df, logs)
    s.to_csv(os.path.join(args.results, "summary.csv"), index=False)
    md = to_markdown(s)
    open(os.path.join(args.results, "summary.md"), "w").write(md + "\n")
    print(md)
    print(s[["config", "n_folds", "detect_rate", "dev_mae_px", "dev_r_slice", "scan_dev_mae_px",
             "scan_dev_r", "train_min_per_fold", "infer_ms_per_slice"]].round(3).to_string(index=False))
    best = s[s.n_folds == s.n_folds.max()].sort_values("f1_tol").iloc[-1].config
    figures(df, s, args.data, args.results, best)


if __name__ == "__main__":
    main()
