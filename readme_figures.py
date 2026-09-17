"""Figures for the README, rendered in light and dark variants.

docs/figures/<name>-light.png and <name>-dark.png, used with <picture> so GitHub
shows the variant matching the viewer's theme. Colours: slot 1 (blue) is the
annotation / primary data, slot 2 (orange) is the model prediction; both pass
the colour-vision checks on either surface.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

from aggregate import LABELS, load

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  s1="#2a78d6", s2="#eb6834", fold="#c9c8c1"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 s1="#3987e5", s2="#d95926", fold="#4a4a47"),
}
SHORT = {"a0_unet_scratch_bce_legacy_masks": "A0", "a_unet_scratch_bce": "A", "b_unet_scratch_bcedice": "B",
         "c_resnet34_bcedice": "C", "d_resnet34_cldice": "D", "e_resnet34_cldice_25d": "E"}


def style(t):
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["grid"], "axes.labelcolor": t["ink2"], "axes.titlecolor": t["ink"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"], "text.color": t["ink"], "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.labelcolor": t["ink2"], "font.family": "DejaVu Sans",
    })


def rgb(img):
    return np.stack([img] * 3, -1).astype(float) / 255.0


def paint(base, mask, hex_color, alpha):
    c = np.array(matplotlib.colors.to_rgb(hex_color))
    out = base.copy()
    out[mask] = out[mask] * (1 - alpha) + c * alpha
    return out


def crop_box(mask, pad_y=12, pad_x=22, shape=None):
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad_y, 0), min(ys.max() + pad_y, shape[0])
    x0, x1 = max(xs.min() - pad_x, 0), min(xs.max() + pad_x, shape[1])
    return slice(y0, y1), slice(x0, x1)


def smooth_centerline(c, smooth=5):
    pts = np.argwhere(c)
    ys = np.unique(pts[:, 0])
    xs = np.array([pts[pts[:, 0] == y, 1].mean() for y in ys])
    return ys, ndimage.uniform_filter1d(xs, smooth, mode="nearest")


def load_preds(results, cfg):
    preds = {}
    for p in sorted(glob.glob(os.path.join(results, cfg, "fold*", "preds.npz"))):
        preds.update(dict(np.load(p)))
    return preds


def save(fig, out, name, mode):
    fig.savefig(os.path.join(out, f"{name}-{mode}.png"), dpi=170, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------- figures

def fig_overview(t, mode, ctx):
    """CT slice -> predicted septum -> centreline, chord and deviation."""
    d, preds, data = ctx["best_df"], ctx["best_preds"], ctx["data"]
    good = d[(d.f1_tol >= 0.95) & (d.septum_len_gt >= d.septum_len_gt.median())]
    sl = int(good.sort_values("dev_ratio_gt").iloc[-1].slice)
    f = f"slice_{sl:04d}.png"
    img = np.array(Image.open(os.path.join(data, "images", f)))
    gt_c = np.array(Image.open(os.path.join(data, "centerlines", f))) > 0
    prob = preds[f"slice_{sl:04d}"] / 255.0
    pred = prob > 0.5
    pred_c = skeletonize(pred)
    ys, xs = smooth_centerline(pred_c)
    top, bot = np.array([ys[0], xs[0]]), np.array([ys[-1], xs[-1]])
    v = bot - top
    dist = np.abs(v[1] * (ys - top[0]) - v[0] * (xs - top[1])) / np.hypot(*v)
    k = int(np.argmax(dist))
    p = np.array([ys[k], xs[k]])
    foot = top + v * np.dot(p - top, v) / np.dot(v, v)
    by, bx = crop_box(gt_c | pred, 14, 30, img.shape)

    crop_ratio = (bx.stop - bx.start) / (by.stop - by.start)
    full_ratio = img.shape[1] / img.shape[0]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.35),
                             gridspec_kw=dict(wspace=0.05, width_ratios=[full_ratio] + [crop_ratio] * 3))
    panels = [
        ("1  Coronal CT crop", rgb(img)),
        ("2  Predicted septum", paint(rgb(img), pred, t["s2"], 0.75)),
        ("3  Against the annotation", paint(paint(rgb(img), pred, t["s2"], 0.55), gt_c, t["s1"], 1.0)),
        ("4  Deviation from straight", rgb(img) * 0.85),
    ]
    for i, (ax, (title, im)) in enumerate(zip(axes, panels)):
        view = im if i == 0 else im[by, bx]
        ax.imshow(view, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].add_patch(plt.Rectangle((bx.start, by.start), bx.stop - bx.start, by.stop - by.start,
                                    fill=False, ec=t["s2"], lw=1.2))
    ax = axes[3]
    oy, ox = by.start, bx.start
    ax.plot(xs - ox, ys - oy, color=t["s2"], lw=2)
    ax.plot([top[1] - ox, bot[1] - ox], [top[0] - oy, bot[0] - oy], color="white", lw=1.4, ls=(0, (3, 2)))
    ax.plot([foot[1] - ox, p[1] - ox], [foot[0] - oy, p[0] - oy], color=t["s1"], lw=3, solid_capstyle="butt")
    ax.scatter([foot[1] - ox, p[1] - ox], [foot[0] - oy, p[0] - oy], s=18, color=t["s1"], zorder=4)
    ratio = 100 * dist[k] / np.hypot(*v)
    ax.text(0.03, 0.03, f"d = {dist[k]:.1f} px  ({ratio:.1f}% of length)", transform=ax.transAxes,
            color="white", fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.6))
    handles = [Patch(color=t["s2"], label="Prediction"), Line2D([], [], color=t["s1"], lw=2, label="Annotation"),
               Line2D([], [], color=t["ink2"], lw=1.4, ls=(0, (3, 2)), label="Straight chord (panel 4)"),
               Line2D([], [], color=t["s1"], lw=3, label="Deviation d (panel 4)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.06))
    save(fig, ctx["out"], "overview", mode)


def fig_label_fix(t, mode, ctx):
    """Why thresholding the Label Studio strokes at >127 broke the targets."""
    src = ctx["src"]
    files = sorted(glob.glob(os.path.join(src, "*_mask", "*.png")))
    connected = {127: [], 0: []}
    for fpath in files:
        raw = np.array(Image.open(fpath).convert("L"))
        for thr in connected:
            m = raw > thr
            connected[thr].append(m.any() and ndimage.label(m, np.ones((3, 3)))[1] == 1)
    share = {thr: 100 * np.mean(v) for thr, v in connected.items()}

    # an example stroke that breaks up at >127
    ex = None
    for fpath in files[::3]:
        raw = np.array(Image.open(fpath).convert("L"))
        n = ndimage.label(raw > 127, np.ones((3, 3)))[1]
        if n >= 4 and (raw > 0).sum() > 60:
            ex = fpath
            break
    raw = np.array(Image.open(ex).convert("L"))
    img_path = ex.replace("_mask", "_images", 1).replace("_mask.png", ".png")
    img = np.array(Image.open(img_path).convert("L"))
    by, bx = crop_box(raw > 0, 6, 14, img.shape)

    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 6, width_ratios=[0.85, 0.85, 0.08, 1.2, 0.28, 1.75], wspace=0.1)
    for i, (thr, title) in enumerate([(127, "Stroke > 127 (2024)"), (0, "Stroke > 0 (fixed)")]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(paint(rgb(img), raw > thr, t["s1"] if thr == 0 else t["s2"], 1.0)[by, bx],
                  interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax = fig.add_subplot(gs[0, 3])
    vals = [(share[127], t["s2"], "> 127"), (share[0], t["s1"], "> 0")]
    ax.bar([0, 1], [v for v, _, _ in vals], width=0.6, color=[c for _, c, _ in vals])
    for xi, (v, _, _) in enumerate(vals):
        ax.text(xi, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=9.5, color=t["ink"],
                fontweight="bold" if xi else "normal")
    ax.set_xticks([0, 1], [lab for _, _, lab in vals])
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", visible=False)
    ax.set_title("Traces kept as one line")

    sweep_path = os.path.join(ctx["results"], "threshold_sweep.csv")
    ax = fig.add_subplot(gs[0, 5])
    if os.path.exists(sweep_path):
        sw = pd.read_csv(sweep_path)
        for cfg, color, lab in (("a0_unet_scratch_bce_legacy_masks", t["s2"], "Trained on > 127 masks"),
                                ("a_unet_scratch_bce", t["s1"], "Trained on fixed masks")):
            d = sw[sw.config == cfg].sort_values("threshold")
            ax.plot(d.threshold, d.f1_tol, color=color, lw=2, marker="o", ms=5, label=lab)
        ax.axvline(0.5, color=t["muted"], lw=1)
        ax.text(0.47, 0.04, "default\nthreshold", ha="right", va="bottom", fontsize=8.5, color=t["ink2"])
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.02)
        ax.set(xlabel="Probability threshold", ylabel="Centreline F1 @ 2 px")
        ax.legend(loc="lower left", fontsize=8.5)
        ax.set_title("Same U-Net, scored across thresholds")
    save(fig, ctx["out"], "label_fix", mode)


def fig_ablation(t, mode, ctx):
    """Per-fold scores for every setting, folds joined to show they are paired."""
    df, s = ctx["df"], ctx["summary"]
    # A0 (legacy masks) predicts nothing at 0.5 and is shown in the label-fix figure instead
    order = [c for c in SHORT if c in set(df.config) and c != "a0_unet_scratch_bce_legacy_masks"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw=dict(wspace=0.28))
    for ax, (k, title, fmt) in zip(axes, [("f1_tol", "Centreline F1 within 2 px (higher is better)", "{:.3f}"),
                                          ("cl_mean_dist", "Mean centreline distance, px (lower is better)",
                                           "{:.2f}")]):
        fm = df[df.config.isin(order)].groupby(["config", "fold"])[k].mean().unstack()
        for fold in fm.columns:
            vals = [fm.loc[c, fold] if c in fm.index else np.nan for c in order]
            ax.plot(x, vals, color=t["fold"], lw=1, marker="o", ms=3.5, zorder=2)
        means = [df[df.config == c][k].mean() for c in order]
        ax.scatter(x, means, s=70, color=t["s1"], edgecolor=t["surface"], linewidth=2, zorder=4)
        for xi, m in zip(x, means):
            ax.annotate(fmt.format(m), (xi, m), xytext=(9, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=t["ink"])
        ax.set_xticks(x, [SHORT[c] for c in order])
        ax.set_xlim(-0.4, len(order) - 0.3)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], color=t["fold"], marker="o", ms=3.5, lw=1, label="One held-out fold (4-5 scans)"),
               Line2D([], [], color=t["s1"], marker="o", ms=8, lw=0, label="Mean over all slices")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    save(fig, ctx["out"], "ablation", mode)


def fig_deviation(t, mode, ctx):
    d = ctx["best_df"]
    per_scan = d.groupby("scan").agg(p=("dev_pred", "max"), g=("dev_gt", "max"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(wspace=0.28))
    ax = axes[0]
    lim = max(per_scan.g.max(), per_scan.p.max()) * 1.1
    ax.plot([0, lim], [0, lim], color=t["muted"], lw=1)
    ax.scatter(per_scan.g, per_scan.p, s=48, color=t["s1"], edgecolor=t["surface"], linewidth=1.5, zorder=3)
    r = per_scan.corr().iloc[0, 1]
    mae = (per_scan.p - per_scan.g).abs().mean()
    ax.text(0.04, 0.95, f"r = {r:.2f}\nMAE = {mae:.2f} px\n{len(per_scan)} scans", transform=ax.transAxes,
            va="top", color=t["ink"], fontsize=9.5)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="From annotation (px)", ylabel="From prediction (px)")
    ax.set_title("Largest septal deviation per scan")

    ax = axes[1]
    dd = d.dropna(subset=["dev_pred", "dev_gt"])
    mean, diff = (dd.dev_pred + dd.dev_gt) / 2, dd.dev_pred - dd.dev_gt
    md, sd = diff.mean(), diff.std()
    ax.scatter(mean, diff, s=9, color=t["s1"], alpha=0.45, linewidth=0)
    ax.axhline(md, color=t["ink2"], lw=1.2)
    for yv in (md - 1.96 * sd, md + 1.96 * sd):
        ax.axhline(yv, color=t["muted"], lw=1)
    tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for yv, lab in ((md, f"bias {md:+.2f}"), (md + 1.96 * sd, f"+1.96 SD {md + 1.96 * sd:+.2f}"),
                    (md - 1.96 * sd, f"-1.96 SD {md - 1.96 * sd:+.2f}")):
        ax.text(1.02, yv, lab, transform=tr, va="center", ha="left", fontsize=8.5, color=t["ink2"], clip_on=False)
    ax.set(xlabel="Mean of prediction and annotation (px)", ylabel="Prediction - annotation (px)")
    ax.set_title(f"Per-slice agreement (Bland-Altman, {len(dd)} slices)")
    save(fig, ctx["out"], "deviation", mode)


def fig_examples(t, mode, ctx):
    """One typical slice from each of 8 scans, plus the 4 hardest slices."""
    d, preds, data = ctx["best_df"], ctx["best_preds"], ctx["data"]
    scans = d.groupby("scan").f1_tol.mean().sort_values().index
    pick_scans = list(scans[np.linspace(0, len(scans) - 1, 8).round().astype(int)])
    typical = []
    for sc in pick_scans:
        ds = d[d.scan == sc].sort_values("pos_in_scan")
        typical.append(ds.iloc[len(ds) // 2])
    hardest = list(d.sort_values("f1_tol").head(4).itertuples())
    fig = plt.figure(figsize=(9.2, 11.6))
    top_fig, bottom_fig = fig.subfigures(2, 1, height_ratios=[2, 1.2], hspace=0.0)
    top_fig.suptitle("Typical slices, one per scan (held-out predictions)", x=0.02, ha="left",
                     fontsize=11.5, fontweight="bold", color=t["ink"])
    bottom_fig.suptitle("Hardest slices: disagreement sits at the upper end of the septum",
                        x=0.02, ha="left", fontsize=11.5, fontweight="bold", color=t["ink"])
    axes = list(top_fig.subplots(2, 4, gridspec_kw=dict(hspace=0.16, wspace=0.05)).ravel()) + \
        list(bottom_fig.subplots(1, 4, gridspec_kw=dict(wspace=0.05)).ravel())
    for sub, top in ((top_fig, 0.93), (bottom_fig, 0.84)):
        sub.set_facecolor(t["surface"])
        sub.subplots_adjust(left=0.01, right=0.99, top=top, bottom=0.03)
    items = [(r, "typical") for r in typical] + [(r, "hard") for r in hardest]
    for ax, (r, kind) in zip(axes, items):
        sl = int(r.slice)
        f = f"slice_{sl:04d}.png"
        img = np.array(Image.open(os.path.join(data, "images", f)))
        gt_c = np.array(Image.open(os.path.join(data, "centerlines", f))) > 0
        pred = preds[f"slice_{sl:04d}"] > 127
        by, bx = crop_box(gt_c | pred, 12, 26, img.shape)
        ax.imshow(paint(paint(rgb(img), pred, t["s2"], 0.6), gt_c, t["s1"], 1.0)[by, bx], interpolation="nearest",
                  aspect="auto")
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"scan {int(r.scan)} · F1 {r.f1_tol:.2f}", fontsize=9, fontweight="normal",
                     color=t["ink2"])
    handles = [Patch(color=t["s2"], label="Prediction"), Line2D([], [], color=t["s1"], lw=2, label="Annotation")]
    bottom_fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    save(fig, ctx["out"], "examples", mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--src", default="../data/final-v4")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--best", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, _ = load(args.results)
    summary = pd.read_csv(os.path.join(args.results, "summary.csv"))
    best = args.best or summary[summary.config != "a0_unet_scratch_bce_legacy_masks"].sort_values(
        "f1_tol").iloc[-1].config
    ctx = dict(df=df, summary=summary, data=args.data, src=args.src, out=args.out, results=args.results,
               best_df=df[df.config == best], best_preds=load_preds(args.results, best))
    print("best config:", best, LABELS[best])
    for mode, t in THEMES.items():
        style(t)
        for fn in (fig_overview, fig_label_fix, fig_ablation, fig_deviation, fig_examples):
            fn(t, mode, ctx)
            print("wrote", fn.__name__, mode)


if __name__ == "__main__":
    main()
