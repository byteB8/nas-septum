"""Build a clean slice index from data/final-v4.

- Label Studio strokes are anti-aliased 1-px lines: thresholding at >127 breaks
  them into dashes (only 32% stay connected), while >0 keeps 98% as a single
  continuous line. So: mask > 0 -> skeleton (the annotated centreline) -> a
  3-px band around it as the training target.
- Recovers the scan each slice came from. Slices were cropped per DICOM series
  and renumbered, and every series has its own crop size, so crop size == scan.
- Finds the previous/next slice within the same scan (for 2.5D input).
- Assigns scan-grouped cross-validation folds, balanced by labeled slice count.
- A slice with an empty mask would be marked unlabeled (kept only as a 2.5D
  neighbour); with the >0 threshold every slice has a stroke.

Writes data/slices.csv, data/{images,masks,centerlines,masks_legacy}/*.png and data/stats.json.
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/final-v4")
    ap.add_argument("--out", default="data")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "centerlines"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "masks_legacy"), exist_ok=True)

    rows = []
    for split in ("train", "val"):
        for fname in os.listdir(os.path.join(args.src, f"{split}_images")):
            idx = int(re.search(r"(\d+)", fname).group(1))
            img = np.array(Image.open(os.path.join(args.src, f"{split}_images", fname)).convert("L"))
            mask = np.array(Image.open(os.path.join(args.src, f"{split}_mask",
                                                    fname.replace(".png", "_mask.png"))).convert("L"))
            assert img.shape == mask.shape, fname
            # the 2024 target, kept only to measure what the threshold fix is worth
            Image.fromarray((mask > 127).astype(np.uint8) * 255).save(os.path.join(args.out, "masks_legacy", f"slice_{idx:04d}.png"))
            center = skeletonize(mask > 0)
            mask = ndimage.binary_dilation(center, structure=np.ones((3, 3))).astype(np.uint8)
            name = f"slice_{idx:04d}.png"
            Image.fromarray(img).save(os.path.join(args.out, "images", name))
            Image.fromarray(mask * 255).save(os.path.join(args.out, "masks", name))
            Image.fromarray(center.astype(np.uint8) * 255).save(os.path.join(args.out, "centerlines", name))
            rows.append(dict(slice=idx, h=img.shape[0], w=img.shape[1], fg_px=int(mask.sum()),
                             labeled=int(mask.sum() > 0), file=name))

    # crop size -> scan id, numbered in order of first appearance
    rows.sort(key=lambda r: r["slice"])
    scan_of_size = {}
    for r in rows:
        scan_of_size.setdefault((r["h"], r["w"]), len(scan_of_size))
        r["scan"] = scan_of_size[(r["h"], r["w"])]

    by_scan = defaultdict(list)
    for r in rows:
        by_scan[r["scan"]].append(r)
    for scan_rows in by_scan.values():
        for i, r in enumerate(scan_rows):
            r["prev_file"] = scan_rows[max(i - 1, 0)]["file"]
            r["next_file"] = scan_rows[min(i + 1, len(scan_rows) - 1)]["file"]
            r["pos_in_scan"] = i

    # greedy balanced grouped folds: biggest scans first, into the lightest fold
    load = [0] * args.folds
    fold_of_scan = {}
    for scan in sorted(by_scan, key=lambda s: -sum(r["labeled"] for r in by_scan[s])):
        f = int(np.argmin(load))
        fold_of_scan[scan] = f
        load[f] += sum(r["labeled"] for r in by_scan[scan])
    for r in rows:
        r["fold"] = fold_of_scan[r["scan"]]

    # sanity check: consecutive slices in a scan should look more alike than random pairs
    def load_img(f):
        return np.array(Image.open(os.path.join(args.out, "images", f)), dtype=np.float32)
    rng = np.random.default_rng(0)
    adj, rnd = [], []
    for s, scan_rows in by_scan.items():
        for i in rng.choice(len(scan_rows) - 1, size=min(5, len(scan_rows) - 1), replace=False):
            a, b = load_img(scan_rows[i]["file"]), load_img(scan_rows[i + 1]["file"])
            c = load_img(scan_rows[rng.integers(len(scan_rows))]["file"])
            adj.append(np.abs(a - b).mean())
            rnd.append(np.abs(a - c).mean())

    labeled = [r for r in rows if r["labeled"]]
    pix = np.concatenate([load_img(r["file"]).ravel() for r in rows[::7]]) / 255.0
    stats = dict(
        n_slices=len(rows), n_labeled=len(labeled), n_scans=len(by_scan),
        slices_per_scan={int(s): len(v) for s, v in by_scan.items()},
        labeled_per_fold=load, scans_per_fold=[sum(1 for v in fold_of_scan.values() if v == f)
                                               for f in range(args.folds)],
        fg_px_median=float(np.median([r["fg_px"] for r in labeled])),
        max_h=max(r["h"] for r in rows), max_w=max(r["w"] for r in rows),
        mean=float(pix.mean()), std=float(pix.std()),
        adjacent_slice_absdiff=float(np.mean(adj)), random_same_scan_absdiff=float(np.mean(rnd)),
    )
    cols = ["slice", "scan", "fold", "pos_in_scan", "labeled", "fg_px", "h", "w", "file", "prev_file", "next_file"]
    with open(os.path.join(args.out, "slices.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(args.out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
