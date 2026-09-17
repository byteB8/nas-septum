"""Score saved out-of-fold probabilities over a range of thresholds.

The legacy-mask model (A0) predicts nothing at 0.5, so it is also reported at
the threshold that suits it best. That is optimistic for A0 (the threshold is
picked on its own test predictions) and is done identically for the setting it
is compared with. Writes results/threshold_sweep.csv.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image

from septum.metrics import slice_metrics

THRESHOLDS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--configs", nargs="+", default=["a0_unet_scratch_bce_legacy_masks", "a_unet_scratch_bce"])
    args = ap.parse_args()

    gts = {}
    rows = []
    for cfg in args.configs:
        preds = {}
        for p in sorted(glob.glob(os.path.join(args.results, cfg, "fold*", "preds.npz"))):
            preds.update(dict(np.load(p)))
        maxp = np.array([v.max() / 255.0 for v in preds.values()])
        print(f"{cfg}: {len(preds)} slices, max probability per slice: median {np.median(maxp):.3f}, "
              f"p90 {np.percentile(maxp, 90):.3f}")
        for thr in THRESHOLDS:
            recs = []
            for key, prob in preds.items():
                if key not in gts:
                    f = key + ".png"
                    gts[key] = (np.array(Image.open(os.path.join(args.data, "masks", f))) > 0,
                                np.array(Image.open(os.path.join(args.data, "centerlines", f))) > 0)
                recs.append(slice_metrics(prob / 255.0, *gts[key], thr=thr))
            d = pd.DataFrame(recs)
            rows.append(dict(config=cfg, threshold=thr, n=len(d), detect_rate=d.detected.mean(),
                             f1_tol=d.f1_tol.mean(), cldice=d.cldice.mean(), dice_band=d.dice_band.mean(),
                             cl_mean_dist=d.cl_mean_dist.mean()))
            print(f"  thr {thr:<5} detected {d.detected.mean():.3f}  F1@2px {d.f1_tol.mean():.3f}  "
                  f"clDice {d.cldice.mean():.3f}  dist {d.cl_mean_dist.mean():.2f}px", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.results, "threshold_sweep.csv"), index=False)
    best = out.loc[out.groupby("config").f1_tol.idxmax()]
    print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
