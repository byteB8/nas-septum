"""Train on 4 folds of scans and evaluate on the held-out fold.

The last-epoch model is evaluated (no checkpoint picking on the test fold).
Writes <out>/<config>/fold<k>/{metrics.csv, preds.npz, log.json}.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from septum.data import SeptumDataset, read_index
from septum.losses import build_loss
from septum.metrics import slice_metrics
from septum.models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    probs, times = {}, []
    for x, _, geom, idx in loader:
        x = x.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
            if tta:
                logits = (logits + torch.flip(model(torch.flip(x, [3])).float(), [3])) / 2
        p = torch.sigmoid(logits).cpu().numpy()[:, 0]
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / len(x))
        for pi, (top, left, h, w), i in zip(p, geom.tolist(), idx.tolist()):
            probs[i] = pi[top:top + h, left:left + w]
    return probs, float(np.median(times) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name = os.path.splitext(os.path.basename(args.config))[0]
    epochs = args.epochs or cfg["epochs"]
    out_dir = os.path.join(args.out, name, f"fold{args.fold}")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.get("seed", 0) + args.fold)
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    rows, stats = read_index(args.data)
    train_rows = [r for r in rows if r["labeled"] and r["fold"] != args.fold]
    test_rows = [r for r in rows if r["labeled"] and r["fold"] == args.fold]
    ctx = cfg.get("context", 1)
    train_ds = SeptumDataset(args.data, train_rows, stats, context=ctx, augment=True,
                             mask_dir=cfg.get("train_masks", "masks"))
    test_ds = SeptumDataset(args.data, test_rows, stats, context=ctx, augment=False)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4,
                          pin_memory=True, drop_last=True, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    loss_fn = build_loss(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    total = epochs * len(train_dl)
    warm = min(5 * len(train_dl), total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    log = dict(config=cfg, name=name, fold=args.fold, epochs=epochs, n_train=len(train_rows),
               n_test=len(test_rows), test_scans=sorted({r["scan"] for r in test_rows}), loss=[])
    t_start = time.time()
    for ep in range(epochs):
        model.train()
        run = 0.0
        for x, y, _, _ in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
        log["loss"].append(run / len(train_dl))
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[{name} f{args.fold}] epoch {ep:3d} loss {log['loss'][-1]:.4f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
    log["train_seconds"] = time.time() - t_start

    probs, ms = predict(model, test_dl, device, tta=cfg.get("tta", False))
    log["infer_ms_per_slice"] = ms
    recs = []
    for i, r in enumerate(test_rows):
        gt = np.array(Image.open(os.path.join(args.data, "masks", r["file"]))) > 0
        gt_c = np.array(Image.open(os.path.join(args.data, "centerlines", r["file"]))) > 0
        m = slice_metrics(probs[i], gt, gt_c)
        recs.append(dict(slice=r["slice"], scan=r["scan"], fold=r["fold"], pos_in_scan=r["pos_in_scan"], **m))
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        wr.writeheader()
        wr.writerows(recs)
    np.savez_compressed(os.path.join(out_dir, "preds.npz"),
                        **{f"slice_{r['slice']:04d}": (probs[i] * 255).astype(np.uint8)
                           for i, r in enumerate(test_rows)})
    for k in ("f1_tol", "cldice", "dice_band", "cl_mean_dist"):
        log[f"mean_{k}"] = float(np.nanmean([r[k] for r in recs]))
    json.dump(log, open(os.path.join(out_dir, "log.json"), "w"), indent=1)
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    print(f"[{name} f{args.fold}] done: F1@2px {log['mean_f1_tol']:.4f} clDice {log['mean_cldice']:.4f} "
          f"Dice(band) {log['mean_dice_band']:.4f} centreline dist {log['mean_cl_mean_dist']:.2f}px "
          f"train {log['train_seconds'] / 60:.1f} min, {ms:.2f} ms/slice", flush=True)


if __name__ == "__main__":
    main()
