import csv
import json
import os

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PAD = 272  # >= largest crop (183 x 257), divisible by 16


def read_index(data_dir):
    with open(os.path.join(data_dir, "slices.csv")) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("slice", "scan", "fold", "pos_in_scan", "labeled", "fg_px", "h", "w"):
            r[k] = int(r[k])
    with open(os.path.join(data_dir, "stats.json")) as fh:
        stats = json.load(fh)
    return rows, stats


def pad_to(arr, size=PAD):
    """Zero-pad (H, W, ...) to size x size, centred. Returns array and (top, left)."""
    h, w = arr.shape[:2]
    top, left = (size - h) // 2, (size - w) // 2
    out = np.zeros((size, size) + arr.shape[2:], dtype=arr.dtype)
    out[top:top + h, left:left + w] = arr
    return out, (top, left)


def train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    ])


class SeptumDataset(Dataset):
    """Native-resolution slices, zero-padded to PAD x PAD.

    context=1 gives the slice repeated to 3 channels (for ImageNet encoders);
    context=3 stacks previous / current / next slice of the same scan (2.5D).
    """

    def __init__(self, data_dir, rows, stats, context=1, augment=False, mask_dir="masks"):
        self.data_dir, self.rows, self.context, self.mask_dir = data_dir, rows, context, mask_dir
        self.mean, self.std = stats["mean"], stats["std"]
        self.tf = train_transform() if augment else None
        self._cache = {}

    def _img(self, fname):
        if fname not in self._cache:
            self._cache[fname] = np.array(Image.open(os.path.join(self.data_dir, "images", fname)))
        return self._cache[fname]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.context == 3:
            img = np.stack([self._img(r["prev_file"]), self._img(r["file"]), self._img(r["next_file"])], -1)
        else:
            img = np.repeat(self._img(r["file"])[..., None], 3, -1)
        mask = (np.array(Image.open(os.path.join(self.data_dir, self.mask_dir, r["file"]))) > 0).astype(np.uint8)
        img, (top, left) = pad_to(img)
        mask, _ = pad_to(mask)
        if self.tf is not None:
            out = self.tf(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()),
                torch.from_numpy(mask[None].astype(np.float32)),
                torch.tensor([top, left, r["h"], r["w"]]),
                i)
