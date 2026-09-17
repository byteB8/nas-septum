"""Metrics for a thin, curvilinear target.

The annotation is a 1-px centreline, so overlap scores on a 3-px band (Dice) are
dominated by 1-px offsets. The primary scores compare centrelines directly:
tolerance F1 (share of centreline within TOL px of the other), mean and 95th
percentile centreline distance, clDice, and the error in septal deviation.
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

TOL = 2.0  # px


def largest_component(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def dice(pred, gt):
    s = pred.sum() + gt.sum()
    return 2 * np.logical_and(pred, gt).sum() / s if s else 1.0


def centerline_scores(pred_c, gt_c, tol=TOL):
    """pred_c, gt_c: boolean 1-px centrelines."""
    if pred_c.sum() == 0 or gt_c.sum() == 0:
        return dict(f1_tol=0.0, prec_tol=0.0, rec_tol=0.0, cl_mean_dist=np.nan, cl_hd95=np.nan)
    d_to_g = ndimage.distance_transform_edt(~gt_c)[pred_c]
    d_to_p = ndimage.distance_transform_edt(~pred_c)[gt_c]
    prec, rec = float((d_to_g <= tol).mean()), float((d_to_p <= tol).mean())
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(f1_tol=f1, prec_tol=prec, rec_tol=rec,
                cl_mean_dist=float(np.concatenate([d_to_g, d_to_p]).mean()),
                cl_hd95=float(max(np.percentile(d_to_g, 95), np.percentile(d_to_p, 95))))


def cldice(pred, gt, pred_c, gt_c):
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    tprec = (pred_c & gt).sum() / max(pred_c.sum(), 1)
    tsens = (gt_c & pred).sum() / max(gt_c.sum(), 1)
    return float(2 * tprec * tsens / (tprec + tsens)) if tprec + tsens else 0.0


def septal_deviation(centerline, min_len=5, smooth=5):
    """Deviation of the septum centreline from a straight septum.

    The centreline is reduced to one x per row and smoothed over `smooth` rows,
    so hand-drawn jitter in the annotation does not read as deviation. The chord
    joining its top and bottom points is where a straight septum would run.
    Returns the largest perpendicular distance from that chord (px), that
    distance as % of chord length, and the chord length.
    """
    pts = np.argwhere(centerline)  # (y, x)
    if len(pts) < min_len:
        return np.nan, np.nan, np.nan
    ys = np.unique(pts[:, 0])
    xs = np.array([pts[pts[:, 0] == y, 1].mean() for y in ys])
    if smooth > 1 and len(xs) >= smooth:
        xs = ndimage.uniform_filter1d(xs, smooth, mode="nearest")
    pts = np.stack([ys, xs], 1).astype(float)
    top, bot = pts[np.argmin(pts[:, 0])], pts[np.argmax(pts[:, 0])]
    v = (bot - top).astype(float)
    length = float(np.hypot(*v))
    if length < min_len:
        return np.nan, np.nan, length
    d = np.abs(v[1] * (pts[:, 0] - top[0]) - v[0] * (pts[:, 1] - top[1])) / length
    return float(d.max()), 100.0 * float(d.max()) / length, length


def slice_metrics(prob, gt_band, gt_c, thr=0.5):
    raw = prob > thr
    pred = largest_component(raw) if raw.any() else raw
    pred_c = skeletonize(pred) if pred.any() else pred
    dev_p, ratio_p, _ = septal_deviation(pred_c)
    dev_g, ratio_g, len_g = septal_deviation(gt_c)
    return dict(detected=int(pred.any()), n_components=int(ndimage.label(raw, np.ones((3, 3)))[1]),
                dice_band=float(dice(pred, gt_band)), cldice=cldice(pred, gt_band, pred_c, gt_c),
                **centerline_scores(pred_c, gt_c),
                dev_pred=dev_p, dev_gt=dev_g, dev_ratio_pred=ratio_p, dev_ratio_gt=ratio_g,
                septum_len_gt=len_g)
