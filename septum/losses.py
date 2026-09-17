import torch
import torch.nn.functional as F


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


def _soft_erode(x):
    return torch.min(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def _soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, iters):
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021)."""
    skel = F.relu(x - _soft_dilate(_soft_erode(x)))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_dilate(_soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(logits, target, iters=3, eps=1.0):
    p = torch.sigmoid(logits)
    sp, st = soft_skeleton(p, iters), soft_skeleton(target, iters)
    tprec = ((sp * target).sum((1, 2, 3)) + eps) / (sp.sum((1, 2, 3)) + eps)
    tsens = ((st * p).sum((1, 2, 3)) + eps) / (st.sum((1, 2, 3)) + eps)
    return 1 - (2 * tprec * tsens / (tprec + tsens)).mean()


def build_loss(cfg):
    name = cfg["loss"]
    w_cl = cfg.get("cldice_weight", 0.5)

    def loss_fn(logits, target):
        logits = logits.float()
        loss = F.binary_cross_entropy_with_logits(logits, target)
        if name in ("bce_dice", "bce_dice_cldice"):
            loss = loss + soft_dice_loss(logits, target)
        if name == "bce_dice_cldice":
            loss = loss + w_cl * soft_cldice_loss(logits, target)
        return loss

    assert name in ("bce", "bce_dice", "bce_dice_cldice"), name
    return loss_fn
