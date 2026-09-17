import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    """The original 2024 U-Net (segment/unet_pytorch/model.py), kept as the baseline."""

    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[-(i // 2) - 1]
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])
            x = self.ups[i + 1](torch.cat((skip, x), 1))
        return self.final_conv(x)


def build_model(cfg):
    if cfg["model"] == "unet_scratch":
        return Unet(in_channels=3)
    if cfg["model"] == "smp_unet":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name=cfg.get("encoder", "resnet34"),
                        encoder_weights=cfg.get("encoder_weights", "imagenet"),
                        in_channels=3, classes=1)
    raise ValueError(cfg["model"])
