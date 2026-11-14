"""The network trained on OUR neutral picture -> the person's final (scripts/lr_neutral_model.py trains it and keeps the training): a ResNet-18 on a 128 px thumb of the neutral picture plus the picture's features, predicting the 19 parameters of the transform (look/transform.py)."""

import threading

import cv2
import numpy as np
import torch

from stratastills.look.features import THUMB
from stratastills.look.transform import NP


class Net(torch.nn.Module):
    """ResNet-18 (ImageNet weights) on the thumb(s), its pooled features joined with the standardised scalar features, two layers to the 19 parameters, started at the person's typical transform."""

    def __init__(self, n_scalar, typical, n_img=1, pretrained=True):
        super().__init__()
        import torchvision

        self.cnn = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        )  # inference (load) replaces every weight from the checkpoint: no ImageNet download there
        w = self.cnn.conv1.weight.data
        if n_img > 1:
            self.cnn.conv1 = torch.nn.Conv2d(3 * n_img, 64, 7, 2, 3, bias=False)
            self.cnn.conv1.weight.data = torch.cat([w / n_img] * n_img, 1)
        self.cnn.fc = torch.nn.Identity()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 + n_scalar, 256), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, NP)
        )
        torch.nn.init.zeros_(self.head[-1].weight)
        self.head[-1].bias.data = torch.as_tensor(typical, dtype=torch.float32)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406] * n_img).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225] * n_img).view(1, -1, 1, 1))

    def forward(self, img, x):  # img: (N, 3*n_img, H, W) RGB 0..1
        return self.head(torch.cat([self.cnn((img - self.mean) / self.std), x], 1))


_lock = (
    threading.Lock()
)  # one picture at a time through a loaded network (two threads at once gave flat, NaN-filled pictures in build_arena)


def load(path, dev):
    """The saved network (`lr_neutral_model.py eval --save`): {net, mu, sd} on `dev`."""
    z = torch.load(path, map_location=dev, weights_only=False)
    net = Net(len(z["mu"]), z["typical"], pretrained=False).to(dev)
    net.load_state_dict(z["state"])
    net.eval()
    return dict(net=net, mu=z["mu"], sd=z["sd"], dev=dev)


def predict_params(m, neu_bgr, x, lo, hi):
    """The 19 transform parameters for our neutral picture `neu_bgr` (uint8 BGR) with feature row `x`, each held to [lo, hi] (the range the training fits span, 0.5-99.5 %: an unusual picture can otherwise get a transform no edit ever had)."""
    with _lock:
        t = cv2.resize(neu_bgr, (THUMB, THUMB), interpolation=cv2.INTER_AREA)[..., ::-1].copy()
        img = torch.from_numpy(t).permute(2, 0, 1)[None].float().to(m["dev"]) / 255
        xs = torch.from_numpy(
            np.nan_to_num((np.asarray(x, np.float64) - m["mu"]) / m["sd"]).clip(-5, 5).astype(np.float32)
        )[None].to(m["dev"])
        with torch.no_grad():
            out = m["net"](img, xs)[0].cpu().numpy()
    return np.clip(out, lo, hi)
