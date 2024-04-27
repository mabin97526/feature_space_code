import random

import torch as th
import torch.nn as nn
from PIL import ImageFilter, ImageOps
from torchvision import transforms as T
class Transform(nn.Module):
    def __init__(self, transform):
        super().__init__()

        self.transform = transform

    def forward(self, x: th.Tensor) -> th.Tensor:
        if len(x.shape) >= 4:
            batch = x.shape[:-3]
            x = x.reshape(-1, *x.shape[-3:])
            x = x.permute([0, 3, 1, 2])
            x = self.transform(x).permute([0, 2, 3, 1])
            return x.reshape(*batch, *x.shape[1:])
        else:
            raise Exception('The dimension of input should be greater than or equal to 4')


class GaussianNoise:
    def __init__(self, mean=0., std=.1):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        is_torch = isinstance(img, th.Tensor)

        if not is_torch:
            img = T.ToTensor()(img)

        img = th.clamp(img + th.rand(img.shape, dtype=th.float32, device=img.device) * self.std + self.mean, 0., 1.)

        if not is_torch:
            return T.ToPILImage()(img)
        else:
            return img


class SaltAndPepperNoise:
    """
    Args:
        snr (float): Signal Noise Rate
    """

    def __init__(self, snr=.3, p=.9):
        self.snr = snr
        self.p = p

    def __call__(self, img):
        is_torch = isinstance(img, th.Tensor)

        if not is_torch:
            img = T.ToTensor()(img)

        batch, c, h, w = img.shape

        signal_p = self.p
        noise_p = (1 - self.p)

        mask = th.rand((batch, 1, h, w), dtype=th.float32, device=img.device).repeat(1, c, 1, 1)

        img = th.where(mask < noise_p / 2., th.clamp(img + self.snr, 0., 1.), img)
        img = th.where(mask > noise_p / 2. + signal_p, th.clamp(img - self.snr, 0., 1.), img)

        if not is_torch:
            return T.ToPILImage()(img)
        else:
            return img


class DepthNoise(object):
    """
    Args:
        p (float): Signal Noise Rate
    """

    def __init__(self, p):
        if isinstance(p, tuple):
            self.p = p
        else:
            self.p = (-p, p)

    def __call__(self, img):
        is_torch = isinstance(img, th.Tensor)

        if not is_torch:
            img = T.ToTensor()(img)

        noise = th.rand(1, dtype=th.float32, device=img.device) * (self.p[1] - self.p[0]) + self.p[0]
        img = (img + noise).clip(0., 1.)

        if not is_torch:
            return T.ToPILImage()(img)
        else:
            return img


class DepthSaltAndPepperNoise(object):
    """
    Args:
        snr (float): Signal Noise Rate
    """

    def __init__(self, snr=1., p=0.03):
        self.snr = snr
        self.p = p

    def __call__(self, img):
        is_torch = isinstance(img, th.Tensor)

        if not is_torch:
            img = T.ToTensor()(img)

        batch, c, h, w = img.shape

        noise_p = self.p
        signal_p = (1 - self.p)

        mask = th.rand((batch, c, h, w), dtype=th.float, device=img.device)

        img = th.where(mask < noise_p / 2., th.clamp(img + th.tensor(self.snr), 0., 1.), img)
        img = th.where(mask > noise_p / 2. + signal_p, th.clamp(img - th.tensor(self.snr), 0., 1.), img)

        if not is_torch:
            return T.ToPILImage()(img)
        else:
            return img

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
