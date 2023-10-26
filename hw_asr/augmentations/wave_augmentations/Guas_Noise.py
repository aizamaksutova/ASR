import torch_audiomentations
from torch import Tensor
from torch import distributions
import numpy as np

from hw_asr.augmentations.base import AugmentationBase


class Guassian_Noise(AugmentationBase):
    def __init__(self, p, *args, **kwargs):
        self.noiser = distributions.Normal(0, 0.003)
        self.p = p

    def __call__(self, data: Tensor):
        if np.random.rand() < self.p:
            x = data.unsqueeze(1)
            return (x + self.noiser.sample(x.size())).squeeze(1)
        return data