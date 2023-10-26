import torchaudio.transforms
from torch import Tensor
import numpy as np 
from torch import nn
import torch

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, p, *args, **kwargs):
        self.aug = torchaudio.transforms.TimeMasking(100)
        self.p = p

    def __call__(self, data: Tensor):
        if np.random.rand() < self.p:
            x = data.unsqueeze(2)
            return self.aug(x).squeeze(2)
        return data
        