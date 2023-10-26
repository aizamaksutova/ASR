import torch_audiomentations
from torch import Tensor
import librosa
import torchaudio
import torch

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.wave_augmentations.noise import get_noise_sample


class AddBackgroundNoise(AugmentationBase):
    def __init__(self, p, *args, **kwargs):
        self.p = p

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        noise, _ = get_noise_sample(resample=16000)
        noise = noise[:, : x.shape[1]]
        snr_dbs = torch.tensor([40, 10, 3])
        return  (x + noise).squeeze(1)