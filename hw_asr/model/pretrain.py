from unicodedata import bidirectional
from torch import nn
import torch

from hw_asr.base import BaseModel


class DeepSpeech_pretrain(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(11, 41), stride=(2, 2), padding=(5, 20)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(32, 32, kernel_size=(11, 41), stride=(2, 2), padding=(5, 20)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(32, 32, kernel_size=(11, 21), stride=(1, 2), padding=(5, 10)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
        )
        self.normalization_bef = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=n_feats * 32 // 2 // 2 // 2,
            hidden_size=fc_hidden,
            num_layers=6,
            bidirectional=True,
            batch_first=True,
        )

        self.final = nn.Sequential(
            nn.LayerNorm(fc_hidden * 2),
            nn.Linear(fc_hidden * 2, n_class)
        )
    def forward(self, spectrogram, **batch):
        spectrogram = spectrogram.transpose(1, 2).unsqueeze(1)
        feats = self.conv(spectrogram)

        feats = self.normalization_bef(feats)
        batch = feats.size(0)
        n = feats.size(2)

        feats = torch.transpose(feats, 1, 2).reshape((batch, n, -1))

        feats, _ = self.gru(feats)

        logits = self.final(feats)

        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths + 3) // 4