import logging
import torch
from typing import List
import torch.nn.utils.rnn as F
from hw_asr.base.base_text_encoder import BaseTextEncoder

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    spectrograms = []
    audio_paths = []
    text = []
    text_encoded = []
    text_encoded_length = []
    spec_length = []
    max_sz_audio = 0

    for item in dataset_items:
        # spectrograms.append(item['spectrogram'].transpose(1, 2))
        text.append(BaseTextEncoder.normalize_text(item['text']))
        text_encoded.append(item['text_encoded'].squeeze(0).T)
        spectrograms.append(item['spectrogram'].squeeze(0).T)
        audio_paths.append(item['audio_path'])
        
    max_audio_size = max([item["audio"].size(1) for item in dataset_items])
    result_batch["audio"] = torch.cat([
        torch.nn.functional.pad(item["audio"], (0, max_audio_size - item["audio"].size(1)))
        for item in dataset_items
    ])

    result_batch["text_encoded_length"] = torch.tensor([item["text_encoded"].size(1) for item in dataset_items])
    result_batch["spectrogram_length"] = torch.tensor([item["spectrogram"].size(2) for item in dataset_items])

    result_batch['text'] = text
    result_batch['text_encoded'] = (F.pad_sequence(text_encoded).T).int()
    result_batch['spectrogram'] = F.pad_sequence(spectrograms, batch_first=True).transpose(1, 2)
    result_batch['audio_path'] = audio_paths

    
    return result_batch
