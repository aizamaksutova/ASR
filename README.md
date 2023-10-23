# ASR Project
ASR Project within a course 'Deep Learning in Audio'

# Model choice 

For my implementation of ASR, I chose the model from paper [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595.pdf).
Model architecture in the project is heavily dependant on this architecture, though a bit reduced due to resources limitations. 

![Model architecture from paper Deep Speech 2](https://github.com/aizamaksutova/DL_Audio/blob/main/images/model-arch-ds2.png)

# Training pipeline

1. First I am training my model only on LibriSpeech dataset train-clean-100 and train-clean-360 for 50 epochs and testing the model quality on LibriSpeech dataset test-clean. [since the audios are clean i did not add any noise augmentations]
2. Then I am fine-tuning the model on LibriSpeech dataset train-other-500 for 50 epochs and testing its final quality on test-other. [on the fine-tuning step I am adding noise augmentations to upsample the training data and adapt the model to noisy sounds]


# Experiments
 ### First try
 First I implemented a model with 2 convolution layers and 4 GRU layers. Optimizer - SGD, lr_scheduler - OneCycleLR, no augmentations for train-clean part. See the config for the clean-part training [here](https://github.com/aizamaksutova/DL_Audio/blob/main/configs/1exp_train_clean.json). 
 The metrics on this step while training only on clean part were CER: 0.39 and I would like to not state the WER, because I was not computing it in the right way.

 ### Second try
 I decided to carry on with training on the clean part and get better cer and wer metrics on them, so I am adding more GRU layers and augmentations. I decided to take the architecture which is shown in the picture in part 1 [3 convolution layers and 7 GRU layers]. First for 50 epochs I trained on the train-clean datasets adding such augmentations: for waves - [Shift](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/shift.py), [Gain](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/gain.py) and [Guassian Noise](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/add_gaussian_noise.py); for spectrograms: [Frequency masking](https://pytorch.org/audio/main/generated/torchaudio.transforms.FrequencyMasking.html) and [Time masking](https://pytorch.org/audio/main/generated/torchaudio.transforms.TimeMasking.html). Afterwards, I trained the same model for 50 epochs on train-other with the same augmentations, but without adding the Guassian Noise. Configs for training -  [first 50 epochs](https://github.com/aizamaksutova/DL_Audio/blob/main/configs/secondexp_firstiteration.json) and second [50 epochs](https://github.com/aizamaksutova/DL_Audio/blob/main/configs/secondexp_seconditer.json). Logs of training: [first 50 epochs](https://github.com/aizamaksutova/DL_Audio/blob/main/training_logs/second_exp_firstiter_train50.log) and [second 50 epochs](https://github.com/aizamaksutova/DL_Audio/blob/main/training_logs/second_exp_seconditer_train.log). 
 
#### Metrics:
On test-clean were 0.16 CER and 0.45 WER. On test-other I had 0.27 CER and 0.63 WER. [model numbers were first train_ds2/1021_175629 second train_ds2_other/1022_173704]

#### What failed? 
Using TimeInversion Augmentation was a failure, the audio became impossible to listen to and this augmentation was rather vague.

### Third try
 
 I decided to try to train the model for 100 epochs on the whole dataset(clean + other) and see how well that goes. Model architecture was a bit changed since i had to downsize the number of GRU layers and make it only 4 due to enlargening the dataset. This attempt rather failed too [see metrics]. [logs](https://github.com/aizamaksutova/DL_Audio/blob/main/training_logs/third_alltrain.log) and [config](https://github.com/aizamaksutova/DL_Audio/blob/main/configs/train_all.json)

 #### Metrics
 For test-clean: CER 0.71 and WER 0.99. For test-other: CER 0.68 and WER 1.02.

 #### Why failed?
 My hypothesis is that using Guassian noise augmentation with the "other" dataset was an overkill. Also, SGD optimizer might be less efficient than Adam in this case, but I wanted to try the CyclirLR scheduler with triangular2 mode and it is compatible only with SGD.


 
