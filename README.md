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
 I decided to carry on with training on the clean part and get better cer and wer metrics on them, so I am adding more GRU layers and augmentations. 
 
