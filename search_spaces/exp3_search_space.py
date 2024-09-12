import nni
from nni.nas.hub.pytorch import MobileNetV3Space

from nni.mutable import MutableExpression

import torch.nn as nn
import torchaudio.transforms as T


class Experiment3SearchSpace(MobileNetV3Space):
    """
    Experiment Nr. 3 consists of a search space that optimizes the preprocessing parameters
    and the model architecture.
    """
    def __init__(self, **kwargs):
        orig_sr = kwargs.pop('orig_sr', 16000)
        super().__init__(**kwargs)

        candidate_sampling_rates = [4000, 6000, 8000, 11000, 16000, 22050, 44100]
        # remove sampling rates that are higher than the original sampling rate
        candidate_sampling_rates = [sr for sr in candidate_sampling_rates if sr <= orig_sr]

        self.add_mutable(nni.choice('method', ["mel", "mfcc", "wavelet"]))
        self.add_mutable(nni.choice('sample_rate', candidate_sampling_rates))
        self.add_mutable(nni.choice('use_db', [False, True]))

        self.add_mutable(nni.choice('n_fft', [128, 256, 512, 1024, 2048]))
        self.add_mutable(nni.choice('hop_length', [32, 64, 128, 256, 512]))
        self.add_mutable(nni.choice('n_mels', [20, 30, 40, 60, 80, 100, 128]))
        self.add_mutable(nni.choice('stft_power', [1, 2]))
        self.add_mutable(nni.choice('n_mfcc', [12, 13, 20, 30, 40]))

        self.add_mutable(nni.choice('wavelet_scaling', ["lin", "log"]))
        self.add_mutable(nni.choice('wavelet_resize', [64, 128]))


    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def freeze(self, sample):
        with nni.nas.space.model_context(sample):
            return self.__class__()  # Return a new instance with the frozen context