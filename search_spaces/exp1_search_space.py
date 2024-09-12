import nni
from nni.nas.hub.pytorch import MobileNetV3Space

from nni.mutable import MutableExpression

import torch.nn as nn
import torchaudio.transforms as T


class Experiment1SearchSpace(MobileNetV3Space):
    def __init__(self, **kwargs):
        orig_sr = kwargs.pop('orig_sr', 16000)
        super().__init__(**kwargs)

        # Experiment Nr. 1 consists of fixed preprocessing parameters   
        self.add_mutable(nni.choice('method', ["mel"]))
        self.add_mutable(nni.choice('n_fft', [1024]))
        self.add_mutable(nni.choice('hop_length', [512]))
        self.add_mutable(nni.choice('n_mels', [40]))
        self.add_mutable(nni.choice('use_db', [True]))
        self.add_mutable(nni.choice('sample_rate', [orig_sr]))
        self.add_mutable(nni.choice('stft_power', [1]))


    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def freeze(self, sample):
        with nni.nas.space.model_context(sample):
            return self.__class__()  # Return a new instance with the frozen context