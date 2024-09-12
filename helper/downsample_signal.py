import torch
import torch.nn as nn
import torchaudio.transforms as T


# Custom downsampling class
class DownsampleSignal(nn.Module):
    def __init__(self, orig_sr, target_sr):
        super(DownsampleSignal, self).__init__()
        self.orig_sr = orig_sr
        self.target_sr = target_sr

    def forward(self, waveform):
        # Downsample the waveform
        resampler = T.Resample(orig_freq=self.orig_sr, new_freq=self.target_sr)
        return resampler(waveform)