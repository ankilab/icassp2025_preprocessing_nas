import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import fcwt
from torch.nn.functional import interpolate



class WaveletTransform(nn.Module):
    def __init__(self, wavelet_scaling, sampling_rate, wavelet_resize):
        super(WaveletTransform, self).__init__()
        self.wavelet_scaling = wavelet_scaling
        self.sampling_rate = sampling_rate
        self.wavelet_resize = wavelet_resize

    def __call__(self, x):
        # x is a 1D tensor (audio signal)
        x_np = x.numpy()

        _, coeffs = fcwt.cwt(x_np[0], fs=self.sampling_rate, f0=1, f1=self.sampling_rate // 2, fn=32, scaling=self.wavelet_scaling)
        coeffs = np.abs(coeffs)

        coeffs = torch.tensor(coeffs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Resize the wavelet coefficients
        coeffs = interpolate(coeffs, size=[self.wavelet_resize, self.wavelet_resize], mode='bilinear')

        coeffs = coeffs.squeeze(0).squeeze(0)

        return coeffs

    def __repr__(self):
        return self.__class__.__name__ + f'(wavelet={self.wavelet})'