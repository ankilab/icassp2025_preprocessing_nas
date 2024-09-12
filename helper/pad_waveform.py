import torch
import torch.nn as nn
import torchaudio.transforms as T

# Custom padding class
class PadWaveform(nn.Module):
    def __init__(self, target_length=16000):
        super(PadWaveform, self).__init__()
        self.target_length = target_length

    def forward(self, waveform):
        # Get the length of the input waveform
        current_length = waveform.shape[-1]

        # Check if padding is needed
        if current_length >= self.target_length:
            # If the current waveform is longer than or equal to the target length, trim it
            return waveform[..., :self.target_length]
        else:
            # Calculate the padding needed
            padding_needed = self.target_length - current_length
            # Pad the waveform with zeros
            padded_waveform = torch.nn.functional.pad(waveform, (0, padding_needed), mode='constant', value=0)
            return padded_waveform