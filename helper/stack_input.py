import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Custom transform class to stack grayscale to RGB
class StackSingleChannelToThreeChannels(nn.Module):
    def forward(self, x):
        # x shape: [1, height, width]
        return x.repeat(3, 1, 1)  # Repeat the single channel across three channels