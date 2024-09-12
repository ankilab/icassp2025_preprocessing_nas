import torch.nn as nn
import torchaudio.transforms as T

from helper.pad_waveform import PadWaveform
from helper.stack_input import StackSingleChannelToThreeChannels
from helper.downsample_signal import DownsampleSignal
from helper.wavelet_transform import WaveletTransform

def get_pre_processing_transform(params: dict, orig_sr: int, sample_length: int) -> nn.Sequential:
    """
    Returns a transform that applies the preprocessing steps specified by the given
    parameters. The transform can be applied to an audio signal to prepare it for
    input to a neural network.

    Args:
        params (dict): Dictionary containing the preprocessing parameters.
        orig_sr (int): Original sampling rate of the audio signals.
        sample_length (int): Length of the audio signals in seconds.

    Returns:
        nn.Sequential: A PyTorch Sequential module that applies the specified preprocessing
            steps to an audio signal.
    """
    transforms = []
    if orig_sr != params["sample_rate"]:
        transforms.append(DownsampleSignal(orig_sr, params["sample_rate"]))

    transforms.append(PadWaveform(target_length=sample_length * params["sample_rate"]))

    if params["method"] == "stft":
        transforms.append(get_stft_transform(params))
    elif params["method"] == "mel":
        transforms.append(get_mel_transform(params))
    elif params["method"] == "mfcc":
        transforms.append(get_mfcc_transform(params))
    elif params["method"] == "wavelet":
        transforms.append(get_wavelet_transform(params))
    else:
        raise ValueError(f"Method {params['method']} not supported")
    
    if params["use_db"] and params["method"] != "mfcc": # MFCC already returns db-scaled values
        transforms.append(T.AmplitudeToDB())

    transforms.append(StackSingleChannelToThreeChannels())  # Stack single channel to three channels as needed by MobileNetV2/MobileNetV3
    return nn.Sequential(*transforms)
    

def get_stft_transform(params: dict) -> nn.Sequential:
    return T.Spectrogram(
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        power=params["stft_power"]
    )

def get_mel_transform(params: dict) -> nn.Sequential:
    return T.MelSpectrogram(
        sample_rate=params["sample_rate"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        n_mels=params["n_mels"],
        power=params["stft_power"]
    )

def get_mfcc_transform(params: dict) -> nn.Sequential:
    """
    Returns a transform that computes the Mel-frequency cepstral coefficients (MFCC) of an audio signal. Signal is already db-scaled.
    """
    # Check if n_mels >= n_mfcc as required by torchaudio.transforms.MFCC
    if params["n_mels"] < params["n_mfcc"]:
        # we just set n_mels to n_mfcc
        params["n_mels"] = params["n_mfcc"]

    return T.MFCC(
        sample_rate=params["sample_rate"],
        n_mfcc=params["n_mfcc"],
        melkwargs={
            "n_fft": params["n_fft"],
            "hop_length": params["hop_length"],
            "n_mels": params["n_mels"],
            "power": params["stft_power"]
        }
    )

def get_wavelet_transform(params: dict) -> nn.Sequential:
    return WaveletTransform(params['wavelet_scaling'], params['sample_rate'], params['wavelet_resize'])
