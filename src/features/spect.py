import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchaudio
import torchaudio.transforms as T
from omegaconf import DictConfig

class Spectrogram(nn.Module):
    def __init__(
        self,
        conf: DictConfig
    ) -> None:
        super().__init__()
        self.n_fft = conf.feat.n_fft
        self.hop = conf.feat.hop
        self.mel = conf.feat.mel
        self.n_mels = conf.feat.n_mels

    def forward(self, audio: Tensor, sr: int) -> Tensor:
        if self.mel:
            spect = T.MelSpectrogram(
                sample_rate = sr,
                n_fft = self.n_fft,
                hop_length = int(self.hop * sr),
                n_mels = self.n_mels,
                window_fn=(torch.hamming_window),
                power=1.0
            )(audio)
        else:
            spect = T.Spectrogram(
                n_fft = self.n_fft,
                hop_length = int(self.hop * sr),
                window_fn = torch.hamming_window,
                power=1.0
            )(audio)

        return spect