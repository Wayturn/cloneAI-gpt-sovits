# gpt_sovits/module/denoiser.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Denoiser(nn.Module):
    def __init__(self, vocoder, filter_length=1024, n_overlap=256, win_length=1024, mode='zeros'):
        super(Denoiser, self).__init__()
        self.stft = vocoder.stft
        self.vocoder = vocoder
        self.mode = mode
        mel_input = torch.zeros(
            (1, 80, 88), device=next(vocoder.parameters()).device)
        bias_audio = self.vocoder.infer(mel_input)
        bias_spec, _ = self.stft.transform(bias_audio)
        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_phase = self.stft.transform(audio)
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        return self.stft.inverse(audio_spec_denoised, audio_phase)
