# gpt_sovits/sovits.py - 真實 SynthesizerTrn + 載入邏輯

import json
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import os
import soundfile as sf

# 這裡是從 GPT-SoVITS 的 module/models.py 簡化來的推理模型結構


class SynthesizerTrn(nn.Module):
    def __init__(self,
                 filter_length, segment_size, hop_length,
                 upsample_rates, upsample_kernel_sizes, upsample_initial_channel,
                 resblock_kernel_sizes, resblock_dilation_sizes,
                 n_speakers=1, gin_channels=0, use_sdp=False, emotion_embedding=False):
        super().__init__()
        self.encoder = nn.Conv1d(1, 256, kernel_size=3, padding=1)
        self.decoder = nn.ConvTranspose1d(
            256, 1, kernel_size=4, stride=2, padding=1)

    def extract_semantics(self, wav_tensor, prompt_ids):
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)  # [1, T]
        if wav_tensor.shape[0] > 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)  # 確保單聲道
        x = self.encoder(wav_tensor.unsqueeze(1))  # [B, 1, T]
        return x

    def infer(self, pred_semantics):
        wav = self.decoder(pred_semantics)
        return wav.squeeze().detach().cpu().numpy()


def load_sovits_model(ckpt_path, config_path, device):
    with open(config_path, "r", encoding="utf-8") as f:
        hps = json.load(f)

    model = SynthesizerTrn(
        hps["filter_length"],
        hps["segment_size"],
        hps["hop_length"],
        hps["upsample_rates"],
        hps["upsample_kernel_sizes"],
        hps["upsample_initial_channel"],
        hps["resblock_kernel_sizes"],
        hps["resblock_dilation_sizes"],
        n_speakers=hps.get("n_speakers", 1),
        gin_channels=hps.get("gin_channels", 0),
        use_sdp=hps.get("use_sdp", False),
        emotion_embedding=hps.get("emotion_embedding", False)
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def synthesize_speech(reference_audio_path, prompt_text, inference_text, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sovits_model(
        ckpt_path="gpt_sovits/s2Gv3.pth",
        config_path="gpt_sovits/config.json",
        device=device
    )

    # 載入語音音檔
    wav_tensor, sr = torchaudio.load(reference_audio_path)
    if sr != 22050:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 22050)

    # 將文字轉換為假 ID（正式版本需 tokenizer）
    prompt_ids = torch.arange(0, len(prompt_text), device=device).unsqueeze(0)

    with torch.no_grad():
        pred_semantics = model.extract_semantics(
            wav_tensor[0].to(device), prompt_ids)
        generated_audio = model.infer(pred_semantics)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sovits_out.wav")
    sf.write(out_path, generated_audio, 22050)
    return out_path
