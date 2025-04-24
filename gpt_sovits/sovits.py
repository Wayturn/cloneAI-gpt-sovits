
import json
import torch
import torchaudio
import os
import soundfile as sf
from gpt_sovits.module.models import SynthesizerTrn
from gpt_sovits.module.vocoder import VocoderBigVGAN


def load_sovits_model(ckpt_path, config_path, device):
    print(f"🔍 Loading SoVITS from: {ckpt_path}")
    print(f"🔍 Using config: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            hps = json.load(f)
    except Exception as e:
        raise RuntimeError(f"讀取 config.json 失敗: {e}")

    model = SynthesizerTrn(
        spec_channels=hps["data"]["n_mel_channels"],
        segment_size=hps["train"]["segment_size"],
        inter_channels=hps["model"]["inter_channels"],
        hidden_channels=hps["model"]["hidden_channels"],
        filter_channels=hps["model"]["filter_channels"],
        n_heads=hps["model"]["n_heads"],
        n_layers=hps["model"]["n_layers"],
        kernel_size=hps["model"]["kernel_size"],
        p_dropout=hps["model"]["p_dropout"],
        resblock=hps["model"]["resblock"],
        resblock_kernel_sizes=hps["model"]["resblock_kernel_sizes"],
        resblock_dilation_sizes=hps["model"]["resblock_dilation_sizes"],
        upsample_rates=hps["model"]["upsample_rates"],
        upsample_initial_channel=hps["model"]["upsample_initial_channel"],
        upsample_kernel_sizes=hps["model"]["upsample_kernel_sizes"],
        gin_channels=hps["model"]["gin_channels"],
        use_sdp=hps["model"].get("use_sdp", False),
        semantic_frame_rate=hps["model"]["semantic_frame_rate"],
        freeze_quantizer=hps["model"].get("freeze_quantizer", True),
        version="v3"
    ).to(device)

    try:
        model.load_state_dict(torch.load(
            ckpt_path, map_location=device), strict=False)
    except Exception as e:
        raise RuntimeError(f"模型參數載入失敗: {e}")

    model.eval()
    print("✅ SoVITS loaded successfully.")
    return model, hps


def load_gpt_model(model_path, config_path, device):
    print(f"🔍 Loading GPT from: {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        hps = json.load(f)

    model = SynthesizerTrn(
        spec_channels=hps["data"]["n_mel_channels"],
        segment_size=hps["train"]["segment_size"],
        inter_channels=hps["model"]["inter_channels"],
        hidden_channels=hps["model"]["hidden_channels"],
        filter_channels=hps["model"]["filter_channels"],
        n_heads=hps["model"]["n_heads"],
        n_layers=hps["model"]["n_layers"],
        kernel_size=hps["model"]["kernel_size"],
        p_dropout=hps["model"]["p_dropout"],
        resblock=hps["model"]["resblock"],
        resblock_kernel_sizes=hps["model"]["resblock_kernel_sizes"],
        resblock_dilation_sizes=hps["model"]["resblock_dilation_sizes"],
        upsample_rates=hps["model"]["upsample_rates"],
        upsample_initial_channel=hps["model"]["upsample_initial_channel"],
        upsample_kernel_sizes=hps["model"]["upsample_kernel_sizes"],
        gin_channels=hps["model"]["gin_channels"],
        use_sdp=hps["model"].get("use_sdp", False),
        semantic_frame_rate=hps["model"]["semantic_frame_rate"],
        freeze_quantizer=hps["model"].get("freeze_quantizer", True),
        version="v3"
    ).to(device)

    model.load_state_dict(torch.load(
        model_path, map_location=device), strict=False)
    model.eval()
    print("✅ GPT model loaded successfully.")
    return model
