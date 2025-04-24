import torch
import torchaudio
import os
import json

from gpt_sovits.module.denoiser import Denoiser
from gpt_sovits.module.models import BigVGANGenerator


class VocoderBigVGAN:
    def __init__(self, vocoder_dir, device="cpu"):
        self.device = device

        checkpoint_path = os.path.join(vocoder_dir, "bigvgan_generator.pt")
        config_path = os.path.join(vocoder_dir, "config.json")

        print(f"🔧 Loading BigVGAN vocoder from {checkpoint_path}")
        print(f"🔧 Using vocoder config: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        safe_config = {
            "resblock": config.get("resblock", "1"),
            "upsample_rates": config.get("upsample_rates", [8, 8, 2, 2]),
            "upsample_kernel_sizes": config.get("upsample_kernel_sizes", [16, 16, 4, 4]),
            "resblock_kernel_sizes": config.get("resblock_kernel_sizes", [3, 7, 11]),
            "resblock_dilation_sizes": config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
            "upsample_initial_channel": config.get("upsample_initial_channel", 512),
            "use_spectral_norm": config.get("use_spectral_norm", False),
            "sampling_rate": config.get("sampling_rate", 24000),
            "n_fft": config.get("n_fft", 1024),
            "hop_size": config.get("hop_size", 256),
            "win_size": config.get("win_size", 1024),
            "fmin": config.get("fmin", 0),
            "fmax": config.get("fmax", None),
            "num_mels": config.get("num_mels", 100)
        }

        print(f"🧩 Parsed safe_config for vocoder: {safe_config}")

        vocoder = BigVGANGenerator(safe_config)
        vocoder.load_state_dict(torch.load(
            checkpoint_path, map_location=device), strict=False)
        self.vocoder = vocoder.to(device).eval()

        print("✅ BigVGAN model loaded successfully.")

        self.denoiser = Denoiser(self.vocoder).to(device)
        print("✅ Denoiser initialized.")

    def infer(self, mel):
        print(f"🎵 Synthesizing from mel shape: {mel.shape}")
        with torch.no_grad():
            audio = self.vocoder(mel)
            audio = self.denoiser(audio, strength=0.01)[:, 0]
        return audio
