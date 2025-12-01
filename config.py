import os
import sys
from pathlib import Path
import torch

# 專案根目錄
BASE_DIR = Path(__file__).parent.absolute()

class Config:
    # --- 路徑設定 ---
    GPT_SOVITS_DIR = BASE_DIR / "gpt_sovits"
    PRETRAINED_MODELS_DIR = GPT_SOVITS_DIR / "pretrained_models"
    
    # 模型檔案路徑
    SOVITS_PATH = PRETRAINED_MODELS_DIR / "s2Gv3.pth"
    GPT_PATH = PRETRAINED_MODELS_DIR / "s1v3.ckpt"
    CONFIG_PATH = PRETRAINED_MODELS_DIR / "config.json"
    
    # Vocoder 路徑
    VOCODER_DIR = PRETRAINED_MODELS_DIR / "models--nvidia--bigvgan_v2_24khz_100band_256x"
    
    # --- 參數設定 ---
    # 自動偵測裝置: CUDA > CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Whisper 設定
    WHISPER_MODEL_SIZE = "small"
    WHISPER_COMPUTE_TYPE = "int8"
    
    def validate(self):
        """檢查必要檔案是否存在，若缺少則拋出錯誤"""
        missing = []
        if not self.SOVITS_PATH.exists(): missing.append(f"SoVITS Model: {self.SOVITS_PATH}")
        if not self.GPT_PATH.exists(): missing.append(f"GPT Model: {self.GPT_PATH}")
        if not self.CONFIG_PATH.exists(): missing.append(f"Config JSON: {self.CONFIG_PATH}")
        if not self.VOCODER_DIR.exists(): missing.append(f"Vocoder Dir: {self.VOCODER_DIR}")
        
        if missing:
            error_msg = "❌ 缺少必要模型檔案:\n" + "\n".join(missing)
            raise FileNotFoundError(error_msg)

# 建立全域配置實例
conf = Config()