# gpt_sovits/inference_webui.py - 真實語音合成流程整合版

import os
import torch
import torchaudio
import soundfile as sf
from gpt_sovits.sovits import load_sovits_model


def synthesize_speech(reference_audio_path, prompt_text, inference_text, out_dir):
    """
    使用 GPT-SoVITS 模型進行語音合成。
    :param reference_audio_path: 參考語音樣本路徑（wav）
    :param prompt_text: 該語音對應的原始文字內容（用於語者特徵抽取）
    :param inference_text: 想要合成的目標句子
    :param out_dir: 合成語音輸出資料夾
    :return: 合成後音訊檔案完整路徑
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_sovits_model(
        ckpt_path="gpt_sovits/s2Gv3.pth",
        config_path="gpt_sovits/config.json",
        device=device
    )

    # 載入參考語音音訊
    wav_tensor, sr = torchaudio.load(reference_audio_path)
    if sr != 22050:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 22050)

    # 將文字轉為 dummy prompt_ids（簡化處理）
    prompt_ids = torch.arange(0, len(prompt_text), device=device).unsqueeze(0)

    with torch.no_grad():
        pred_semantics = model.extract_semantics(
            wav_tensor[0].to(device), prompt_ids)
        generated_audio = model.infer(pred_semantics)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sovits_out.wav")
    sf.write(out_path, generated_audio, 22050)
    return out_path
