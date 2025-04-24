# gpt_sovits/inference_webui.py - 真實語音合成流程整合強化版

import os
import torch
import librosa
import soundfile as sf
from transformers import BertTokenizer
from gpt_sovits.sovits import load_sovits_model, load_gpt_model
from gpt_sovits.module.vocoder import VocoderBigVGAN

# ✅ 裝置偵測
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ✅ 全域變數
g_sovits = None
gpt_model = None
tokenizer = None
vocoder = None


def load_models():
    global g_sovits, gpt_model, tokenizer, vocoder

    if g_sovits and gpt_model and tokenizer and vocoder:
        return

    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
    PRETRAINED_ROOT = os.path.join(MODEL_DIR, "pretrained_models")
    VOCODER_DIR = os.path.join(
        PRETRAINED_ROOT, "models--nvidia--bigvgan_v2_24khz_100band_256x")

    SOVITS_PATH = os.path.join(PRETRAINED_ROOT, "s2Gv3.pth")
    CONFIG_PATH = os.path.join(PRETRAINED_ROOT, "config.json")
    GPT_PATH = os.path.join(PRETRAINED_ROOT, "s1v3.ckpt")

    assert os.path.exists(SOVITS_PATH), f"❌ 找不到 SoVITS 模型檔案：{SOVITS_PATH}"
    assert os.path.exists(CONFIG_PATH), f"❌ 找不到 Config：{CONFIG_PATH}"
    assert os.path.exists(GPT_PATH), f"❌ 找不到 GPT 模型檔案：{GPT_PATH}"

    print(f"🔍 Loading SoVITS from: {SOVITS_PATH}")
    print(f"🔧 Using config: {CONFIG_PATH}")
    g_sovits, _ = load_sovits_model(SOVITS_PATH, CONFIG_PATH, DEVICE)
    print("✅ SoVITS loaded successfully.")

    print(f"🔍 Loading GPT from: {GPT_PATH}")
    gpt_model = load_gpt_model(GPT_PATH, CONFIG_PATH, DEVICE)
    print("✅ GPT model loaded successfully.")

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    vocoder = VocoderBigVGAN(vocoder_dir=VOCODER_DIR, device=DEVICE)
    print("✅ Vocoder loaded successfully.")


def synthesize_speech(reference_audio_path, prompt_text, inference_text, out_dir):
    load_models()

    # 載入語音並轉成 tensor
    print(f"🎧 Loading reference audio: {reference_audio_path}")
    wav, sr = librosa.load(reference_audio_path, sr=24000)
    if len(wav.shape) > 1:
        wav = librosa.to_mono(wav)
    wav_tensor = torch.tensor(wav).unsqueeze(0).to(DEVICE)
    print(f"🎚️ Wav tensor shape: {wav_tensor.shape}")

    # 文字轉 token
    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                           padding=True, truncation=True).input_ids.to(DEVICE)
    inference_ids = tokenizer(inference_text, return_tensors="pt",
                              padding=True, truncation=True).input_ids.to(DEVICE)
    print(f"📝 Prompt text: {prompt_text}")
    print(f"📝 Inference text: {inference_text}")
    print(f"🔡 Inference token shape: {inference_ids.shape}")

    try:
        with torch.no_grad():
            prompt_semantics = g_sovits.extract_semantics(
                wav_tensor, prompt_ids)
            print(
                f"📦 Extracted prompt semantics: {prompt_semantics.shape if hasattr(prompt_semantics, 'shape') else type(prompt_semantics)}")

            pred_semantics = gpt_model.generate(
                prompt_semantics, inference_ids)
            print(
                f"📦 Generated pred_semantics: {pred_semantics.shape if hasattr(pred_semantics, 'shape') else type(pred_semantics)}")

            y_lengths = torch.LongTensor([wav_tensor.size(1)]).to(DEVICE)
            text_lengths = torch.LongTensor([inference_ids.size(1)]).to(DEVICE)

            mel, *_ = g_sovits.infer(
                ssl=pred_semantics,
                y=wav_tensor,
                y_lengths=y_lengths,
                text=inference_ids,
                text_lengths=text_lengths
            )
            print(f"🎛️ Mel shape: {mel.shape}")

            audio = vocoder.infer(mel).squeeze().cpu().numpy()
            print(f"🔊 Generated audio shape: {audio.shape}")

    except Exception as e:
        print(f"❌ 合成失敗: {e}")
        raise

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sovits_out.wav")
    sf.write(out_path, audio, 24000)
    print(f"✅ 合成完成，輸出至: {out_path}")

    return out_path
