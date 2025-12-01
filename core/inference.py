import os
import torch
import librosa
import numpy as np
from transformers import BertTokenizer
from config import conf
from gpt_sovits.sovits import load_sovits_model, load_gpt_model
from gpt_sovits.module.vocoder import VocoderBigVGAN

class InferenceEngine:
    def __init__(self):
        self.sovits_model = None
        self.gpt_model = None
        self.tokenizer = None
        self.vocoder = None
        self.is_loaded = False

    def load_models(self):
        if self.is_loaded:
            return

        print("🚀 Initializing Inference Engine...")
        conf.validate()

        # 載入 SoVITS
        self.sovits_model, _ = load_sovits_model(conf.SOVITS_PATH, conf.CONFIG_PATH, conf.DEVICE)
        
        # 載入 GPT
        self.gpt_model = load_gpt_model(conf.GPT_PATH, conf.CONFIG_PATH, conf.DEVICE)
        
        # 載入 Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        
        # 載入 Vocoder
        self.vocoder = VocoderBigVGAN(vocoder_dir=conf.VOCODER_DIR, device=conf.DEVICE)
        
        self.is_loaded = True
        print("✅ All models loaded successfully.")

    def synthesize(self, reference_audio_path: str, prompt_text: str, inference_text: str):
        """
        執行語音合成
        :return: (sample_rate, audio_numpy_array)
        """
        if not self.is_loaded:
            self.load_models()

        # 1. 處理參考音訊
        print(f"🎧 Processing reference audio: {reference_audio_path}")
        wav, sr = librosa.load(reference_audio_path, sr=24000)
        if len(wav.shape) > 1:
            wav = librosa.to_mono(wav)
        wav_tensor = torch.tensor(wav).unsqueeze(0).to(conf.DEVICE)

        # 2. 處理文字
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(conf.DEVICE)
        inference_ids = self.tokenizer(inference_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(conf.DEVICE)

        # 3. 推理
        with torch.no_grad():
            # 提取語義
            prompt_semantics = self.sovits_model.extract_semantics(wav_tensor, prompt_ids)
            
            # GPT 生成
            pred_semantics = self.gpt_model.generate(prompt_semantics, inference_ids)
            
            # SoVITS 生成 Mel
            y_lengths = torch.LongTensor([wav_tensor.size(1)]).to(conf.DEVICE)
            text_lengths = torch.LongTensor([inference_ids.size(1)]).to(conf.DEVICE)
            
            mel, *_ = self.sovits_model.infer(
                ssl=pred_semantics,
                y=wav_tensor,
                y_lengths=y_lengths,
                text=inference_ids,
                text_lengths=text_lengths
            )
            
            # Vocoder 轉波形
            audio = self.vocoder.infer(mel).squeeze().cpu().numpy()

        return 24000, audio
