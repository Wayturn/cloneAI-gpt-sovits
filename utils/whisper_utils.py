import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
import librosa

# Whisper 模型初始化（可根據 Hugging Face Spaces 資源調整模型大小）
model = WhisperModel("small", device="cpu", compute_type="int8")


def transcribe_audio(audio_path):
    """
    使用 faster-whisper 轉錄音訊檔案，回傳辨識文字。
    """
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = "".join([seg.text for seg in segments])
    return text.strip()


def get_audio_duration(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return librosa.get_duration(y=y, sr=sr)
