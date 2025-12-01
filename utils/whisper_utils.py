import librosa
from faster_whisper import WhisperModel
from config import conf

class WhisperASR:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            print(f"⏳ Loading Whisper model ({conf.WHISPER_MODEL_SIZE})...")
            # 根據 config 決定 device
            device = "cuda" if conf.DEVICE == "cuda" else "cpu"
            # 注意: int8 在 CPU 上可能不支援，視情況調整
            compute_type = conf.WHISPER_COMPUTE_TYPE
            
            self.model = WhisperModel(
                conf.WHISPER_MODEL_SIZE, 
                device=device, 
                compute_type=compute_type
            )
            print("✅ Whisper model loaded.")

    def transcribe(self, audio_path):
        self.load_model()
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        text = "".join([seg.text for seg in segments])
        return text.strip()

# 單例實例
asr = WhisperASR()

def transcribe_audio(audio_path):
    """相容舊版 API 的 wrapper"""
    return asr.transcribe(audio_path)

def get_audio_duration(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return librosa.get_duration(y=y, sr=sr)
