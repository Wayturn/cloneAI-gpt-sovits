import os
import gradio as gr
import tempfile
import shutil
import soundfile as sf
from utils.whisper_utils import transcribe_audio, get_audio_duration
from core.inference import InferenceEngine

# 初始化推理引擎 (Lazy Loading，這裡只是建立實例)
engine = InferenceEngine()

def save_temp_audio(original_path):
    """將上傳的音訊存為暫存檔"""
    suffix = os.path.splitext(original_path)[-1]
    if not suffix: suffix = ".wav"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copy(original_path, tmp.name)
        return tmp.name

def check_audio_length(audio_path, min_sec=3, max_sec=10):
    """檢查音訊長度是否符合要求"""
    duration = get_audio_duration(audio_path)
    if duration < min_sec:
        return False, f"語音長度僅 {duration:.1f} 秒，請上傳 3–10 秒語音。"
    if duration > max_sec:
        return False, f"語音長度為 {duration:.1f} 秒，請上傳 3–10 秒語音。"
    return True, None

def process(reference_audio, inference_text):
    if reference_audio is None:
        return None, "請上傳語音樣本。"

    audio_path = save_temp_audio(reference_audio)
    
    try:
        # 1. 檢查長度
        valid, msg = check_audio_length(audio_path)
        if not valid:
            return None, msg
            
        # 2. 語音辨識 (ASR)
        print(f"🎤 正在辨識語音: {audio_path}")
        prompt_text = transcribe_audio(audio_path)
        print(f"📝 辨識結果: {prompt_text}")
        
        # 3. 語音合成 (TTS)
        print(f"🤖 開始合成語音...")
        sr, audio_data = engine.synthesize(
            reference_audio_path=audio_path,
            prompt_text=prompt_text,
            inference_text=inference_text
        )
        
        # 4. 儲存結果
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            out_wav_path = tmp.name
            
        sf.write(out_wav_path, audio_data, sr)
        print(f"✅ 合成成功，檔案已儲存: {out_wav_path}")
        
        return out_wav_path, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"合成失敗: {str(e)}"
    finally:
        # 清理暫存檔
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

def ui():
    with gr.Blocks(title="GPT-SoVITS Voice Clone") as demo:
        gr.Markdown("# 🎙️ 中文語音克隆 (GPT-SoVITS v3)")
        gr.Markdown("請上傳一段 3–10 秒的中文語音作為參考，並輸入你希望 AI 說的話。")
        
        with gr.Row():
            with gr.Column():
                reference_audio = gr.Audio(
                    label="1. 上傳參考語音 (3-10秒)", 
                    sources=["upload", "microphone"], 
                    type="filepath"
                )
                inference_text = gr.Textbox(
                    label="2. 輸入目標文字 (中文)", 
                    placeholder="你好，這是一個測試語音。",
                    lines=3
                )
                run_btn = gr.Button("🚀 開始語音克隆", variant="primary")
            
            with gr.Column():
                output_audio = gr.Audio(label="合成結果", type="filepath")
                status_msg = gr.Markdown("")

        def _wrapped(ref, text):
            audio_out, err_msg = process(ref, text)
            
            if audio_out is None:
                return None, f"❌ {err_msg}"
            
            return audio_out, "✅ 合成完成！"

        run_btn.click(
            _wrapped, 
            inputs=[reference_audio, inference_text], 
            outputs=[output_audio, status_msg]
        )
        
    return demo

if __name__ == "__main__":
    ui().launch()
