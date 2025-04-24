import os
import gradio as gr
import tempfile
import shutil
from utils.whisper_utils import transcribe_audio, get_audio_duration
# 假設 gpt_sovits.inference_webui 提供 synthesize_speech 函數
from gpt_sovits.inference_webui import synthesize_speech


def save_temp_audio(original_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(original_path)[-1]) as tmp:
        shutil.copy(original_path, tmp.name)
        return tmp.name


def check_audio_length(audio_path, min_sec=3, max_sec=10):
    duration = get_audio_duration(audio_path)
    if duration < min_sec:
        return False, f"語音長度僅 {duration:.1f} 秒，請上傳 3–10 秒語音。"
    if duration > max_sec:
        return False, f"語音長度為 {duration:.1f} 秒，請上傳 3–10 秒語音。"
    return True, None


def process(reference_audio, inference_text):
    if reference_audio is None:
        return None, "請上傳語音樣本。", ""

    audio_path = save_temp_audio(reference_audio)
    # 檢查長度
    valid, msg = check_audio_length(audio_path)
    if not valid:
        os.remove(audio_path)
        return None, msg, ""
    # 語音辨識
    prompt_text = transcribe_audio(audio_path)
    # 語音合成
    try:
        out_wav_path = synthesize_speech(
            reference_audio_path=audio_path,
            prompt_text=prompt_text,
            inference_text=inference_text,
            out_dir=tempfile.gettempdir()
        )
        print("▶️ 語音樣本辨識文字（prompt_text）:", prompt_text)
        print("▶️ 使用者輸入的目標句子（inference_text）:", inference_text)
        if not os.path.exists(out_wav_path):
            return None, "❌ 錯誤：語音合成失敗，未產出音訊檔案"
        # 輸出 Gradio 可播放、下載的檔案
        with open(out_wav_path, "rb") as f:
            audio_bytes = f.read()
        download_link = f"點此下載合成語音: [下載]({out_wav_path})"
        return (audio_bytes, download_link)
    except Exception as e:
        return None, f"合成失敗: {str(e)}", ""
    finally:
        os.remove(audio_path)


def ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 中文語音克隆（GPT-SoVITS v3）\n請上傳 3–10 秒語音樣本，輸入想讓 AI 說的句子。")
        with gr.Row():
            reference_audio = gr.Audio(label="語音樣本（3–10 秒）", sources=[
                                       "upload", "microphone"], type="filepath")
            inference_text = gr.Textbox(label="希望 AI 說的句子（中文）")
        run_btn = gr.Button("開始語音克隆")
        output_audio = gr.Audio(label="合成語音", type="filepath")
        download_html = gr.HTML()

        def _wrapped(reference_audio, inference_text):
            result = process(reference_audio, inference_text)
            if result is None:
                return None, "❌ 錯誤：無法處理資料"
            if isinstance(result[1], str) and result[1].startswith("合成失敗"):
                return None, result[1]  # 顯示合成失敗原因
            if isinstance(result[1], str) and result[1].startswith("❌ 錯誤"):
                return None, result[1]  # 顯示自定義錯誤
            return result[0], result[1]
        run_btn.click(_wrapped, inputs=[reference_audio, inference_text], outputs=[
                      output_audio, download_html], show_progress=True)
    return demo


if __name__ == "__main__":
    ui().launch()
