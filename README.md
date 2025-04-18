# 中文語音克隆應用（Hugging Face Spaces）

本專案為可直接部署於 Hugging Face Spaces 的中文語音克隆應用，基於 Gradio UI、faster-whisper 進行語音辨識，並以 GPT-SoVITS v3 進行語音合成。

## 功能特色
1. 上傳 3–10 秒語音（或錄音）
2. 語音自動辨識（Whisper/faster-whisper）
3. 輸入希望 AI 模仿說出的句子
4. 語音克隆合成並產生 .wav 檔

## 如何部署
1. 下載本專案所有檔案
2. 於 `gpt_sovits/` 目錄放入預訓練模型（如 `sovits_weights.pth`），或依 README 指示下載
3. 部署於 Hugging Face Spaces（選擇 Gradio 類型）

## 依賴安裝
```
pip install -r requirements.txt
```

## 注意事項
- 僅保留 GPT-SoVITS v3 推理相關程式碼，無訓練模組
- 若需模型檔案，請依 Hugging Face 或官方 repo 指示下載

## 參考來源
- [GPT-SoVITS v3](https://github.com/innnky/gpt-sovits)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

# cloneAI-gpt-sovits