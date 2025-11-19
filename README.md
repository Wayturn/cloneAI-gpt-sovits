# cloneAI-gpt-sovits

中文語音克隆 / Text-to-Speech 應用，基於 **GPT-SoVITS v3** 與 **faster-whisper**，  
可直接部署至 **Hugging Face Spaces**，提供簡單易用的 Gradio 介面與 API。

> 🎯 Goal: 提供一個「開箱即用」的中文語音克隆 Demo，示範從語音上傳 → 語音辨識 → 指定文本 → 語音合成的完整推理流程。

---

## ✨ Features

- 🎙️ 上傳 3–10 秒語音樣本（或透過麥克風錄音）
- 🔎 使用 **faster-whisper** 自動進行語音辨識（ASR）
- ✍️ 輸入希望 AI 說出的中文句子
- 🧠 透過 **GPT-SoVITS v3** 完成語音克隆與風格遷移
- 🔊 輸出 `.wav` 檔，可線上試聽或下載
- ☁️ 支援一鍵部署至 **Hugging Face Spaces (Gradio)**

---

## 🧱 Tech Stack

- **Language**：Python 3.10+
- **Core Models**
  - GPT-SoVITS v3（中文語音克隆 / TTS）
  - faster-whisper（語音辨識）
- **Frameworks**
  - Gradio（Web UI）
  - FastAPI（如需 API 化可擴充）
- **Others**
  - PyTorch
  - ffmpeg（音訊處理）
  - Hugging Face Spaces（部署）

---

## 📁 Project Structure

```bash
.
├── app.py                 # 入口程式，啟動 Gradio / Web 介面
├── requirements.txt       # 套件需求
├── space.yaml             # Hugging Face Spaces 設定
├── gpt_sovits/
│   ├── module/            # GPT-SoVITS 相關模組
│   ├── pretrain_models/   # 預訓練模型放置位置（需自行下載）
│   ├── text/              # 文本處理相關工具
│   ├── f5_tts/            # F5 TTS / vocoder 等模型
│   ├── sovits.py          # SoVITS 推理主程式
│   └── inference_webui.py # 原版 WebUI 推理流程（部分邏輯沿用）
└── utils/
    └── whisper_utils.py   # 工具函式（路徑處理、音訊工具等）
```

---

## 🚀 Quick Start (Local)

### 1. Clone 專案

```bash
git clone https://github.com/Wayturn/cloneAI-gpt-sovits.git
cd cloneAI-gpt-sovits
```

### 2. 建立虛擬環境（可選）

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. 安裝依賴

```bash
pip install -r requirements.txt
```

### 4. 準備模型權重

請將所需的 GPT-SoVITS / vocoder 模型下載後，放入對應目錄，例如：

```bash
gpt_sovits/pretrain_models/
    ├── sovits_weights.pth
    ├── gpt_weights.ckpt
    └── bigvgan_generator.pt
```

⚠️ **模型檔不隨專案提供**，請依照 GPT-SoVITS 官方或 Hugging Face 頁面說明下載對應權重。

如果有使用 `.env` 管理模型路徑，可在根目錄建立：

```bash
.env
```

內容如下：

```
MODEL_DIR=./gpt_sovits/pretrain_models
```

### 5. 啟動服務

```bash
python app.py
```

啟動後終端機會顯示本機網址，例如：

```
Running on http://127.0.0.1:7860
```

在瀏覽器打開該網址即可看到語音克隆介面。

---

## ☁️ Deploy to Hugging Face Spaces

1. 建立一個新的 Hugging Face Space
2. Space type 選擇：**Gradio**
3. 將此 repo push 到該 Space 的 Git repository
4. 在 `space.yaml` 中確認：
   ```yaml
   app_file: app.py
   sdk: gradio
   ```
5. 在 Spaces 的「Settings」中設定必要的環境變數（例如模型路徑、是否使用 GPU 等）
6. 儲存後，Spaces 會自動 Build & Deploy，完成後即可在線上使用

---

## ⚙️ Configuration

你可以透過下列方式調整推理行為（實際項目以程式碼為準）：

- 參考語音長度（預設 3–10 秒）
- 語速、停頓時間
- 推理策略（如 top_k, top_p, temperature）
- 語音切片設定（長句分段合成）

可以在 `gpt_sovits/sovits.py` 或 `inference_webui.py` 裡調整預設參數。

---

## 🔒 Notes / Limitations

- 本專案僅包含 **推理（inference）** 相關程式碼，不含完整訓練流程
- 請自行確認使用自有或合法授權之語音作為輸入樣本
- 若在 CPU 環境執行，推理速度可能較慢，建議使用 GPU

---

## 🧬 Roadmap

- [ ] 提供標準化 REST API（FastAPI 版）
- [ ] 加入批次合成功能（從文字清單產生多個語音檔）
- [ ] 整合字幕與簡易影片輸出 Pipeline
- [ ] 新增英文 / 多語系支援
- [ ] 提供 Docker 映像檔

---

## 🙏 Acknowledgements

本專案基於以下開源專案進行調整與封裝：

- [GPT-SoVITS v3](https://github.com/RVC-Boss/GPT-SoVITS)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)

其他相關模型與工具請見原專案授權條款。

---

## 🌏 English Summary

**cloneAI-gpt-sovits** is a Chinese voice cloning demo built on GPT-SoVITS v3 and faster-whisper, packaged as a Gradio app ready to be deployed to Hugging Face Spaces.

It demonstrates an end-to-end flow:

1. Upload a short reference audio (3–10 seconds)
2. Transcribe the audio with Whisper / faster-whisper
3. Enter any target sentence
4. Generate a cloned voice `.wav` with the same tone and style

The project focuses on inference only, with a clear directory structure, configuration via `.env`, and can be extended into a production-ready TTS / voice cloning service.

---
