# Android Bootcamp 影片轉 Markdown 筆記 — 完整指南

## 📌 專案概述

將 Android Bootcamp 英文影片自動轉換為結構化英文 Markdown 筆記，全程在 **Jetson Orin Thor** 上使用本地算力執行，不依賴任何雲端 API。

---

## 🖥️ 硬體配置

| 硬體 | 用途 | 記憶體 |
|------|------|--------|
| **Jetson Orin Thor** | 執行整個管線（FFmpeg + Whisper + LLM） | 最高 128 GB 統一記憶體 |
| **RTX 4080**（備用） | 若需分流可用來跑 Whisper | 16 GB VRAM |

> **整個專案直接跑在 Jetson Orin Thor 上**，Thor 的統一記憶體架構足以同時負擔 Whisper 語音辨識與大型 LLM 推理。

### 處理流程

```
英文影片 ─→ [FFmpeg 抽音軌] ─→ 音檔 (.wav)
                │
                ▼
[Jetson Thor: faster-whisper large-v3]  ─→ 英文逐字稿 (.txt / .srt)
                │
                ▼
[Jetson Thor: Qwen2.5 72B / Llama 3.1 70B]  ─→ 英文 Markdown 筆記
                │
                ▼
          最終筆記 (.md)
```

---

## 🧠 模型選擇

### 1. 語音轉文字 (STT) — 在 Thor 上執行

| 模型 | 大小 | 記憶體需求 | 說明 |
|------|------|----------|------|
| **faster-whisper large-v3-turbo** ★推薦 | 809M | ~4 GB | 速度快，品質接近 large-v3 |
| faster-whisper large-v3 | 1.5B | ~4 GB (CTranslate2) | 最高精準度 |
| faster-whisper medium | 769M | ~2 GB | 輕量備案 |

> 影片為全英文，Whisper 已設定 `language: "en"` 以提升辨識速度與準確度。

### 2. 文字摘要 / 筆記生成 (LLM) — 在 Thor 上執行

| 模型 | 大小 | 記憶體需求 | 說明 |
|------|------|----------|------|
| **Qwen2.5-72B-Instruct (Q4_K_M)** ★推薦 | ~44 GB | ~50 GB | 綜合能力最強，摘要品質高 |
| Llama-3.1-70B-Instruct (Q4_K_M) | ~42 GB | ~48 GB | 英文理解力強 |
| Qwen2.5-32B-Instruct (Q4_K_M) | ~20 GB | ~24 GB | 平衡選擇 |
| Qwen2.5-14B-Instruct (Q4_K_M) | ~9 GB | ~12 GB | 輕量快速 |

> Thor 有 128 GB 統一記憶體，可直接跑 72B Q4 量化模型。LLM 負責將英文逐字稿整理為結構化英文筆記。

---

## 🔧 環境安裝（在 Jetson Orin Thor 上）

### Step 1：安裝基礎環境

```bash
# 安裝系統依賴
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv ffmpeg git

# 進入專案目錄
cd /path/to/android-bootcamp   # 換成你的專案路徑

# 建立並啟用 Python 虛擬環境
python3 -m venv venv
source venv/bin/activate

# 安裝 Python 依賴
pip install -r requirements.txt
```

### Step 2：安裝 Ollama 與下載 LLM 模型

```bash
# 安裝 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下載推薦模型（約 44 GB，需要一些時間）
ollama pull qwen2.5:72b-instruct-q4_K_M

# 確認 Ollama 服務已啟動
curl http://localhost:11434/api/tags
```

### Step 3：驗證 GPU 與環境

```bash
# 確認 CUDA 可用
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# 確認 FFmpeg
ffmpeg -version

# 確認 Ollama 模型
ollama list
```

### 快速安裝（一鍵腳本）

```bash
bash setup_jetson.sh
```

---

## 📂 專案目錄結構

```
android bootcamp/
├── README.md                    # 本文件
├── config.yaml                  # 設定檔
├── requirements.txt             # Python 依賴
├── setup_jetson.sh              # Jetson 一鍵安裝腳本
├── scripts/
│   ├── 01_extract_audio.py      # Step1: 影片抽音軌
│   ├── 02_transcribe.py         # Step2: 英文語音轉文字
│   ├── 03_generate_notes.py     # Step3: 英文逐字稿 → 結構化英文 Markdown 筆記
│   ├── run_pipeline.py          # 一鍵全自動流程
│   └── utils.py                 # 工具函數
├── videos/                      # 放入英文影片檔案
│   └── (把 .mp4 / .mkv 放這裡)
├── audio/                       # 抽出的音檔
├── transcripts/                 # 英文逐字稿
└── notes/                       # 結構化英文 Markdown 筆記
```

---

## 🚀 使用方式

### 一鍵執行（推薦）

```bash
# 1. 把所有英文影片放進 videos/ 目錄
# 2. 確認 config.yaml 設定正確
# 3. 啟用虛擬環境
source venv/bin/activate
# 4. 執行
python scripts/run_pipeline.py
```

### 分步驟執行

```bash
source venv/bin/activate

# Step 1: 抽音軌
python scripts/01_extract_audio.py

# Step 2: 英文語音轉文字
python scripts/02_transcribe.py

# Step 3: 英文逐字稿 → 結構化英文筆記
python scripts/03_generate_notes.py
```

### 進階選項

```bash
# 跳過已完成的步驟
python scripts/run_pipeline.py --skip-audio        # 跳過抽音軌
python scripts/run_pipeline.py --skip-transcribe    # 跳過語音辨識
python scripts/run_pipeline.py --skip-notes         # 跳過筆記生成
```

---

## ⚙️ 設定檔說明 (config.yaml)

參見 `config.yaml` 中的詳細註解。關鍵設定：

| 設定項 | 目前值 | 說明 |
|-------|--------|------|
| `whisper.model` | `large-v3-turbo` | Whisper 模型大小 |
| `whisper.language` | `en` | 英文影片 |
| `whisper.device` | `cuda` | 使用 GPU 加速 |
| `llm.model` | `qwen2.5:72b-instruct-q4_K_M` | LLM 模型 |
| `llm.api_base` | `http://localhost:11434/v1` | Ollama 本地端點 |
| `notes.language` | `en` | 輸出英文筆記 |

---

## 🔄 完整處理流程圖

```
┌─────────────────────────────────────────────────────────────┐
│                  輸入：英文影片檔案                            │
│              (.mp4, .mkv, .avi, .webm, .mov)                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: FFmpeg 抽取音軌                                     │
│  • 轉成 16kHz mono WAV (Whisper 最佳格式)                     │
│  • 運算裝置：CPU                                             │
│  • 預估速度：即時（幾乎不耗時）                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: faster-whisper 英文語音辨識                          │
│  • 模型：large-v3-turbo                                      │
│  • 運算裝置：Jetson Orin Thor (CUDA)                          │
│  • 輸入語言：英文 (language: "en")                             │
│  • 輸出：含時間戳記的英文逐字稿                                 │
│  • 預估速度：1小時影片 ≈ 3~8 分鐘                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: LLM 生成結構化筆記                                    │
│  • 將英文逐字稿分段（每段 ~3000 字）送入 LLM                    │
│  • LLM 依照 Prompt 模板整理為結構化英文 Markdown                 │
│  • 運算裝置：Jetson Orin Thor (Ollama)                        │
│  • 模型：Qwen2.5-72B Q4                                      │
│  • 最後合併各段筆記為一份完整 .md                               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  輸出：結構化英文 Markdown 筆記                                │
│  • 標題、課程摘要                                            │
│  • 核心知識點                                                │
│  • 分段筆記（含時間戳記）                                      │
│  • 程式碼片段（自動偵測並格式化）                                │
│  • 關鍵術語、重點整理、Q&A                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 效能預估（Jetson Orin Thor）

| 影片長度 | 抽音軌 | 語音轉文字 | LLM 筆記生成 | 總計 |
|---------|--------|----------|-------------|------|
| 30 分鐘 | ~3 秒 | ~2 分鐘 | ~3 分鐘 | **~5 分鐘** |
| 1 小時 | ~5 秒 | ~5 分鐘 | ~6 分鐘 | **~11 分鐘** |
| 2 小時 | ~8 秒 | ~10 分鐘 | ~12 分鐘 | **~22 分鐘** |

> 不需翻譯，直接英文→英文整理，速度更快

---

## 🛠️ 疑難排解

### Whisper 記憶體不足
```yaml
# 在 config.yaml 中改用較小的模型
whisper:
  model: "medium"  # 或 "small"
```

### LLM 記憶體不足（OOM）
```yaml
# 改用較小的模型
llm:
  model: "qwen2.5:32b-instruct-q4_K_M"  # 或 14b
```

### LLM 回應太慢
- 減少 `chunk_size` 來降低每次送入的文字量
- 改用較小的模型

### 英文辨識有誤
```yaml
# 確認語言已設為英文
whisper:
  language: "en"
  # 可調整 initial_prompt 加入課程特定術語
```

### FFmpeg 找不到
```bash
sudo apt-get install -y ffmpeg
```

### Ollama 無法啟動
```bash
# 檢查服務狀態
sudo systemctl status ollama
# 重啟
sudo systemctl restart ollama
# 查看日誌
journalctl -u ollama -f
```
