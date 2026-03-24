# Android Bootcamp 影片轉 Markdown 筆記

## 專案概述

這個專案的目標是把 Android Bootcamp 英文影片，轉成可閱讀、可追溯、可增量重跑的 Markdown 筆記。
目前實作不是單一「黑盒摘要器」，而是一條明確分段的本地管線：

1. 影片抽音軌
2. 音檔做英文語音辨識
3. 逐字稿交給 LLM 產生英文詳細筆記與英文摘要
4. 再由英文筆記改寫出繁體中文筆記與中文摘要

整個流程的核心設計原則是：

- 所有中間產物都落地成檔案，方便檢查與重跑
- 每個 step 都盡量可跳過已完成輸出
- LLM 推理拆成多段，避免一次塞完整場逐字稿造成截斷或漂移
- 中文不是直接吃逐字稿，而是以英文筆記為中介，降低中文模型負擔

目前主要依賴：

- `FFmpeg`：抽音軌
- `faster-whisper`：英文語音辨識
- `OpenAI` Python client：連接 OpenAI 相容 API
- `Ollama` / `llama.cpp server`：本地 LLM 端點
- 可選 `pdftotext`：從 slides PDF 擷取文字

---

## 目前輸入與輸出

### 輸入

- `videos/<stem>.<ext>`：原始影片
- 可選 `slides/<stem>.txt | .md | .pdf`：同名投影片內容

### 中間產物

- `audio/<stem>.wav`：16kHz mono WAV
- `transcripts/<stem>.json`：逐字稿主資料來源，格式為 `[{start, end, text}, ...]`
- `transcripts/<stem>.txt`：方便人工掃讀的逐字稿
- `transcripts/<stem>.srt`：字幕檔

### 最終輸出

- `notes/<stem>.md`：英文筆記（英文摘要 + 詳細筆記）
- `notes/<stem>.zh-TW.summary.md`：繁中筆記（繁中摘要 + 繁中詳細筆記）

---

## 專案架構總覽

| 模組 | 角色 | 主要輸入 | 主要輸出 |
|---|---|---|---|
| `scripts/run_pipeline.py` | 一鍵 orchestration | `videos/` | 依序觸發三個 step |
| `scripts/01_extract_audio.py` | 抽音軌 | 影片 | `audio/*.wav` |
| `scripts/02_transcribe.py` | 語音辨識 | `audio/*.wav` | `transcripts/*.json/.txt/.srt` |
| `scripts/03_generate_notes.py` | 主筆記生成器 | `transcripts/*.json`、可選 `slides/` | `notes/*.md`、`notes/*.zh-TW.summary.md` |
| `scripts/03_generate_notes_oneshot.py` | 單檔 one-shot 實驗入口 | 單一 transcript | `notes/*.one-shot.md` |
| `scripts/utils.py` | 共用工具 | config / path / text | chunking、路徑、時間格式等 |
| `config.yaml` | 全域設定中心 | 無 | 控制 STT / LLM / chunking / slides / zh 改寫 |

這個專案目前沒有資料庫、任務佇列、Web API 或背景 worker。整個系統就是一個以檔案系統為中心的離線批次處理器。

---

## 目錄結構

```text
android_bootcamp_conversion_tool/
├── README.md
├── config.yaml
├── requirements.txt
├── setup_jetson.sh
├── Dockerfile.agx
├── docker-compose.agx.yml
├── run_docker_agx.sh
├── videos/                         # 原始影片輸入
├── audio/                          # Step 1 輸出 WAV
├── transcripts/                    # Step 2 輸出逐字稿
├── slides/                         # 可選投影片文字 / PDF
├── notes/                          # Step 3 最終輸出
├── logs/                           # 長時間執行的 log（例如 screen + logfile）
├── notes_backup_*/                 # 歷史筆記備份
├── venv/
└── scripts/
    ├── 01_extract_audio.py
    ├── 02_transcribe.py
    ├── 03_generate_notes.py
    ├── 03_generate_notes_oneshot.py
    ├── run_pipeline.py
    └── utils.py
```

---

## 完整資料流

```text
影片
  -> Step 1: FFmpeg 抽音軌
  -> audio/<stem>.wav
  -> Step 2: faster-whisper
  -> transcripts/<stem>.json   # 主資料來源
  -> Step 3: LLM 生成英文筆記
  -> notes/<stem>.md
  -> Step 3b: 由英文筆記改寫繁中筆記
  -> notes/<stem>.zh-TW.summary.md
```

### Step 2 與 Step 3 的資料契約

`transcripts/<stem>.json` 是 Step 3 的 source of truth，格式如下：

```json
[
  {
    "start": 0.0,
    "end": 14.24,
    "text": "Good morning."
  }
]
```

Step 3 不直接吃 `.txt` 或 `.srt`，而是讀取 `.json`，自行決定是否把時間戳嵌入 prompt。

---

## 核心推論邏輯

這一節描述的是「目前程式實際怎麼推理」，不是理想流程圖。

### 0. orchestration：`scripts/run_pipeline.py`

`run_pipeline.py` 是最外層入口，做的事情很單純：

- 讀 `config.yaml`
- 建立必要目錄
- 掃描 `videos/`
- 依序以 `subprocess.run()` 呼叫：
  - `01_extract_audio.py`
  - `02_transcribe.py`
  - `03_generate_notes.py`
- 任一步失敗就中止整條管線

這代表目前流程是嚴格同步、線性執行，不做平行調度，也不做跨 step 的共享記憶體優化。

### 1. Step 1：抽音軌邏輯

實作位置：`scripts/01_extract_audio.py`

流程：

- 從 `videos/` 掃所有支援副檔名
- 以 `utils.get_stem()` 將檔名正規化，避免 Windows/Unix 不相容字元
- 呼叫 `ffmpeg` 轉成：
  - `pcm_s16le`
  - `16kHz`
  - `mono`
- 若 `audio/<stem>.wav` 已存在則直接跳過

這一層幾乎沒有推論邏輯，重點是把各種影片來源統一成 Whisper 最穩定的輸入格式。

### 2. Step 2：語音辨識邏輯

實作位置：`scripts/02_transcribe.py`

#### 2.1 模型載入與設定

每處理一個音檔，`transcribe_audio()` 會依 `config.yaml` 建立一次 `WhisperModel`：

- `whisper.model`
- `whisper.device`
- `whisper.compute_type`
- `whisper.language`
- `whisper.beam_size`
- `whisper.vad_filter`
- `whisper.initial_prompt`

這個設計的優點是簡單穩定，缺點是多檔批次時會反覆載入模型，不是吞吐量最優。

#### 2.2 輸出策略

`faster-whisper` 回傳 segment generator 後，程式把每一段收集成：

- `start`
- `end`
- `text`

然後同時輸出：

- `.json`：完整結構化逐字稿
- `.txt`：含 `[HH:MM:SS]` 的純文字稿
- `.srt`：字幕檔

其中 `.json` 是後續所有筆記生成的唯一主來源。

#### 2.3 進度顯示

Step 2 有 heartbeat thread，會根據 `info.duration`、目前處理到的 segment end time、wall-clock elapsed time 去估算：

- 已處理秒數
- 速率（`x realtime`）
- 預估剩餘時間

這只是 CLI 體驗改善，不影響推論內容。

### 3. Step 3：英文筆記生成邏輯

實作位置：`scripts/03_generate_notes.py`

Step 3 是整個專案的核心。它不是「單次摘要」，而是「分階段寫作 + 清理 + 驗證 + 再改寫」。

#### 3.1 輸入組裝

對每份 `transcripts/<stem>.json`：

- 讀取 segments
- 若 `notes.include_timestamps = true`，把每段轉成：
  - `[HH:MM:SS] text`
- 合併成 `full_text`

這個 `full_text` 是英文摘要與英文詳細筆記的原始語料。

#### 3.2 transcript chunking

英文詳細筆記預設走 multi-chunk 路徑：

- 使用 `utils.chunk_text()`
- 依 `llm.chunk_size` / `llm.chunk_overlap` 切段
- 目前預設：
  - `chunk_size: 6000`
  - `chunk_overlap: 120`

`chunk_text()` 是字元級切段，不是 token-aware，也不是語意切段。它會盡量在：

- 空行
- 換行
- 中英文句號 / 問號 / 驚嘆號
- 逗號

附近找切點。

#### 3.3 每段英文詳細筆記的 prompt 策略

每個 transcript chunk 都會產生一個 `CHUNK_PROMPT_TEMPLATE`，要求模型：

- 只依 transcript 證據寫
- 避免 boilerplate
- 保留 certainty（available / preview / planned）
- 避免孤兒標題、半句、殘缺條列
- 只在允許且有證據時輸出 code block

如果 `notes.use_slides_context = true`，還會把檢索到的 slides 片段一起塞進 prompt，但要求模型只在相關時使用。

#### 3.4 `call_llm()` 的推論控制

所有 LLM 呼叫都走 `call_llm()`，它做了幾件重要事情：

- 使用 OpenAI-compatible chat completions API
- 支援 `max_tokens_override` / `temperature_override` / `max_retries_override`
- 顯示 heartbeat 進度
- 若 `finish_reason == "length"`，會自動續寫
- 續寫時把前文當作 assistant content，追加一個 user 訊息要求：
  - 接著上次中斷處繼續
  - 不要重複
  - 在完整的 Markdown 邊界結束
- 最多續寫 `llm.max_continuations` 次

這是目前專案用來降低「模型輸出被截斷」的主要機制。

#### 3.5 英文詳細筆記後處理

multi-chunk 生成完成後，會把各 chunk 用 `---` 串起來，再進 `cleanup_english_detailed_notes()`：

1. 先把詳細筆記分 batch（預設約 9000 chars）
2. 對每個 batch 做 conservative cleanup
3. 移除：
   - 空標題
   - 占位字串
   - 重複 separator
   - 不完整 bullet / dangling fragments
4. 若清理後長度不大，再做一次 global normalize

這層不是用來補內容，而是做保守清理，不應新增事實。

#### 3.6 英文摘要生成

英文摘要不是從詳細筆記直接固定生成，而是兩階段備援：

- primary：`TRANSCRIPT_SUMMARY_PROMPT_TEMPLATE`
  - 直接從 transcript 做摘要
- fallback：`SUMMARY_PROMPT_TEMPLATE`
  - 若 primary 沒產出完整標題，再改用 cleaned detailed notes 做摘要

目前英文摘要固定只保留這四段：

- `## Course Summary`
- `## What Changed / Announced`
- `## Why It Matters`
- `## Key Terms`

程式會檢查這些 heading 是否齊全，不齊就重試或 fallback。

#### 3.7 最終英文檔格式

英文輸出固定為：

```text
# <video_title>
> Duration
> Generated by

---
<English Summary>

---
## Detailed Notes
<Cleaned English Detailed Notes>
```

### 4. 中文筆記生成邏輯

中文不是直接由逐字稿生成，而是由英文筆記改寫。這是目前專案很重要的架構選擇。

實作位置：`generate_zh_notes_from_english_markdown()`

#### 4.1 為什麼先英文再中文

這樣做的原因是：

- 英文 transcript 對英文模型更直接
- 中文模型只需要處理已經壓縮過、結構化的英文筆記
- 能把中英文的章節結構對齊
- 減少中文端一次吃完整逐字稿造成的 context 壓力

代價是：英文若已漏內容，中文會繼承該問題。

#### 4.2 英文筆記切段方式

中文詳細筆記不再直接用 raw char chunking，而是先：

- `split_english_markdown_sections()`：把英文 summary 與 detailed notes 拆開
- `chunk_markdown_by_sections()`：盡量按 `## / ### / ####` 區段打包
- 若單一 section 太大，才退回 `chunk_text()` 再切

這比直接對整段英文筆記做 char chunking 更適合 Markdown 內容。

#### 4.3 中文改寫 prompt

每個英文 note chunk 會套 `ZH_REWRITE_CHUNK_PROMPT_TEMPLATE`，要求模型：

- 使用繁體中文（台灣）
- 不要逐句翻譯
- 保留技術脈絡
- 保留 certainty
- API / 類別 / 方法名稱保留英文 + backticks
- 不要新增全域摘要或評論
- 不要輸出簡體中文

#### 4.4 中文品質檢查與自我修正

每個繁中 chunk 生成後，會檢查：

- 長度是否低於 `llm_zh.min_chunk_chars`
- 是否被整段 code fence 包住
- 是否含簡體或翻譯腔提示詞

若有問題，會再用 `ZH_REVIEW_PROMPT_TEMPLATE` 對該 chunk 做一次修正。

#### 4.5 中文摘要生成

中文摘要有兩種來源：

- 若英文檔已有英文摘要：
  - 用 `ZH_SUMMARY_FROM_ENGLISH_PROMPT_TEMPLATE` 直接改寫英文摘要
- 否則：
  - 退回 `ZH_SUMMARY_PROMPT_TEMPLATE`，從中文詳細筆記再做摘要

目前中文摘要固定保留：

- `## 課程摘要`
- `## 本次更新／新宣布事項`
- `## 影響與價值`
- `## 關鍵術語`

#### 4.6 最終中文檔格式

中文輸出固定為：

```text
# <video_title>｜繁體中文筆記
> Duration
> Generated by
> Source

---
<中文摘要>

---
## 詳細筆記
<繁中詳細筆記>
```

### 5. slides 檢索邏輯（可選）

如果 `notes.use_slides_context = true`，Step 3 會額外載入同名 slides：

- `slides/<stem>.txt`
- `slides/<stem>.md`
- `slides/<stem>.pdf`（需 `pdftotext`）

#### 5.1 slides 前處理

- 正規化換行與空白
- 去掉明顯雜訊行（例如 confidential / do not distribute）
- 依 page 或一般段落切成 blocks
- 為每個 block 建 token set

#### 5.2 transcript chunk 與 slides block 的對齊方式

目前不是 embedding retrieval，而是 lexical overlap：

- 對 transcript chunk tokenization
- 對 slide block tokenization
- 用 token overlap + lexical length + snippet length bonus 打分
- 選 top-k
- 視需要向前後 block 擴展成較完整 snippet
- 再把 top-k snippets 塞進 prompt

這個方法的優點是完全本地、輕量；缺點是對同義詞與跨句語意不敏感。

### 6. 增量執行與重跑邏輯

目前每個 step 都有自己的 skip 規則：

#### Step 1

- `audio/<stem>.wav` 已存在就跳過

#### Step 2

- `transcripts/<stem>.txt` 與 `transcripts/<stem>.json` 都存在就跳過

#### Step 3

- 英文 `.md` 已存在且中文關閉：跳過
- 英文 `.md` 已存在但中文缺失：只補跑中文
- 中文 `.zh-TW.summary.md` 存在且含 `## 詳細筆記`：視為完成

這使得 Step 3 可以單獨反覆重跑，而不必重做 STT。

### 7. one-shot 實驗路徑

`scripts/03_generate_notes_oneshot.py` 是單檔 transcript 的實驗入口。

它會：

- 接一個 transcript path 或 stem
- 走 `build_english_notes_from_transcript_oneshot()`
- 使用 `llm_oneshot` 設定
- 單次把完整 transcript 送進 LLM 產英文詳細筆記

這條路徑目前不是 `run_pipeline.py` 的預設流程，主要用於比較 one-shot 與 chunked 寫法的品質差異。

---

## 目前實際設定快照（以 `config.yaml` 為準）

| 設定項 | 目前值 | 用途 |
|---|---|---|
| `whisper.model` | `medium` | 語音辨識模型 |
| `whisper.device` | `cuda` | Whisper 推理裝置 |
| `whisper.compute_type` | `float16` | Whisper 精度 |
| `whisper.language` | `en` | 固定英文輸入 |
| `whisper.beam_size` | `5` | STT beam search |
| `whisper.vad_filter` | `false` | 是否做 VAD |
| `llm.model` | `qwen3:30b` | 英文筆記模型 |
| `llm.api_base` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `llm.max_tokens` | `4096` | 單次輸出上限 |
| `llm.max_continuations` | `4` | 被截斷時自動續寫上限 |
| `llm.chunk_size` | `6000` | transcript 英文 chunk 大小 |
| `llm.chunk_overlap` | `120` | transcript overlap |
| `llm.request_delay` | `1.0` | LLM 請求間延遲 |
| `notes.use_slides_context` | `false` | 預設不使用 slides |
| `notes.include_timestamps` | `true` | prompt 中保留時間戳 |
| `notes.detect_code` | `false` | 預設不輸出 code block |
| `notes.generate_summary` | `true` | 生成摘要 |
| `notes.generate_qa` | `false` | 保留欄位，但 prompt 目前不產 Q&A |
| `notes.generate_zh_summary` | `true` | 生成繁中筆記 |
| `llm_zh.model` | `qwen3:30b` | 中文改寫模型 |
| `llm_zh.chunk_size` | `2600` | 中文改寫來源 chunk 大小 |
| `llm_zh.chunk_overlap` | `180` | 中文改寫 overlap |
| `llm_oneshot.max_tokens` | `12288` | one-shot 英文詳細筆記輸出預算 |

---

## 為什麼專案要切成這樣

目前架構是有意識的多階段設計，不是因為寫不出單次大 prompt。

### 優點

- 每個 step 都能單獨驗證
- transcript / notes 都能人工抽查
- Step 3 可以在不重做 STT 的情況下反覆調 prompt
- 中文可直接重跑，不必再吃一次完整逐字稿
- slides 可開可關，不影響主流程

### 代價

- 多次 LLM 呼叫，耗時較長
- transcript 錯字會往下游傳遞
- 英文詳細筆記的切段仍是 char-based，不是 token-aware 或語意切段
- 中文品質上限受英文筆記品質限制

---

## 目前已知限制與設計取捨

1. **STT 錯誤會直接污染筆記**
   - 例如 `uses-feature` 這類術語若被 Whisper 聽歪，LLM 可能會照單全收。

2. **英文 transcript chunking 仍然不是 token-aware**
   - 雖然有 overlap 與後續 cleanup，但仍可能在 topic 邊界損失局部脈絡。

3. **slides 檢索是 lexical overlap，不是語意搜尋**
   - 好處是完全本地、低成本；壞處是對同義詞與改寫不敏感。

4. **中文不是 source-of-truth**
   - 中文稿是英文稿的改寫版本，適合閱讀，不適合拿來當最終技術依據。

5. **摘要適合 briefing，不適合當正式規格**
   - 特別是時程、政策、資格條件，仍建議回看 transcript 或手動校稿。

6. **Step 2 目前每個音檔都重新載入 WhisperModel**
   - 實作簡單，但不是最大吞吐量版本。

---

## 安裝與執行

### 本機安裝

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv ffmpeg git

cd /path/to/android_bootcamp_conversion_tool
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 啟動本地 LLM 服務

以 Ollama 為例：

```bash
ollama pull qwen3:30b
ollama serve
```

### 一鍵執行整條管線

```bash
source venv/bin/activate
python scripts/run_pipeline.py
```

### 分步執行

```bash
source venv/bin/activate
python scripts/01_extract_audio.py
python scripts/02_transcribe.py
python scripts/03_generate_notes.py
```

### 跳過已完成步驟

```bash
python scripts/run_pipeline.py --skip-audio
python scripts/run_pipeline.py --skip-transcribe
python scripts/run_pipeline.py --skip-notes
```

### one-shot 單檔英文實驗

```bash
source venv/bin/activate
python scripts/03_generate_notes_oneshot.py Android_Bootcamp_2026_AI
```

---

## Docker（AGX 版）

本專案保留 AGX 專用 Docker 包裝：

- `Dockerfile.agx`
- `docker-compose.agx.yml`
- `run_docker_agx.sh`

設計上建議：

- 容器只負責執行環境
- `videos/ audio/ transcripts/ notes/ config.yaml` 透過 volume 掛入
- Ollama 跑在 host
- 容器用 `network_mode: host` 連 `http://localhost:11434/v1`

常用操作：

```bash
./run_docker_agx.sh
./run_docker_agx.sh --skip-audio
./run_docker_agx.sh --skip-transcribe
./run_docker_agx.sh --skip-notes
```

---

## 長時間執行建議

Step 3 很容易跑很久，建議不要直接綁在單一 foreground shell。

實務上可用 `screen`：

```bash
screen -L -Logfile logs/notes_run.log -dmS notes_gen venv/bin/python scripts/03_generate_notes.py
screen -r notes_gen
```

或只看 log：

```bash
tail -F logs/notes_run.log
```

---

## 疑難排解

### 1. `audio/` 或 `transcripts/` 已存在舊資料，導致流程跳過

這是預期行為。若要重跑某一步，刪除對應輸出即可。

### 2. LLM 回應被截斷

目前已內建 `finish_reason == length` 的自動續寫；若仍常發生，可調整：

```yaml
llm:
  max_tokens: 4096
  max_continuations: 4
```

也可減少：

```yaml
llm:
  chunk_size: 6000
```

### 3. 中文筆記品質不穩

可優先調整：

```yaml
llm_zh:
  temperature: 0.15
  chunk_size: 2600
  chunk_overlap: 180
```

若英文稿本身有錯，中文通常只會把錯誤改寫得更順，不會自動修正事實。

### 4. 要檢查目前流程實際用了哪些設定

直接看：

- `config.yaml`
- `scripts/02_transcribe.py`
- `scripts/03_generate_notes.py`

README 的描述以目前這三個檔案為準，但若後續程式更新，最終仍應以程式與設定檔為 source of truth。
