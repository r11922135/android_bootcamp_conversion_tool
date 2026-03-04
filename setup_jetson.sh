#!/bin/bash
# ============================================================
# Jetson AGX Orin — 完整環境設定腳本
# 安裝所有必要環境：Python、FFmpeg、Whisper、Ollama、LLM 模型
# 整個影片轉筆記管線都在 AGX Orin 上執行
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "============================================"
echo "  Jetson AGX Orin — 完整環境設定"
echo "  整個管線在此機器上執行"
echo "============================================"
echo ""
echo "專案目錄：$PROJECT_DIR"
echo ""

# ── Step 1: 系統依賴 ──
echo "▶ [1/5] 安裝系統依賴..."
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg \
    git \
    curl

echo "  ✓ 系統依賴安裝完成"
echo ""

# ── Step 2: Python 虛擬環境 ──
echo "▶ [2/5] 建立 Python 虛擬環境..."
cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ 虛擬環境已建立"
else
    echo "  ✓ 虛擬環境已存在"
fi

source venv/bin/activate
pip install --upgrade pip

echo "  安裝 Python 依賴..."
pip install -r requirements.txt
echo "  ✓ Python 依賴安裝完成"
echo ""

# ── Step 3: 驗證 CUDA ──
echo "▶ [3/5] 驗證 GPU / CUDA..."
python3 -c "
try:
    import torch
    print(f'  PyTorch: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f'  GPU Memory: {mem_gb:.1f} GB')
except ImportError:
    print('  ⚠ PyTorch 未安裝，Whisper 將自動安裝')
" 2>&1 || echo "  ⚠ PyTorch 檢查失敗，但不影響安裝繼續"
echo ""

# ── Step 4: Ollama + LLM 模型 ──
echo "▶ [4/5] 安裝 Ollama 與 LLM 模型..."

if ! command -v ollama &> /dev/null; then
    echo "  安裝 Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "  ✓ Ollama 已安裝"
fi

# 確保 Ollama 監聽 localhost
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl restart ollama

# 等待 Ollama 啟動
echo "  等待 Ollama 啟動..."
for i in {1..10}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  ✓ Ollama 服務已啟動"
        break
    fi
    sleep 2
done

echo ""
echo "  選擇要下載的 LLM 模型："
echo "    1) Qwen2.5 32B Q4 (推薦，AGX Orin 64GB 最佳平衡，需要 ~24GB 記憶體)"
echo "    2) Qwen2.5 14B Q4 (輕量快速，需要 ~12GB 記憶體)"
echo "    3) Llama 3.1 70B Q4 (英文理解力強，需要 ~48GB，AGX 上較吃力)"
echo "    4) Qwen2.5 72B Q4 (綜合最強，需要 ~50GB，AGX 上可能 OOM)"
echo ""
read -p "  請輸入選擇 (1/2/3/4) [預設: 1]: " MODEL_CHOICE
MODEL_CHOICE=${MODEL_CHOICE:-1}

case $MODEL_CHOICE in
    1) MODEL="qwen2.5:32b-instruct-q4_K_M" ;;
    2) MODEL="qwen2.5:14b-instruct-q4_K_M" ;;
    3) MODEL="llama3.1:70b-instruct-q4_K_M" ;;
    4) MODEL="qwen2.5:72b-instruct-q4_K_M" ;;
    *) MODEL="qwen2.5:32b-instruct-q4_K_M" ;;
esac

echo "  下載 $MODEL（這需要一些時間）..."
ollama pull $MODEL
echo "  ✓ 模型下載完成"

# 更新 config.yaml 中的模型名稱
if command -v sed &> /dev/null; then
    sed -i "s|model: \"qwen2.5:.*\"|model: \"$MODEL\"|" "$PROJECT_DIR/config.yaml" 2>/dev/null || true
fi
echo ""

# ── Step 5: 建立目錄 ──
echo "▶ [5/5] 建立專案目錄..."
mkdir -p "$PROJECT_DIR/videos"
mkdir -p "$PROJECT_DIR/audio"
mkdir -p "$PROJECT_DIR/transcripts"
mkdir -p "$PROJECT_DIR/notes"
echo "  ✓ 目錄結構已建立"
echo ""

# ── 驗證 ──
echo "============================================"
echo "  驗證安裝結果"
echo "============================================"
echo ""
echo "  FFmpeg:  $(ffmpeg -version 2>&1 | head -1)"
echo "  Python:  $(python3 --version)"
echo "  Ollama:  $(ollama --version 2>&1 || echo '未知')"
echo ""
echo "  已安裝的 Ollama 模型："
ollama list 2>/dev/null || echo "  （無法取得）"
echo ""

echo "============================================"
echo "  ✓ 環境設定完成！"
echo "============================================"
echo ""
echo "📝 使用方式："
echo "  1. 將英文影片檔案放入 videos/ 目錄"
echo "  2. 啟用虛擬環境：source venv/bin/activate"
echo "  3. 一鍵執行：python scripts/run_pipeline.py"
echo ""
echo "  或分步驟執行："
echo "    python scripts/01_extract_audio.py    # 抽音軌"
echo "    python scripts/02_transcribe.py       # 英文語音辨識"
echo "    python scripts/03_generate_notes.py   # 生成結構化英文筆記"
echo ""
echo "⚙️  設定檔：config.yaml"
echo "   目前使用模型：$MODEL"
echo ""
