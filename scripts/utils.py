"""
工具函數：載入設定、路徑管理、文字處理等共用功能
"""

import os
import re
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """取得專案根目錄"""
    return Path(__file__).parent.parent


def load_config() -> dict:
    """載入 config.yaml 設定檔"""
    config_path = get_project_root() / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到設定檔：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict):
    """確保所有輸出目錄存在"""
    root = get_project_root()
    for key in ["videos_dir", "audio_dir", "transcripts_dir", "notes_dir"]:
        dir_path = root / config["paths"][key]
        dir_path.mkdir(parents=True, exist_ok=True)


def get_video_files(config: dict) -> list[Path]:
    """取得所有影片檔案路徑"""
    root = get_project_root()
    videos_dir = root / config["paths"]["videos_dir"]
    extensions = config.get("video_extensions", [".mp4", ".mkv", ".avi", ".webm", ".mov"])
    
    video_files = []
    for ext in extensions:
        video_files.extend(videos_dir.glob(f"*{ext}"))
        video_files.extend(videos_dir.glob(f"*{ext.upper()}"))
    
    # 去重並排序
    video_files = sorted(set(video_files))
    return video_files


def sanitize_filename(name: str) -> str:
    """清理檔案名稱，移除不合法字元"""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def get_stem(video_path: Path) -> str:
    """取得影片的檔名（不含副檔名）"""
    return sanitize_filename(video_path.stem)


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    """
    將長文字分段，每段約 chunk_size 字元，相鄰段落重疊 overlap 字元。
    盡量在句號、換行處切割。
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # 嘗試在句號、換行處切割
        best_break = -1
        search_start = max(start + chunk_size - 500, start)
        
        for sep in ['\n\n', '\n', '。', '！', '？', '. ', '! ', '? ', '，', ', ']:
            pos = text.rfind(sep, search_start, end + 200)
            if pos > best_break:
                best_break = pos + len(sep)
        
        if best_break <= start:
            best_break = end
        
        chunks.append(text[start:best_break])
        start = best_break - overlap
    
    return chunks


def format_timestamp(seconds: float) -> str:
    """將秒數格式化為 HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_banner(text: str):
    """印出分隔橫幅"""
    width = max(len(text) + 4, 50)
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_progress(current: int, total: int, prefix: str = ""):
    """印出進度"""
    pct = current / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({pct:.1f}%)", end="", flush=True)
    if current == total:
        print()
