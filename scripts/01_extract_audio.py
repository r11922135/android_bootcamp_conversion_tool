"""
Step 1: 從影片中抽取音軌
使用 FFmpeg 將影片轉換為 16kHz mono WAV 格式（Whisper 最佳輸入格式）
"""

import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

from utils import load_config, ensure_dirs, get_video_files, get_stem, get_project_root, print_banner


def extract_audio(video_path: Path, output_path: Path, config: dict) -> bool:
    """
    使用 FFmpeg 從影片抽取音軌
    
    Args:
        video_path: 影片路徑
        output_path: 音檔輸出路徑
        config: FFmpeg 設定
    
    Returns:
        bool: 是否成功
    """
    ffmpeg_cfg = config.get("ffmpeg", {})
    sample_rate = ffmpeg_cfg.get("sample_rate", 16000)
    channels = ffmpeg_cfg.get("channels", 1)
    ffmpeg_path = ffmpeg_cfg.get("ffmpeg_path", "ffmpeg")
    
    cmd = [
        ffmpeg_path,
        "-i", str(video_path),
        "-vn",                          # 不要影片
        "-acodec", "pcm_s16le",         # 16-bit PCM
        "-ar", str(sample_rate),        # 取樣率
        "-ac", str(channels),           # 聲道數
        "-y",                           # 覆蓋已存在的檔案
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            errors="replace"
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ FFmpeg 錯誤：{e.stderr[:500]}")
        return False
    except FileNotFoundError:
        print(f"\n✗ 找不到 FFmpeg。請確認 FFmpeg 已安裝並在 PATH 中。")
        print(f"  下載網址：https://www.gyan.dev/ffmpeg/builds/")
        return False


def main():
    print_banner("Step 1: 抽取音軌")
    
    config = load_config()
    ensure_dirs(config)
    
    root = get_project_root()
    audio_dir = root / config["paths"]["audio_dir"]
    video_files = get_video_files(config)
    
    if not video_files:
        print("✗ 在 videos/ 目錄中找不到任何影片檔案。")
        print(f"  支援的格式：{', '.join(config.get('video_extensions', []))}")
        sys.exit(1)
    
    print(f"找到 {len(video_files)} 個影片檔案\n")
    
    success_count = 0
    fail_count = 0
    
    for video_path in tqdm(video_files, desc="抽取音軌", unit="file"):
        stem = get_stem(video_path)
        audio_path = audio_dir / f"{stem}.wav"
        
        # 如果音檔已存在，跳過
        if audio_path.exists():
            tqdm.write(f"⏭ 跳過（已存在）：{stem}")
            success_count += 1
            continue
        
        tqdm.write(f"🎵 處理中：{video_path.name}")
        
        if extract_audio(video_path, audio_path, config):
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            tqdm.write(f"  ✓ 完成：{audio_path.name} ({file_size_mb:.1f} MB)")
            success_count += 1
        else:
            tqdm.write(f"  ✗ 失敗：{video_path.name}")
            fail_count += 1
    
    print(f"\n{'='*50}")
    print(f"完成！成功：{success_count}，失敗：{fail_count}")
    print(f"音檔儲存於：{audio_dir}")


if __name__ == "__main__":
    main()
