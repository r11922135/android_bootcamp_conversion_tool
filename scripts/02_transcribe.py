"""
Step 2: 語音轉文字
使用 faster-whisper 在 RTX 4080 上進行語音辨識，輸出逐字稿和 SRT 字幕檔
"""

import sys
import json
import time
import threading
from pathlib import Path
from tqdm import tqdm

from utils import (
    load_config, ensure_dirs, get_project_root, get_stem,
    format_timestamp, print_banner
)


def transcribe_audio(audio_path: Path, config: dict) -> list[dict]:
    """
    使用 faster-whisper 將音檔轉為文字
    
    Args:
        audio_path: 音檔路徑
        config: Whisper 設定
    
    Returns:
        list[dict]: 包含 start, end, text 的段落列表
    """
    from faster_whisper import WhisperModel
    
    whisper_cfg = config.get("whisper", {})
    model_name = whisper_cfg.get("model", "large-v3-turbo")
    device = whisper_cfg.get("device", "cuda")
    compute_type = whisper_cfg.get("compute_type", "float16")
    language = whisper_cfg.get("language", None)
    beam_size = whisper_cfg.get("beam_size", 5)
    vad_filter = whisper_cfg.get("vad_filter", True)
    initial_prompt = whisper_cfg.get("initial_prompt", None)
    
    print(f"  載入模型：{model_name} ({device}, {compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    
    print(f"  開始辨識...")
    segments_gen, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
        word_timestamps=False,
    )
    
    if info.language:
        print(f"  偵測到語言：{info.language} (機率：{info.language_probability:.2%})")

    total_duration = float(getattr(info, "duration", 0.0) or 0.0)
    start_wall = time.time()
    stop_event = threading.Event()
    progress_state = {"segments": 0, "processed_seconds": 0.0}

    def heartbeat():
        while not stop_event.wait(2.0):
            elapsed = max(time.time() - start_wall, 0.001)
            segments_count = progress_state["segments"]
            processed = progress_state["processed_seconds"]
            speed = processed / elapsed if processed > 0 else 0.0

            if total_duration > 0:
                pct = min(processed / total_duration * 100, 100.0)
                eta = (total_duration - processed) / speed if speed > 0 else 0.0
                print(
                    f"\r  辨識進度：{pct:5.1f}% | "
                    f"已處理 {format_timestamp(processed)}/{format_timestamp(total_duration)} | "
                    f"段落 {segments_count} | 速率 {speed:.2f}x | "
                    f"預估剩餘 {format_timestamp(max(eta, 0.0))}",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\r  辨識進度：已處理 {format_timestamp(processed)} | "
                    f"段落 {segments_count} | 速率 {speed:.2f}x",
                    end="",
                    flush=True,
                )

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    segments = []
    try:
        for seg in segments_gen:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })
            progress_state["segments"] = len(segments)
            progress_state["processed_seconds"] = float(seg.end)
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=0.2)
        if progress_state["segments"] > 0:
            print()

    return segments


def segments_to_txt(segments: list[dict], include_timestamps: bool = True) -> str:
    """將段落列表轉為純文字"""
    lines = []
    for seg in segments:
        if include_timestamps:
            ts = format_timestamp(seg["start"])
            lines.append(f"[{ts}] {seg['text']}")
        else:
            lines.append(seg["text"])
    return "\n".join(lines)


def segments_to_srt(segments: list[dict]) -> str:
    """將段落列表轉為 SRT 字幕格式"""
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = _seconds_to_srt_time(seg["start"])
        end = _seconds_to_srt_time(seg["end"])
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(seg["text"])
        srt_lines.append("")
    return "\n".join(srt_lines)


def _seconds_to_srt_time(seconds: float) -> str:
    """秒數轉 SRT 時間格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main():
    print_banner("Step 2: 語音轉文字")
    
    config = load_config()
    ensure_dirs(config)
    
    root = get_project_root()
    audio_dir = root / config["paths"]["audio_dir"]
    transcripts_dir = root / config["paths"]["transcripts_dir"]
    whisper_cfg = config.get("whisper", {})
    
    # 取得所有音檔
    audio_files = sorted(audio_dir.glob("*.wav"))
    
    if not audio_files:
        print("✗ 在 audio/ 目錄中找不到任何音檔。請先執行 Step 1。")
        sys.exit(1)
    
    print(f"找到 {len(audio_files)} 個音檔\n")
    
    success_count = 0
    fail_count = 0
    
    for audio_path in audio_files:
        stem = get_stem(audio_path)
        txt_path = transcripts_dir / f"{stem}.txt"
        srt_path = transcripts_dir / f"{stem}.srt"
        json_path = transcripts_dir / f"{stem}.json"
        
        # 如果逐字稿已存在，跳過
        if txt_path.exists() and json_path.exists():
            print(f"⏭ 跳過（已存在）：{stem}")
            success_count += 1
            continue
        
        print(f"\n🎤 辨識中：{audio_path.name}")
        
        try:
            segments = transcribe_audio(audio_path, config)
            
            if not segments:
                print(f"  ⚠ 未偵測到任何語音內容")
                fail_count += 1
                continue
            
            # 儲存 JSON（完整資料，含時間戳記）
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
            
            # 儲存純文字
            if whisper_cfg.get("output_txt", True):
                txt_content = segments_to_txt(segments, include_timestamps=True)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(txt_content)
                print(f"  ✓ TXT：{txt_path.name}")
            
            # 儲存 SRT
            if whisper_cfg.get("output_srt", True):
                srt_content = segments_to_srt(segments)
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                print(f"  ✓ SRT：{srt_path.name}")
            
            total_duration = segments[-1]["end"] if segments else 0
            print(f"  ✓ 共 {len(segments)} 段，總時長 {format_timestamp(total_duration)}")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ 錯誤：{e}")
            fail_count += 1
    
    print(f"\n{'='*50}")
    print(f"完成！成功：{success_count}，失敗：{fail_count}")
    print(f"逐字稿儲存於：{transcripts_dir}")


if __name__ == "__main__":
    main()
