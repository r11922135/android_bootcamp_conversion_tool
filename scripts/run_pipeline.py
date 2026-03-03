"""
一鍵全自動流程：影片 → 音檔 → 逐字稿 → Markdown 筆記
依序執行 Step 1 ~ Step 3
"""

import sys
import time
import argparse
from pathlib import Path

# 確保 scripts 目錄在 Python path 中
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, ensure_dirs, get_video_files, get_project_root, print_banner


def main():
    parser = argparse.ArgumentParser(
        description="Android Bootcamp 影片轉 Markdown 筆記 — 一鍵全自動流程"
    )
    parser.add_argument(
        "--skip-audio", action="store_true",
        help="跳過 Step 1 (音軌抽取)"
    )
    parser.add_argument(
        "--skip-transcribe", action="store_true",
        help="跳過 Step 2 (語音轉文字)"
    )
    parser.add_argument(
        "--skip-notes", action="store_true",
        help="跳過 Step 3 (筆記生成)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="指定設定檔路徑 (預設: config.yaml)"
    )
    args = parser.parse_args()
    
    print_banner("Android Bootcamp 影片轉 Markdown 筆記")
    print()
    
    config = load_config()
    ensure_dirs(config)
    
    video_files = get_video_files(config)
    print(f"📁 找到 {len(video_files)} 個影片檔案")
    
    if not video_files:
        print("\n✗ 在 videos/ 目錄中找不到任何影片檔案。")
        print("  請將影片放入 videos/ 目錄後重新執行。")
        sys.exit(1)
    
    for vf in video_files:
        print(f"   • {vf.name}")
    print()
    
    total_start = time.time()
    
    # ── Step 1: 抽取音軌 ──
    if not args.skip_audio:
        print("\n" + "=" * 60)
        step_start = time.time()
        
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "01_extract_audio.py")],
            cwd=str(get_project_root()),
        )
        if result.returncode != 0:
            print("✗ Step 1 失敗，中止流程。")
            sys.exit(1)
        
        step_time = time.time() - step_start
        print(f"⏱ Step 1 耗時：{step_time:.1f} 秒\n")
    else:
        print("⏭ 跳過 Step 1 (音軌抽取)\n")
    
    # ── Step 2: 語音轉文字 ──
    if not args.skip_transcribe:
        print("\n" + "=" * 60)
        step_start = time.time()
        
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "02_transcribe.py")],
            cwd=str(get_project_root()),
        )
        if result.returncode != 0:
            print("✗ Step 2 失敗，中止流程。")
            sys.exit(1)
        
        step_time = time.time() - step_start
        print(f"⏱ Step 2 耗時：{step_time:.1f} 秒\n")
    else:
        print("⏭ 跳過 Step 2 (語音轉文字)\n")
    
    # ── Step 3: 生成筆記 ──
    if not args.skip_notes:
        print("\n" + "=" * 60)
        step_start = time.time()
        
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "03_generate_notes.py")],
            cwd=str(get_project_root()),
        )
        if result.returncode != 0:
            print("✗ Step 3 失敗，中止流程。")
            sys.exit(1)
        
        step_time = time.time() - step_start
        print(f"⏱ Step 3 耗時：{step_time:.1f} 秒\n")
    else:
        print("⏭ 跳過 Step 3 (筆記生成)\n")
    
    # ── 完成 ──
    total_time = time.time() - total_start
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "=" * 60)
    print_banner("全部完成！")
    print(f"\n⏱ 總耗時：{minutes} 分 {seconds} 秒")
    
    root = get_project_root()
    notes_dir = root / config["paths"]["notes_dir"]
    notes = list(notes_dir.glob("*.md"))
    
    if notes:
        print(f"\n📓 產生了 {len(notes)} 份筆記：")
        for note in sorted(notes):
            size_kb = note.stat().st_size / 1024
            print(f"   • {note.name} ({size_kb:.1f} KB)")
    
    print(f"\n📂 筆記目錄：{notes_dir}")


if __name__ == "__main__":
    main()
