"""
Step 3: 使用 LLM 將逐字稿轉換為結構化 Markdown 筆記
支援透過 OpenAI 相容 API 連接本地 Ollama / llama.cpp
"""

import json
import sys
import time
import threading
from pathlib import Path

from openai import OpenAI

from utils import (
    load_config,
    ensure_dirs,
    get_project_root,
    get_stem,
    chunk_text,
    format_timestamp,
    print_banner,
    print_progress,
)


# ============================================================
# Prompt 模板
# ============================================================

SYSTEM_PROMPT_DEFAULT = """You are a professional Android development technical writer. Your task is to convert course transcripts into structured, readable Markdown notes in English.

## Output requirements:
1. Write in clear, professional English
2. Use clear heading hierarchy (##, ###, ####)
3. Bold important concepts: **concept**
4. Use proper code blocks with language tags (```kotlin, ```java, ```xml)
5. Use bullet points for key takeaways
6. Keep important timestamps [MM:SS]
7. Filter out filler words (um, uh, you know, like, so, etc.)
8. Add context to make notes self-contained and independently readable

## Note structure:
- Brief topic introduction
- Core concepts and explanations
- Step-by-step instructions (if applicable)
- Code examples (properly formatted)
- Key takeaways (bullet points)"""

CHUNK_PROMPT_TEMPLATE = """Below is a transcript segment ({chunk_num}/{total_chunks}) from an Android development course.
Please convert it into structured Markdown notes.

Course title: {video_title}

---
Transcript:

{transcript_chunk}

---

Please generate structured Markdown notes based on the transcript above. Guidelines:
- Properly format any code snippets with correct language tags
- Use `backtick` for Android API names, class names, and method names
- Organize key concepts using bullet points
- Keep important timestamps [MM:SS]"""

SUMMARY_PROMPT_TEMPLATE = """Below is the complete notes for an Android development course. Please generate:
1. **Course Summary**: 2-3 sentences summarizing the key points
2. **Core Concepts**: Bullet list of the 5-10 most important takeaways
3. **Key Terms**: List the important terms mentioned with brief explanations
{qa_section}

Course title: {video_title}

---
Notes content:

{full_notes}
"""

ZH_SYSTEM_PROMPT = """你是資深技術編輯。請使用「繁體中文（台灣）」撰寫，並使用台灣慣用技術用詞與語氣（例如：影片、逐字稿、章節、重點整理、程式碼、執行、設定、效能、記憶體）。

請避免中國大陸常見用語（例如：视频、脚本、内存、运行、配置、优化），也避免中英混雜與口語贅字。

目標讀者是台灣工程同仁：內容要精準、短句、可執行，2-3 分鐘可快速掌握。"""

ZH_SUMMARY_PROMPT_TEMPLATE = """以下是 Android 課程筆記內容，請輸出「繁體中文濃縮摘要（Markdown）」。

請使用繁體中文（台灣）與台灣慣用技術詞彙。

請嚴格遵守格式：
## 一句話總結
- 1 句話（不超過 30 字）

## 三行重點
- 共 3 點，每點不超過 20 字

## 快速重點（給同仁）
- 8~12 點條列
- 每點不超過 25 字

## 立即可做
- 3~5 點可直接執行的行動

## 術語速查
- 列出 8~12 個術語，格式：`English`：中文解釋（<=20字）

課程名稱：{video_title}

---
筆記內容：

{full_notes}
"""

QA_SECTION = """4. **Q&A**: Generate 3-5 potential questions and answers based on the content"""


def get_system_prompt(language: str) -> str:
    """Get system prompt based on language setting"""
    return SYSTEM_PROMPT_DEFAULT


def call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    config: dict,
    progress_label: str | None = None,
) -> str:
    """呼叫 LLM API，支援段內心跳進度顯示。"""
    llm_cfg = config.get("llm", {})
    max_tokens = llm_cfg.get("max_tokens", 4096)
    temperature = llm_cfg.get("temperature", 0.3)
    max_retries = llm_cfg.get("max_retries", 3)

    spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    for attempt in range(max_retries):
        request_start = time.time()
        stop_event = threading.Event()
        heartbeat_thread = None

        if progress_label:
            def heartbeat():
                frame_index = 0
                while not stop_event.wait(1.0):
                    elapsed = format_timestamp(time.time() - request_start)
                    frame = spinner_frames[frame_index % len(spinner_frames)]
                    frame_index += 1
                    print(
                        f"\r    {progress_label} {frame} 執行中... 已耗時 {elapsed}",
                        end="",
                        flush=True,
                    )

            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            content = response.choices[0].message.content.strip()

            if progress_label:
                elapsed = format_timestamp(time.time() - request_start)
                print(f"\r    {progress_label} ✓ 完成，耗時 {elapsed}{' ' * 20}")

            return content

        except Exception as e:
            if progress_label:
                elapsed = format_timestamp(time.time() - request_start)
                print(f"\r    {progress_label} ✗ 失敗，耗時 {elapsed}{' ' * 20}")

            print(f"  ⚠ API 呼叫失敗 (嘗試 {attempt + 1}/{max_retries})：{e}")
            if attempt < max_retries - 1:
                wait_seconds = 5 * (attempt + 1)
                print(f"    等待 {wait_seconds} 秒後重試...")
                time.sleep(wait_seconds)
            else:
                raise

        finally:
            stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.2)


def process_transcript(transcript_json_path: Path, config: dict,
                       client: OpenAI) -> tuple[str, str]:
    """處理一份逐字稿，生成 Markdown 筆記。"""
    llm_cfg = config.get("llm", {})
    notes_cfg = config.get("notes", {})
    model = llm_cfg.get("model", "qwen2.5:7b-instruct-q4_K_M")
    chunk_size = llm_cfg.get("chunk_size", 3000)
    chunk_overlap = llm_cfg.get("chunk_overlap", 200)
    request_delay = llm_cfg.get("request_delay", 1.0)
    language = notes_cfg.get("language", "en")
    include_timestamps = notes_cfg.get("include_timestamps", True)
    generate_summary = notes_cfg.get("generate_summary", True)
    generate_qa = notes_cfg.get("generate_qa", True)
    generate_zh_summary = notes_cfg.get("generate_zh_summary", True)

    video_title = transcript_json_path.stem
    system_prompt = get_system_prompt(language)

    with open(transcript_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if include_timestamps:
        full_text = "\n".join(
            f"[{format_timestamp(seg['start'])}] {seg['text']}"
            for seg in segments
        )
    else:
        full_text = "\n".join(seg["text"] for seg in segments)

    chunks = chunk_text(full_text, chunk_size, chunk_overlap)
    total_chunks = len(chunks)
    print(f"  逐字稿共 {len(full_text)} 字，分為 {total_chunks} 段")

    all_notes = []
    generation_start_time = time.time()

    for i, chunk in enumerate(chunks, 1):
        print(f"\n  📝 段落 {i}/{total_chunks} 開始")

        user_prompt = CHUNK_PROMPT_TEMPLATE.format(
            chunk_num=i,
            total_chunks=total_chunks,
            video_title=video_title,
            transcript_chunk=chunk,
        )

        note = call_llm(
            client,
            model,
            system_prompt,
            user_prompt,
            config,
            progress_label=f"段落 {i}/{total_chunks}",
        )
        all_notes.append(note)

        elapsed_total = time.time() - generation_start_time
        avg_per_chunk = elapsed_total / i
        eta_seconds = avg_per_chunk * (total_chunks - i)

        print_progress(i, total_chunks, prefix="  整體進度")
        print(f"    已完成 {i}/{total_chunks}，預估剩餘 {format_timestamp(eta_seconds)}")

        if i < total_chunks:
            time.sleep(request_delay)

    combined_notes = "\n\n---\n\n".join(all_notes)
    total_duration = format_timestamp(segments[-1]["end"]) if segments else "00:00"

    summary_section = ""
    notes_for_summary = combined_notes
    if generate_summary:
        print("\n  📋 生成課程摘要...")

        if len(notes_for_summary) > 6000:
            notes_for_summary = (
                combined_notes[:3000]
                + "\n\n[...content omitted...]\n\n"
                + combined_notes[-3000:]
            )

        qa_part = QA_SECTION if generate_qa else ""
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            video_title=video_title,
            full_notes=notes_for_summary,
            qa_section=qa_part,
        )

        summary_section = call_llm(
            client,
            model,
            system_prompt,
            summary_prompt,
            config,
            progress_label="課程摘要",
        )

    final_md = f"""# {video_title}

> **Duration**: {total_duration}  
> **Generated by**: {model}  

---

{summary_section}

---

## Detailed Notes

{combined_notes}
"""

    zh_summary_md = ""
    if generate_zh_summary:
        print("\n  🇹🇼 生成繁中濃縮摘要...")
        zh_input = notes_for_summary if generate_summary else combined_notes[:6000]
        zh_prompt = ZH_SUMMARY_PROMPT_TEMPLATE.format(
            video_title=video_title,
            full_notes=zh_input,
        )
        zh_content = call_llm(
            client,
            model,
            ZH_SYSTEM_PROMPT,
            zh_prompt,
            config,
            progress_label="繁中濃縮摘要",
        )
        zh_summary_md = f"""# {video_title}｜繁中濃縮摘要

> **Source**: {video_title} transcript  
> **Generated by**: {model}

---

{zh_content}
"""

    return final_md, zh_summary_md


def main():
    print_banner("Step 3: 生成 Markdown 筆記")

    config = load_config()
    ensure_dirs(config)

    root = get_project_root()
    transcripts_dir = root / config["paths"]["transcripts_dir"]
    notes_dir = root / config["paths"]["notes_dir"]
    llm_cfg = config.get("llm", {})

    api_base = llm_cfg.get("api_base", "http://localhost:11434/v1")
    api_key = llm_cfg.get("api_key", "not-needed")

    print(f"LLM API 端點：{api_base}")
    print(f"模型：{llm_cfg.get('model', 'N/A')}\n")

    client = OpenAI(base_url=api_base, api_key=api_key)

    try:
        models = client.models.list()
        print("✓ 成功連接 LLM 伺服器")
        available = [m.id for m in models.data]
        print(f"  可用模型：{', '.join(available[:5])}")
    except Exception as e:
        print(f"✗ 無法連接 LLM 伺服器：{e}")
        print("  請確認 Ollama 或 llama.cpp 已啟動")
        print("  Ollama: ollama serve")
        print("  llama.cpp: ./llama-server -m model.gguf --host 0.0.0.0 --port 8080")
        sys.exit(1)

    transcript_files = sorted(transcripts_dir.glob("*.json"))

    if not transcript_files:
        print("\n✗ 在 transcripts/ 目錄中找不到任何逐字稿。請先執行 Step 2。")
        sys.exit(1)

    print(f"\n找到 {len(transcript_files)} 份逐字稿\n")

    success_count = 0
    fail_count = 0

    for transcript_path in transcript_files:
        stem = get_stem(transcript_path)
        output_path = notes_dir / f"{stem}.md"
        zh_summary_path = notes_dir / f"{stem}.zh-TW.summary.md"

        if output_path.exists() and zh_summary_path.exists():
            print(f"⏭ 跳過（已存在）：{stem}")
            success_count += 1
            continue

        print(f"\n{'─' * 50}")
        print(f"📓 處理：{stem}")

        try:
            markdown_en, markdown_zh = process_transcript(transcript_path, config, client)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_en)

            if markdown_zh:
                with open(zh_summary_path, "w", encoding="utf-8") as f:
                    f.write(markdown_zh)

            print(f"  ✓ 英文筆記已儲存：{output_path.name}")
            if markdown_zh:
                print(f"  ✓ 中文摘要已儲存：{zh_summary_path.name}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ 錯誤：{e}")
            fail_count += 1

    print(f"\n{'=' * 50}")
    print(f"完成！成功：{success_count}，失敗：{fail_count}")
    print(f"筆記儲存於：{notes_dir}")


if __name__ == "__main__":
    main()
