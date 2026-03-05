"""
Step 3: 使用 LLM 將逐字稿轉換為結構化 Markdown 筆記
支援透過 OpenAI 相容 API 連接本地 Ollama / llama.cpp
"""

import json
import re
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

ZH_REWRITE_CHUNK_PROMPT_TEMPLATE = """以下是英文版技術筆記片段（{chunk_num}/{total_chunks}）。
請改寫成「繁體中文（台灣）」的結構化技術筆記，維持與英文相同的資訊密度與章節層次。

課程名稱：{video_title}

---
英文筆記片段：

{english_notes_chunk}

---

改寫要求（請嚴格遵守）：
- 不要逐句翻譯，請重組語句與段落，寫成可閱讀的技術筆記
- 使用 `##` / `###` / `####` 標題階層
- 保留重要時間戳記（例如 [MM:SS]）
- 程式碼區塊使用正確語言標籤（```kotlin / ```java / ```xml）
- Android API、類別、方法名稱保留英文並用 `backtick`
- 絕對不要輸出簡體中文
- 不要把整份內容包在 ```markdown code fence``` 裡
- 內容要具備技術脈絡，不能只列翻譯句

術語偏好（可直接採用）：
{glossary_text}
"""

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

目標讀者是台灣工程同仁：內容要精準、可執行，保留技術脈絡，不要過度壓縮。"""

ZH_SUMMARY_PROMPT_TEMPLATE = """以下是 Android 課程完整筆記。請產出「繁體中文（台灣）Markdown 摘要」，需與英文摘要章節一致且資訊完整。

請輸出以下區塊（標題與順序固定）：
1. **課程摘要**：2-3 句，完整描述主題與成果（不要只有一句口號）
2. **核心概念**：條列 5-10 個最重要重點，每點 1-2 句說明
3. **關鍵術語**：列出重要術語，格式為 `English Term`：繁中解釋（可含用途）
{qa_section}

格式要求：
- 使用 `##`/`###` 標題層級
- 內容必須保留技術脈絡與上下文，不可只寫極短句
- 術語、類別、方法名稱保留英文原名並可附繁中說明

課程名稱：{video_title}

---
筆記內容：

{full_notes}
"""

ZH_QA_SECTION = """4. **延伸問答**：根據內容產出 3-5 題可能問題與答案（每題答案 1-3 句）"""

QA_SECTION = """4. **Q&A**: Generate 3-5 potential questions and answers based on the content"""

ZH_REVIEW_PROMPT_TEMPLATE = """請修正以下繁中筆記段落，讓品質符合台灣工程團隊閱讀需求。

修正規則：
- 僅使用繁體中文（台灣），禁止簡體字
- 保留技術內容與章節層次，不可縮短成摘要
- 不要逐句翻譯口吻，要改寫成自然技術筆記
- 不要輸出包住全文的 ```markdown code fence```
- 保留原有程式碼區塊與時間戳記

原文：
{bad_chunk}
"""

SIMPLIFIED_HINT_WORDS = [
    "欢迎", "视频", "脚本", "内存", "运行", "配置", "优化", "阶段",
    "系统服务", "发布", "实时", "语音", "翻译", "应用程序", "数据",
]


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
    max_tokens_override: int | None = None,
    temperature_override: float | None = None,
    max_retries_override: int | None = None,
) -> str:
    """呼叫 LLM API，支援段內心跳進度顯示。"""
    llm_cfg = config.get("llm", {})
    max_tokens = max_tokens_override if max_tokens_override is not None else llm_cfg.get("max_tokens", 4096)
    temperature = temperature_override if temperature_override is not None else llm_cfg.get("temperature", 0.3)
    max_retries = max_retries_override if max_retries_override is not None else llm_cfg.get("max_retries", 3)

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


def _build_glossary_text(notes_cfg: dict) -> str:
    glossary = notes_cfg.get("zh_glossary", [])
    if not glossary:
        return "- Activity：活動\n- ViewModel：檢視模型\n- Thread：執行緒\n- Memory：記憶體\n- Pipeline：處理流程"
    if isinstance(glossary, list):
        return "\n".join(f"- {item}" for item in glossary)
    return str(glossary)


def _unwrap_outer_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```markdown") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text.strip()


def _zh_quality_issues(text: str, min_chars: int) -> list[str]:
    issues = []
    stripped = text.strip()
    if len(stripped) < min_chars:
        issues.append("內容過短")
    if stripped.startswith("```markdown") or (stripped.startswith("```") and stripped.endswith("```")):
        issues.append("整段被 code fence 包住")
    for word in SIMPLIFIED_HINT_WORDS:
        if word in stripped:
            issues.append(f"疑似簡體/翻譯腔詞彙：{word}")
            break
    return issues


def _extract_duration_from_markdown(markdown_text: str, fallback: str = "00:00") -> str:
    match = re.search(r"\*\*Duration\*\*:\s*([0-9]{2}:[0-9]{2}(?::[0-9]{2})?)", markdown_text)
    if match:
        return match.group(1)
    return fallback


def _get_zh_settings(config: dict, default_model: str) -> dict:
    llm_cfg = config.get("llm", {})
    notes_cfg = config.get("notes", {})
    llm_zh_cfg = config.get("llm_zh", {})

    return {
        "model": llm_zh_cfg.get("model", default_model),
        "temperature": llm_zh_cfg.get("temperature", 0.15),
        "max_tokens": llm_zh_cfg.get("max_tokens", llm_cfg.get("max_tokens", 4096)),
        "max_retries": llm_zh_cfg.get("max_retries", llm_cfg.get("max_retries", 3)),
        "chunk_size": llm_zh_cfg.get("chunk_size", 2600),
        "chunk_overlap": llm_zh_cfg.get("chunk_overlap", 180),
        "min_chunk_chars": llm_zh_cfg.get("min_chunk_chars", 260),
        "request_delay": llm_cfg.get("request_delay", 1.0),
        "generate_qa": notes_cfg.get("generate_qa", True),
        "glossary_text": _build_glossary_text(notes_cfg),
    }


def generate_zh_notes_from_english_notes(
    english_notes: str,
    video_title: str,
    total_duration: str,
    config: dict,
    client: OpenAI,
    default_model: str,
) -> str:
    settings = _get_zh_settings(config, default_model)
    zh_model = settings["model"]
    zh_source_chunks = chunk_text(english_notes, settings["chunk_size"], settings["chunk_overlap"])
    total_zh_chunks = len(zh_source_chunks)
    zh_detail_chunks = []
    zh_start_time = time.time()

    print("\n  🇹🇼 以英文筆記生成繁中詳細筆記...")

    for i, en_chunk in enumerate(zh_source_chunks, 1):
        zh_user_prompt = ZH_REWRITE_CHUNK_PROMPT_TEMPLATE.format(
            chunk_num=i,
            total_chunks=total_zh_chunks,
            video_title=video_title,
            english_notes_chunk=en_chunk,
            glossary_text=settings["glossary_text"],
        )
        zh_note = call_llm(
            client,
            zh_model,
            ZH_SYSTEM_PROMPT,
            zh_user_prompt,
            config,
            progress_label=f"繁中改寫 {i}/{total_zh_chunks}",
            max_tokens_override=settings["max_tokens"],
            temperature_override=settings["temperature"],
            max_retries_override=settings["max_retries"],
        )
        zh_note = _unwrap_outer_markdown_fence(zh_note)
        issues = _zh_quality_issues(zh_note, settings["min_chunk_chars"])

        if issues:
            fix_prompt = ZH_REVIEW_PROMPT_TEMPLATE.format(bad_chunk=zh_note)
            zh_note = call_llm(
                client,
                zh_model,
                ZH_SYSTEM_PROMPT,
                fix_prompt,
                config,
                progress_label=f"繁中修正 {i}/{total_zh_chunks}",
                max_tokens_override=settings["max_tokens"],
                temperature_override=settings["temperature"],
                max_retries_override=settings["max_retries"],
            )
            zh_note = _unwrap_outer_markdown_fence(zh_note)

        zh_detail_chunks.append(zh_note)
        elapsed_total = time.time() - zh_start_time
        avg_per_chunk = elapsed_total / i
        eta_seconds = avg_per_chunk * (total_zh_chunks - i)
        print_progress(i, total_zh_chunks, prefix="  繁中詳細進度")
        print(f"    已完成 {i}/{total_zh_chunks}，預估剩餘 {format_timestamp(eta_seconds)}")

        if i < total_zh_chunks:
            time.sleep(settings["request_delay"])

    combined_notes_zh = "\n\n---\n\n".join(zh_detail_chunks)

    print("\n  🇹🇼 生成繁中課程摘要...")
    zh_input = combined_notes_zh
    if len(zh_input) > 6000:
        zh_input = (
            zh_input[:3000]
            + "\n\n[...content omitted...]\n\n"
            + zh_input[-3000:]
        )

    zh_qa_part = ZH_QA_SECTION if settings["generate_qa"] else ""
    zh_prompt = ZH_SUMMARY_PROMPT_TEMPLATE.format(
        video_title=video_title,
        full_notes=zh_input,
        qa_section=zh_qa_part,
    )
    zh_content = call_llm(
        client,
        zh_model,
        ZH_SYSTEM_PROMPT,
        zh_prompt,
        config,
        progress_label="繁中課程摘要",
        max_tokens_override=settings["max_tokens"],
        temperature_override=settings["temperature"],
        max_retries_override=settings["max_retries"],
    )
    zh_content = _unwrap_outer_markdown_fence(zh_content)

    return f"""# {video_title}｜繁體中文筆記

> **Duration**: {total_duration}  
> **Generated by**: {zh_model}
> **Source**: {video_title} transcript  

---

{zh_content}

---

## 詳細筆記

{combined_notes_zh}
"""


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
    generate_zh_notes = notes_cfg.get("generate_zh_summary", True)

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
    if generate_zh_notes:
        zh_summary_md = generate_zh_notes_from_english_notes(
            combined_notes,
            video_title,
            total_duration,
            config,
            client,
            model,
        )

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

        if output_path.exists():
            # 英文已存在時，僅補跑中文，避免重跑英文流程。
            zh_ready = False
            if zh_summary_path.exists():
                try:
                    zh_ready = "## 詳細筆記" in zh_summary_path.read_text(encoding="utf-8")
                except Exception:
                    zh_ready = False

            if zh_ready:
                print(f"⏭ 跳過（已存在）：{stem}")
                success_count += 1
                continue

            print(f"\n{'─' * 50}")
            print(f"📓 補產中文：{stem}")

            try:
                english_markdown = output_path.read_text(encoding="utf-8")
                duration = _extract_duration_from_markdown(english_markdown, fallback="00:00")
                detailed_marker = "\n## Detailed Notes\n"
                if detailed_marker in english_markdown:
                    english_source = english_markdown.split(detailed_marker, 1)[1].strip()
                else:
                    english_source = english_markdown

                markdown_zh = generate_zh_notes_from_english_notes(
                    english_source,
                    stem,
                    duration,
                    config,
                    client,
                    llm_cfg.get("model", "N/A"),
                )
                with open(zh_summary_path, "w", encoding="utf-8") as f:
                    f.write(markdown_zh)
                print(f"  ✓ 中文筆記已儲存：{zh_summary_path.name}")
                success_count += 1
                continue
            except Exception as e:
                print(f"  ✗ 錯誤：{e}")
                fail_count += 1
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
                print(f"  ✓ 中文筆記已儲存：{zh_summary_path.name}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ 錯誤：{e}")
            fail_count += 1

    print(f"\n{'=' * 50}")
    print(f"完成！成功：{success_count}，失敗：{fail_count}")
    print(f"筆記儲存於：{notes_dir}")


if __name__ == "__main__":
    main()
