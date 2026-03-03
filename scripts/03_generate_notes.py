"""
Step 3: 使用 LLM 將逐字稿轉換為結構化 Markdown 筆記
支援透過 OpenAI 相容 API 連接本地 Ollama / llama.cpp
"""

import json
import sys
import time
from pathlib import Path
from openai import OpenAI

from utils import (
    load_config, ensure_dirs, get_project_root, get_stem,
    chunk_text, format_timestamp, print_banner
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

QA_SECTION = """4. **Q&A**: Generate 3-5 potential questions and answers based on the content"""


def get_system_prompt(language: str) -> str:
    """Get system prompt based on language setting"""
    return SYSTEM_PROMPT_DEFAULT


def call_llm(client: OpenAI, model: str, system_prompt: str, 
             user_prompt: str, config: dict) -> str:
    """
    呼叫 LLM API
    
    Args:
        client: OpenAI client
        model: 模型名稱
        system_prompt: 系統提示詞
        user_prompt: 使用者提示詞
        config: LLM 設定
    
    Returns:
        str: LLM 回應文字
    """
    llm_cfg = config.get("llm", {})
    max_tokens = llm_cfg.get("max_tokens", 4096)
    temperature = llm_cfg.get("temperature", 0.3)
    max_retries = llm_cfg.get("max_retries", 3)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ API 呼叫失敗 (嘗試 {attempt + 1}/{max_retries})：{e}")
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    等待 {wait} 秒後重試...")
                time.sleep(wait)
            else:
                raise


def process_transcript(transcript_json_path: Path, config: dict,
                       client: OpenAI) -> str:
    """
    處理一份逐字稿，生成 Markdown 筆記
    
    Args:
        transcript_json_path: 逐字稿 JSON 路徑
        config: 完整設定
        client: OpenAI client
    
    Returns:
        str: 完整的 Markdown 筆記
    """
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
    
    video_title = transcript_json_path.stem
    system_prompt = get_system_prompt(language)
    
    # 讀取逐字稿
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    
    # 轉為帶時間戳記的純文字
    if include_timestamps:
        full_text = "\n".join(
            f"[{format_timestamp(seg['start'])}] {seg['text']}" 
            for seg in segments
        )
    else:
        full_text = "\n".join(seg["text"] for seg in segments)
    
    # 分段
    chunks = chunk_text(full_text, chunk_size, chunk_overlap)
    total_chunks = len(chunks)
    print(f"  逐字稿共 {len(full_text)} 字，分為 {total_chunks} 段")
    
    # 逐段生成筆記
    all_notes = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  📝 生成筆記 ({i}/{total_chunks})...", end=" ", flush=True)
        
        user_prompt = CHUNK_PROMPT_TEMPLATE.format(
            chunk_num=i,
            total_chunks=total_chunks,
            video_title=video_title,
            transcript_chunk=chunk,
        )
        
        note = call_llm(client, model, system_prompt, user_prompt, config)
        all_notes.append(note)
        print("✓")
        
        if i < total_chunks:
            time.sleep(request_delay)
    
    # 合併所有段落筆記
    combined_notes = "\n\n---\n\n".join(all_notes)
    
    # 計算影片總時長
    total_duration = format_timestamp(segments[-1]["end"]) if segments else "00:00"
    
    # 生成摘要
    summary_section = ""
    if generate_summary:
        print(f"  📋 生成課程摘要...", end=" ", flush=True)
        
        # 如果筆記太長，只取前後各 2000 字
        notes_for_summary = combined_notes
        if len(notes_for_summary) > 6000:
            notes_for_summary = (
                combined_notes[:3000] + 
                "\n\n[...content omitted...]\n\n" + 
                combined_notes[-3000:]
            )
        
        qa_part = QA_SECTION if generate_qa else ""
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            video_title=video_title,
            full_notes=notes_for_summary,
            qa_section=qa_part,
        )
        
        summary_section = call_llm(client, model, system_prompt, summary_prompt, config)
        print("✓")
    
    # 組合最終 Markdown
    final_md = f"""# {video_title}

> **Duration**: {total_duration}  
> **Generated by**: {model}  

---

{summary_section}

---

## Detailed Notes

{combined_notes}
"""
    
    return final_md


def main():
    print_banner("Step 3: 生成 Markdown 筆記")
    
    config = load_config()
    ensure_dirs(config)
    
    root = get_project_root()
    transcripts_dir = root / config["paths"]["transcripts_dir"]
    notes_dir = root / config["paths"]["notes_dir"]
    llm_cfg = config.get("llm", {})
    
    # 初始化 OpenAI client（連接本地 Ollama/llama.cpp）
    api_base = llm_cfg.get("api_base", "http://localhost:11434/v1")
    api_key = llm_cfg.get("api_key", "not-needed")
    
    print(f"LLM API 端點：{api_base}")
    print(f"模型：{llm_cfg.get('model', 'N/A')}\n")
    
    client = OpenAI(base_url=api_base, api_key=api_key)
    
    # 測試連線
    try:
        models = client.models.list()
        print(f"✓ 成功連接 LLM 伺服器")
        available = [m.id for m in models.data]
        print(f"  可用模型：{', '.join(available[:5])}")
    except Exception as e:
        print(f"✗ 無法連接 LLM 伺服器：{e}")
        print(f"  請確認 Ollama 或 llama.cpp 已啟動")
        print(f"  Ollama: ollama serve")
        print(f"  llama.cpp: ./llama-server -m model.gguf --host 0.0.0.0 --port 8080")
        sys.exit(1)
    
    # 取得所有逐字稿
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
        
        # 如果筆記已存在，跳過
        if output_path.exists():
            print(f"⏭ 跳過（已存在）：{stem}")
            success_count += 1
            continue
        
        print(f"\n{'─'*50}")
        print(f"📓 處理：{stem}")
        
        try:
            markdown = process_transcript(transcript_path, config, client)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            
            print(f"  ✓ 筆記已儲存：{output_path.name}")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ 錯誤：{e}")
            fail_count += 1
    
    print(f"\n{'='*50}")
    print(f"完成！成功：{success_count}，失敗：{fail_count}")
    print(f"筆記儲存於：{notes_dir}")


if __name__ == "__main__":
    main()
