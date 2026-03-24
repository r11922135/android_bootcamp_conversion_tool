"""
Step 3: 使用 LLM 將逐字稿轉換為結構化 Markdown 筆記
支援透過 OpenAI 相容 API 連接本地 Ollama / llama.cpp
"""

import json
import re
import shutil
import subprocess
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
The output is generated chunk-by-chunk and later concatenated, so each chunk must be merge-friendly, evidence-grounded, and avoid boilerplate repetition.

## Output requirements:
1. Write in clear, professional English
2. Use clear heading hierarchy (##, ###, ####)
3. Bold important concepts: **concept**
4. If code blocks are allowed for this run and the source explicitly contains code, use proper code blocks with language tags (```kotlin, ```java, ```xml)
5. Use bullet points for key takeaways
6. Keep timestamps only when they materially help locate an important announcement, requirement, transition, or claim
7. Filter out filler words (um, uh, you know, like, so, etc.)
8. Add context to make notes self-contained and independently readable
9. Ignore non-verbal transcript cues like [Music], [Applause], [Laughter], intro/outro jingles, and pure ambience descriptions unless they are technically relevant
10. Prefer concrete facts: feature/API name, behavior, constraints, owner, version, timeline
11. Avoid generic narrative filler and template phrases
12. Preserve certainty level from source: `Available`, `Preview`, `Planned/Targeted`, `Call to action`, or uncertain / needs verification
13. If the source is fragmentary or ambiguous, stay conservative and avoid inventing precision

## Style constraints:
- Do not use generic lead-ins such as "This section covers", "In this section", or "The transcript highlights".
- Avoid generic headings such as `Introduction`, `Overview`, `Conclusion`, `Key Takeaways`, or `Relevant Slide Snippets` unless the source explicitly uses that section title.
- Do not repeat the course title inside chunk output.
- If chunk overlap repeats information, keep only the most specific version once and move on.
- Do not turn plans, targets, requests for feedback, or early previews into shipped facts.
- Do not invent business impact, action items, timelines, or requirements unless the speaker states them.
- Do not output orphan headings, broken titles, half-finished bullets, or partial sentences."""

SYSTEM_PROMPT_ONE_SHOT = """You are a professional Android development technical writer. Your task is to convert one full Android bootcamp transcript into structured, readable Markdown notes in English.
The output is generated in one pass for the entire session, so it must preserve the full arc of the talk, stay evidence-grounded, and avoid premature stopping.

## Output requirements:
1. Write in clear, professional English
2. Use clear heading hierarchy (##, ###, ####)
3. Bold important concepts: **concept**
4. Use bullet points for key takeaways
5. Keep timestamps only when they materially help locate an important announcement, requirement, transition, or claim
6. Filter out filler words (um, uh, you know, like, so, etc.)
7. Add context to make notes self-contained and independently readable
8. Ignore non-verbal transcript cues like [Music], [Applause], [Laughter], intro/outro jingles, and pure ambience descriptions unless they are technically relevant
9. Prefer concrete facts: feature/API name, behavior, constraints, owner, version, timeline
10. Avoid generic narrative filler and template phrases
11. Preserve certainty level from source: `Available`, `Preview`, `Planned/Targeted`, `Call to action`, or uncertain / needs verification
12. If the source is fragmentary or ambiguous, stay conservative and avoid inventing precision

## Style constraints:
- Do not use generic lead-ins such as "This section covers", "In this section", or "The transcript highlights".
- Avoid generic headings such as `Introduction`, `Overview`, `Conclusion`, `Key Takeaways`, or `Relevant Slide Snippets` unless the source explicitly uses that section title.
- Do not repeat the course title inside the output body.
- Do not turn plans, targets, requests for feedback, or early previews into shipped facts.
- Do not invent business impact, action items, timelines, or requirements unless the speaker states them.
- Do not output orphan headings, broken titles, half-finished bullets, or partial sentences.
- Do not stop after the first few topics; cover the meaningful later-session technical sections as well."""

CHUNK_PROMPT_TEMPLATE = """Below is a transcript segment ({chunk_num}/{total_chunks}) from an Android development course.
Please convert it into structured Markdown notes.

Course title: {video_title}

---
Transcript:

{transcript_chunk}

{slides_context_block}

---

Please generate structured Markdown notes based on the transcript above. Guidelines:
- Treat the transcript as the only evidence source for this chunk; do not add outside knowledge
- This chunk may begin or end mid-thought because of chunking; write only complete notes you can support and skip dangling fragments instead of guessing
- Start directly with a topic-specific heading; no preamble sentence about what the section will cover
- Use topic headings that will merge cleanly with neighboring chunks; if this chunk clearly continues the same topic, reuse a precise heading instead of inventing a recap heading
- Properly format any code snippets with correct language tags
- Use `backtick` for Android API names, class names, and method names
- Organize key concepts using concise bullet points
- Prefer medium-compression notes: capture the important technical content without restating every example, aside comment, or repeated explanation
- Keep only the highest-signal implementation details, requirements, timelines, and platform changes; omit low-value repetition
- Keep timestamps sparingly, only when they materially help the reader locate an important point
- Prefer at most one timestamp for a coherent bullet or subsection; do not attach timestamps to every supporting bullet
- Omit timestamps for routine explanatory details when the surrounding section is already clearly anchored
- Preserve certainty exactly: if something is planned, targeted, previewed, or a request for feedback, label it that way rather than stating it as current fact
- If a claim is plausible but not explicit, either omit it or mark it `(Needs Verification)`
- Ignore non-content cues such as [Music], [Applause], and stage/ambience markers
- If a passage appears to be song lyrics or intermission music, omit it unless it contains technical content
- Exclude logistics/housekeeping content (welcome speech, break times, host intros, NDA reminders) unless technically relevant
- Prioritize high-signal content: what changed, why it matters, and what action is required
- Include "why it matters" or "action required" only when the speaker makes that implication explicit
- Avoid repeating the same takeaway in multiple headings
- Merge adjacent duplicate points into one heading and one bullet list
- Every heading must be followed by at least one complete sentence or bullet
- Do not output boilerplate lines like "This section covers..."
- Do not create standalone sections named "Introduction", "Key Takeaways", "Relevant Slide Snippets", or "Course Summary"
- If slide context is useful, integrate facts inline and cite as `[Slide Page X]`; do not add a separate "Slide Reference" section
{code_policy}"""

ZH_REWRITE_CHUNK_PROMPT_TEMPLATE = """以下是英文版技術筆記片段（{chunk_num}/{total_chunks}）。
請改寫成「繁體中文（台灣）」的結構化技術筆記，維持與英文相同的資訊密度與章節層次。

課程名稱：{video_title}

---
英文筆記片段：

{english_notes_chunk}

---

改寫要求（請嚴格遵守）：
- 不要逐句翻譯，請重組語句與段落，寫成可閱讀的技術筆記
- 句子要自然流暢，避免翻譯腔與生硬直譯
- 來源片段可能從段落中間切開；請整理成完整段落，若資訊殘缺就保守表述，不要補猜
- 使用 `##` / `###` / `####` 標題階層
- 僅保留真正重要的時間戳記（例如重大宣布、版本時程、行動要求、主題切換）
- 同一個小節或條列通常最多保留一個時間戳記，不要每一點都附時間
- 一般補充說明若不影響定位，可省略時間戳記
- 僅在本次允許輸出程式碼，且來源明確出現程式碼時，才使用正確語言標籤（```kotlin / ```java / ```xml）
- Android API、類別、方法名稱保留英文並用 `backtick`
- 專有名詞若難以精準翻譯，請直接保留英文
- 絕對不要輸出簡體中文
- 不要把整份內容包在 ```markdown code fence``` 裡
- 內容要具備技術脈絡，不能只列翻譯句
- 不要在每段開頭重複輸出課程大標題（例如 `# {video_title}`）
- 嚴格保留原文確定性：已可用 / 預覽 / 規劃中 / 徵求合作或回饋；不要把規劃寫成已上線
- 不要新增全域摘要區塊，例如「課程摘要」「本次更新」「影響與價值」
- 合併相鄰重複內容，避免同義標題連續出現
- 不要輸出殘缺標題、半句、孤兒條列
{code_policy}

術語偏好（可直接採用）：
{glossary_text}
"""

SUMMARY_PROMPT_TEMPLATE = """You are writing a fast-read internal brief for engineering colleagues.
They should absorb the course in a few minutes without reading full detailed notes.

Return Markdown using exactly these headings and in exactly this order:
## Course Summary
2-3 concise sentences: overall theme + most important outcomes
## What Changed / Announced
4-8 bullets: only concrete updates from the notes
## Why It Matters
2-5 bullets: ecosystem / product impact
## Key Terms
4-8 bullets: short practical definitions

Hard requirements:
- Prioritize signal over completeness; remove housekeeping/logistics/speaker intros.
- Exclude song lyrics, background music/intermission content, and non-technical chatter.
- Do not repeat the same point across sections.
- Only include claims that are explicit in the notes; do not strengthen tentative language.
- If something is planned, targeted, or previewed, preserve that status instead of writing it as current fact.
- Prefer slightly fuller wording over compressed shorthand when compression would make the meaning ambiguous.
- Each bullet must be understandable on its own; explicitly name the condition, deployment tier, RAM class, API scope, or audience when that changes the meaning.
- If a point applies only to a limited configuration, say so directly instead of implying it applies generally.
- `Why It Matters` may include cautious synthesis, but any inference must be labeled `(Inference)`.
- If evidence is weak, label with `(Needs Verification)` instead of stating as fact.
- Keep wording concise and specific; avoid generic filler.
- Each bullet should be one short sentence whenever possible, but use two short sentences if needed to avoid ambiguity.
- If output budget is tight, shorten bullets and reduce optional detail instead of omitting required sections.
- Never end mid-sentence, mid-bullet, or mid-table.
- Even when evidence is sparse, still output every required heading.
- Return only the requested sections; no title, no preface, no detailed-notes content.

Course title: {video_title}

---
Notes content:

{full_notes}
"""

TRANSCRIPT_SUMMARY_PROMPT_TEMPLATE = """You are writing a fast-read internal brief for engineering colleagues.
They should absorb the course in a few minutes without reading the full transcript.

Return Markdown using exactly these headings and in exactly this order:
## Course Summary
2-3 concise sentences: overall theme + most important outcomes
## What Changed / Announced
4-8 bullets: only concrete updates from the transcript
## Why It Matters
2-5 bullets: ecosystem / product impact
## Key Terms
4-8 bullets: short practical definitions

Hard requirements:
- Use the transcript as the primary source of truth; do not add outside knowledge.
- Cover the full session, not just the opening topics.
- If important later-session topics exist, reflect them somewhere in the summary.
- Prioritize signal over completeness; remove housekeeping/logistics/speaker intros.
- Exclude song lyrics, background music/intermission content, and non-technical chatter.
- Do not repeat the same point across sections.
- Only include claims that are explicit in the transcript; do not strengthen tentative language.
- If something is planned, targeted, or previewed, preserve that status instead of writing it as current fact.
- Prefer slightly fuller wording over compressed shorthand when compression would make the meaning ambiguous.
- Each bullet must be understandable on its own; explicitly name the condition, deployment tier, RAM class, API scope, or audience when that changes the meaning.
- If a point applies only to a limited configuration, say so directly instead of implying it applies generally.
- `Why It Matters` may include cautious synthesis, but any inference must be labeled `(Inference)`.
- If evidence is weak, label with `(Needs Verification)` instead of stating as fact.
- Keep wording concise and specific; avoid generic filler.
- Each bullet should be one short sentence whenever possible, but use two short sentences if needed to avoid ambiguity.
- If output budget is tight, shorten bullets and reduce optional detail instead of omitting required sections.
- Never end mid-sentence, mid-bullet, or mid-table.
- Even when evidence is sparse, still output every required heading.
- Return only the requested sections; no title, no preface, no detailed-notes content.

Course title: {video_title}

---
Transcript content:

{full_transcript}
"""

ZH_SUMMARY_FROM_ENGLISH_PROMPT_TEMPLATE = """以下是英文版 Android 課程摘要 Markdown。請改寫成「繁體中文（台灣）」Markdown 摘要。

請使用以下固定中文標題，並依照英文原文的區塊順序與重點對應：
## 課程摘要
## 本次更新／新宣布事項
## 影響與價值
## 關鍵術語

硬性要求：
- 這是根據英文摘要改寫，不是重新摘要；不要新增英文沒有的重點、時程、條件、行動項或推論。
- 儘量保持各區塊的 bullet 數量、資訊密度與重點順序和英文一致。
- 保留英文原文的確定性與限制條件；不要把 preview / planned / targeted / optional 寫成既定事實。
- 若英文條列有前提條件（例如裝置等級、RAM 門檻、system API 限制、僅適用特定對象），請在中文中明確寫出，不要省略。
- 不要混入新的全域摘要、評論或補充背景。
- 不要輸出簡體中文。
- 不要把整份內容包在 code fence 裡。

課程名稱：{video_title}

---
英文摘要：

{english_summary}
"""

EN_CLEANUP_SYSTEM_PROMPT = """You are a conservative technical editor cleaning up already-generated Android course notes.
Your job is cleanup only. Do not add facts, examples, code, or interpretations."""

EN_DETAILED_CLEANUP_PROMPT_TEMPLATE = """Below is a chunk of already-generated English Markdown notes for an Android bootcamp session.
Clean it up conservatively.

Course title: {video_title}

---
Notes chunk:

{notes_chunk}

---

Hard rules:
- Cleanup only; do not add any new facts, examples, explanations, section themes, timelines, or action items.
- Do not infer missing content.
- Do not output any code block, pseudocode, XML snippet, placeholder example, or synthetic sample.
- Remove placeholder text such as "No content...", "No technical content...", "No notes can be generated", timestamp-only explanations, empty-content notices, and song/intermission omission notices.
- Remove redundant timestamps when multiple nearby bullets or sentences point to the same moment; keep only the most useful timestamp.
- Remove headings that have no meaningful content beneath them.
- Remove incomplete bullets, dangling sentences, broken fragments, and duplicated separator blocks.
- Merge adjacent duplicate or near-duplicate headings only when the underlying content clearly overlaps.
- Preserve valid timestamps, factual certainty, technical terminology, and useful structure.
- Keep the output in English Markdown only.
- If nothing useful remains, return an empty string.
"""

EN_GLOBAL_NORMALIZE_PROMPT_TEMPLATE = """Below are cleaned English detailed notes for an Android bootcamp session.
Perform one final conservative normalization pass.

Course title: {video_title}

---
Detailed notes:

{full_notes}

---

Hard rules:
- Do not add any new facts or infer missing information.
- Do not output any code block, placeholder text, empty-content notices, or synthetic examples.
- Remove excessive timestamps; keep only timestamps that materially improve navigation.
- Remove any remaining empty sections, duplicate headings, dangling fragments, and repeated separators.
- Merge clearly duplicated adjacent sections while preserving chronology and technical meaning.
- Preserve timestamps, certainty, and terminology.
- Return Markdown only.
"""

ONE_SHOT_DETAILED_PROMPT_TEMPLATE = """Below is the full transcript for one complete Android bootcamp session.
This is the entire session transcript, not a chunk.

Course title: {video_title}

---
Full transcript:

{full_transcript}

---

Generate comprehensive English Markdown detailed notes for the full session.

Hard rules:
- Cover the full session from beginning to end; do not stop after the early topics.
- Organize notes in chronological order by major topic shifts.
- Output detailed notes only. Do not include `Course Summary`, `What Changed / Announced`, `Why It Matters`, `Action Items`, `Timeline & Version Signals`, `Key Terms`, or `Q&A` sections here.
- Start directly with topic-specific `##` headings.
- Use `###` and `####` only when they improve readability within a topic.
- Prefer concise but complete bullets and short paragraphs.
- Ignore housekeeping, welcomes, breaks, NDA reminders, applause, music, lyrics, and empty timestamp-only stretches.
- Preserve factual certainty exactly as stated: available, preview, planned, targeted, request for feedback, or needs verification.
- Do not add outside knowledge, synthetic examples, or inferred requirements.
- Keep timestamps sparingly, only when they materially help locate an important announcement, requirement, transition, or claim.
- Make sure later-session technical topics are included if they contain meaningful content.
- Do not output any code block, pseudocode, XML snippet, or synthetic sample code.
- Remove filler and repeated phrasing.
- Do not output placeholder text such as "No content..." or explain omitted empty stretches.

Coverage requirement:
- Aim to capture every major technical section of the session with enough detail that an engineer could skim the notes and understand the full arc of the talk.

{code_policy}
"""

ZH_SYSTEM_PROMPT = """你是資深技術編輯。請使用「繁體中文（台灣）」撰寫，並使用台灣慣用技術用詞與語氣（例如：影片、逐字稿、章節、重點整理、程式碼、執行、設定、效能、記憶體）。

請避免中國大陸常見用語（例如：视频、脚本、内存、运行、配置、优化），也避免中英混雜與口語贅字。

目標讀者是台灣工程同仁：內容要精準、可執行，保留技術脈絡，不要過度壓縮。"""

ZH_SUMMARY_PROMPT_TEMPLATE = """以下是 Android 課程完整筆記。請產出「繁體中文（台灣）Markdown 快速摘要」，給工程同仁在幾分鐘內掌握重點。

請只輸出以下區塊，標題文字與順序都固定：
## 課程摘要
1-2 句短句：主題與最重要成果
## 本次更新／新宣布事項
4-8 點，僅列明確資訊
## 影響與價值
2-5 點，說明為何重要
## 關鍵術語
4-8 點，簡短且實用的定義

硬性要求：
- 以高訊號為主，不要寫主持開場、休息時間、流程提醒、保密提醒。
- 排除歌詞、背景音樂、過場閒聊等非技術內容。
- 各區塊不要重複同一重點。
- 僅能寫筆記中明確出現的資訊；不要把講者的目標、預覽、規劃，寫成已經上線或既定事實。
- 「影響與價值」可做保守歸納；若屬推論請標註 `(推論)`。
- 證據不足時請標註 `(待確認)`，不要寫成既定事實。
- 句子要短、具體、可執行，避免空泛敘述。
- 每個條列盡量只寫一句短句。
- 如果輸出預算吃緊，優先縮短條列與減少次要細節，不要漏掉必要區塊。
- 不可輸出斷句、半個條列或半張表格。
- 即使資訊不足，也必須保留所有必要標題。
- 不要輸出課程標題、前言、結語，也不要混入詳細筆記內容。

課程名稱：{video_title}

---
筆記內容：

{full_notes}
"""

ZH_QA_SECTION = """7. **延伸問答**：產出最多 3 題高價值問題與簡短答案（每題 1-3 句）"""

QA_SECTION = """7. **Q&A**: Generate at most 3 high-value questions and short answers for internal review"""

ZH_REVIEW_PROMPT_TEMPLATE = """請修正以下繁中筆記段落，讓品質符合台灣工程團隊閱讀需求。

修正規則：
- 僅使用繁體中文（台灣），禁止簡體字
- 保留技術內容與章節層次，不可縮短成摘要
- 不要逐句翻譯口吻，要改寫成自然技術筆記
- 專有名詞若難以翻譯，直接保留英文
- 不要輸出包住全文的 ```markdown code fence```
- 保留原有程式碼區塊與時間戳記
- 保留原文的確定性，不要把規劃/預覽寫成已上線
- 修掉重複標題、殘缺標題、半句與孤兒條列

原文：
{bad_chunk}
"""

SIMPLIFIED_HINT_WORDS = [
    "欢迎", "视频", "脚本", "内存", "运行", "配置", "优化", "阶段",
    "系统服务", "发布", "实时", "语音", "翻译", "应用程序", "数据",
]

CODE_EVIDENCE_PATTERNS = [
    r"```",
    r"\b(class|interface|fun|suspend|val|var|public|private|override|import)\b",
    r"\b(onCreate|setContent|@Composable|ViewModel|LiveData|Room|Retrofit)\b",
    r"</?[A-Za-z][A-Za-z0-9:_-]*>",
    r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(",
    r"=\s*new\s+[A-Za-z_]",
    r"\{[^}]{0,120}\}",
]

SLIDE_RETRIEVAL_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "you", "your",
    "have", "has", "was", "were", "will", "can", "not", "but", "all", "any",
    "into", "about", "they", "their", "there", "then", "than", "just", "also",
    "what", "when", "where", "which", "while", "how", "why", "use", "using",
    "android", "bootcamp", "2026", "session", "track", "team", "today", "next",
    "our", "its", "it's", "we", "to", "of", "in", "on", "at", "by", "or", "as",
}

SLIDE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]{1,}")
SLIDE_NOISE_LINE_PATTERNS = [
    re.compile(r"proprietary and confidential", re.IGNORECASE),
    re.compile(r"do not distribute", re.IGNORECASE),
]

EN_REQUIRED_SUMMARY_HEADINGS = [
    "## Course Summary",
    "## What Changed / Announced",
    "## Why It Matters",
    "## Key Terms",
]


def _merge_continuation_text(existing: str, new_text: str) -> str:
    if not existing:
        return new_text
    if not new_text:
        return existing

    max_overlap = min(len(existing), len(new_text), 200)
    for size in range(max_overlap, 24, -1):
        if existing[-size:] == new_text[:size]:
            return existing + new_text[size:]

    return existing + new_text


def _split_markdown_sections(markdown_text: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []

    for line in markdown_text.splitlines():
        if re.match(r"^#{2,4}\s+\S", line) and current:
            section = "\n".join(current).strip()
            if section:
                sections.append(section)
            current = [line]
            continue
        current.append(line)

    if current:
        section = "\n".join(current).strip()
        if section:
            sections.append(section)

    return sections


def _split_oversized_markdown_section(section_text: str, max_chars: int, overlap: int) -> list[str]:
    if len(section_text) <= max_chars:
        return [section_text.strip()]

    lines = section_text.splitlines()
    heading = lines[0].strip() if lines and re.match(r"^#{2,4}\s+\S", lines[0]) else ""
    body = "\n".join(lines[1:]).strip() if heading else section_text.strip()

    if not body:
        return [section_text.strip()]

    body_budget = max_chars
    if heading:
        body_budget = max(600, max_chars - len(heading) - 2)

    body_chunks = chunk_text(body, body_budget, overlap)
    parts: list[str] = []
    for chunk in body_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if heading:
            parts.append(f"{heading}\n\n{chunk}".strip())
        else:
            parts.append(chunk)

    return parts or [section_text.strip()]


def chunk_markdown_by_sections(markdown_text: str, chunk_size: int, overlap: int) -> list[str]:
    sections = _split_markdown_sections(markdown_text.strip())
    if not sections:
        return chunk_text(markdown_text, chunk_size, overlap)

    batches: list[str] = []
    current_sections: list[str] = []
    current_len = 0

    for section in sections:
        if len(section) > chunk_size:
            if current_sections:
                batches.append("\n\n".join(current_sections).strip())
                current_sections = []
                current_len = 0
            batches.extend(_split_oversized_markdown_section(section, chunk_size, overlap))
            continue

        extra_len = len(section) if not current_sections else len(section) + len("\n\n")
        if current_sections and current_len + extra_len > chunk_size:
            batches.append("\n\n".join(current_sections).strip())
            current_sections = [section]
            current_len = len(section)
            continue

        current_sections.append(section)
        current_len += extra_len

    if current_sections:
        batches.append("\n\n".join(current_sections).strip())

    return batches

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
    max_continuations = int(llm_cfg.get("max_continuations", 4))

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
            base_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            content = ""
            continuation_count = 0

            while True:
                if continuation_count == 0:
                    messages = base_messages
                else:
                    messages = [
                        *base_messages,
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "Continue exactly where you stopped. Do not repeat prior text, do not restart the section, "
                                "and end only at a complete Markdown boundary."
                            ),
                        },
                    ]

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
                choice = response.choices[0]
                chunk = choice.message.content or ""
                finish_reason = choice.finish_reason or "stop"
                content = _merge_continuation_text(content, chunk)

                if finish_reason != "length":
                    break

                continuation_count += 1
                if continuation_count > max_continuations:
                    raise RuntimeError(
                        f"LLM output was still truncated after {max_continuations} continuation attempts."
                    )

            if progress_label:
                elapsed = format_timestamp(time.time() - request_start)
                print(f"\r    {progress_label} ✓ 完成，耗時 {elapsed}{' ' * 20}")

            return content.strip()

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


def _normalize_zh_chunk(text: str, video_title: str) -> str:
    cleaned = _unwrap_outer_markdown_fence(text).strip()
    lines = cleaned.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith("#") and video_title in lines[0]:
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


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


def _chunk_has_code_evidence(text: str) -> bool:
    for pattern in CODE_EVIDENCE_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def _build_code_policy(has_code_evidence: bool, allow_code_blocks: bool, language: str = "en") -> str:
    if language == "en":
        if not allow_code_blocks:
            return "- Do not output any code block, pseudocode, XML snippet, or inline sample code for this segment. Summarize implementation details in prose only."
        if has_code_evidence:
            return "- Include code blocks only when code is explicitly present in transcript or slides."
        return "- Do not output any code block for this segment."

    if not allow_code_blocks:
        return "- 本次禁止輸出任何 code block、偽程式碼、XML 片段或行內範例程式碼；請一律改用文字描述實作重點。"
    if has_code_evidence:
        return "- 僅在來源內容明確出現程式碼時，才保留或改寫為 code block。"
    return "- 這一段禁止輸出 code block，請改用文字說明。"


def _extract_duration_from_markdown(markdown_text: str, fallback: str = "00:00") -> str:
    match = re.search(r"\*\*Duration\*\*:\s*([0-9]{2}:[0-9]{2}(?::[0-9]{2})?)", markdown_text)
    if match:
        return match.group(1)
    return fallback


def _split_english_markdown_sections(english_markdown: str) -> tuple[str, str]:
    summary_text = ""
    detailed_text = english_markdown.strip()
    detailed_marker = "\n## Detailed Notes\n"

    if detailed_marker not in english_markdown:
        return summary_text, detailed_text

    prefix, detailed = english_markdown.split(detailed_marker, 1)
    detailed_text = detailed.strip()

    sections = prefix.split("\n\n---\n\n", 1)
    if len(sections) == 2:
        summary_text = sections[1].strip()

    return summary_text, detailed_text


def _has_required_headings(markdown_text: str, headings: list[str]) -> bool:
    stripped = markdown_text.strip()
    if not stripped:
        return False
    return all(heading in stripped for heading in headings)


def _generate_validated_english_summary(
    client: OpenAI,
    model: str,
    system_prompt: str,
    primary_prompt: str,
    config: dict,
    progress_label: str,
    fallback_prompt: str | None = None,
) -> str:
    prompts = [
        (primary_prompt, progress_label),
        (
            primary_prompt
            + "\n\nReminder: output every required heading, and do not return an empty response.",
            f"{progress_label} 重試",
        ),
    ]

    for prompt, label in prompts:
        summary = call_llm(
            client,
            model,
            system_prompt,
            prompt,
            config,
            progress_label=label,
        )
        summary = _unwrap_outer_markdown_fence(summary).strip()
        if _has_required_headings(summary, EN_REQUIRED_SUMMARY_HEADINGS):
            return summary

    if fallback_prompt:
        print("\n  ⚠ 摘要輸出不完整，改用英文筆記備援摘要...")
        summary = call_llm(
            client,
            model,
            system_prompt,
            fallback_prompt,
            config,
            progress_label=f"{progress_label} 備援",
        )
        summary = _unwrap_outer_markdown_fence(summary).strip()
        if _has_required_headings(summary, EN_REQUIRED_SUMMARY_HEADINGS):
            return summary

    raise RuntimeError("English summary generation returned empty or incomplete output.")


def _group_note_batches(notes_text: str, max_chars: int) -> list[str]:
    parts = [part.strip() for part in notes_text.split("\n\n---\n\n") if part.strip()]
    if not parts:
        return []

    batches: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for part in parts:
        extra_len = len(part) if not current_parts else len(part) + len("\n\n---\n\n")
        if current_parts and current_len + extra_len > max_chars:
            batches.append("\n\n---\n\n".join(current_parts))
            current_parts = [part]
            current_len = len(part)
            continue

        current_parts.append(part)
        current_len += extra_len

    if current_parts:
        batches.append("\n\n---\n\n".join(current_parts))

    return batches


def cleanup_english_detailed_notes(
    combined_notes: str,
    video_title: str,
    config: dict,
    client: OpenAI,
    model: str,
) -> str:
    cleaned_source = combined_notes.strip()
    if not cleaned_source:
        return cleaned_source

    llm_cfg = config.get("llm", {})
    request_delay = llm_cfg.get("request_delay", 1.0)
    cleanup_batch_chars = 9000
    global_cleanup_limit = 18000

    cleanup_batches = _group_note_batches(cleaned_source, cleanup_batch_chars)
    if not cleanup_batches:
        return cleaned_source

    print("\n  🧹 清理英文詳細筆記...")
    cleaned_batches: list[str] = []

    for i, batch in enumerate(cleanup_batches, 1):
        cleanup_prompt = EN_DETAILED_CLEANUP_PROMPT_TEMPLATE.format(
            video_title=video_title,
            notes_chunk=batch,
        )
        cleaned_batch = call_llm(
            client,
            model,
            EN_CLEANUP_SYSTEM_PROMPT,
            cleanup_prompt,
            config,
            progress_label=f"英文清理 {i}/{len(cleanup_batches)}",
            temperature_override=0.1,
        )
        cleaned_batch = _unwrap_outer_markdown_fence(cleaned_batch).strip()
        if cleaned_batch:
            cleaned_batches.append(cleaned_batch)

        if i < len(cleanup_batches):
            time.sleep(request_delay)

    if not cleaned_batches:
        return cleaned_source

    cleaned_notes = "\n\n---\n\n".join(cleaned_batches).strip()
    if len(cleaned_notes) > global_cleanup_limit:
        return cleaned_notes

    print("\n  🧭 整理英文詳細筆記整體結構...")
    normalize_prompt = EN_GLOBAL_NORMALIZE_PROMPT_TEMPLATE.format(
        video_title=video_title,
        full_notes=cleaned_notes,
    )
    normalized_notes = call_llm(
        client,
        model,
        EN_CLEANUP_SYSTEM_PROMPT,
        normalize_prompt,
        config,
        progress_label="英文整體整理",
        temperature_override=0.1,
    )
    normalized_notes = _unwrap_outer_markdown_fence(normalized_notes).strip()
    return normalized_notes or cleaned_notes


def _get_oneshot_settings(config: dict, default_model: str) -> dict:
    llm_cfg = config.get("llm", {})
    oneshot_cfg = config.get("llm_oneshot", {})
    return {
        "model": oneshot_cfg.get("model", default_model),
        "max_tokens": oneshot_cfg.get("max_tokens", max(llm_cfg.get("max_tokens", 4096), 12288)),
        "temperature": oneshot_cfg.get("temperature", 0.2),
        "max_retries": oneshot_cfg.get("max_retries", llm_cfg.get("max_retries", 3)),
    }


def build_english_notes_from_transcript_oneshot(
    transcript_json_path: Path,
    config: dict,
    client: OpenAI,
) -> tuple[str, str, str, str]:
    """Process one transcript in a dedicated one-shot flow."""
    llm_cfg = config.get("llm", {})
    notes_cfg = config.get("notes", {})
    default_model = llm_cfg.get("model", "qwen2.5:7b-instruct-q4_K_M")
    settings = _get_oneshot_settings(config, default_model)
    request_delay = llm_cfg.get("request_delay", 1.0)
    include_timestamps = notes_cfg.get("include_timestamps", True)
    generate_summary = notes_cfg.get("generate_summary", True)
    generate_qa = notes_cfg.get("generate_qa", True)
    use_slides_context = notes_cfg.get("use_slides_context", True)
    detect_code = notes_cfg.get("detect_code", True)

    video_title = transcript_json_path.stem

    with open(transcript_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    slide_blocks = []
    if use_slides_context:
        slides_context = _load_slides_context(video_title, config)
        if slides_context:
            print(f"  ✓ 載入 slides 內容（{len(slides_context)} 字）")
            slide_blocks = _build_slide_blocks(slides_context, config)
            print(f"  ✓ slides 已建立檢索索引（{len(slide_blocks)} 段）")
        else:
            print("  ℹ 未找到可用 slides，改為僅使用逐字稿")
    else:
        print("  ⏭ 已關閉 slides 參考（notes.use_slides_context = false）")

    if include_timestamps:
        full_text = "\n".join(
            f"[{format_timestamp(seg['start'])}] {seg['text']}"
            for seg in segments
        )
    else:
        full_text = "\n".join(seg["text"] for seg in segments)

    print(f"  逐字稿共 {len(full_text)} 字，採 one-shot 單次生成")
    code_policy = _build_code_policy(
        _chunk_has_code_evidence(full_text),
        allow_code_blocks=detect_code,
        language="en",
    )
    slides_context_block = _build_slides_context_block_for_chunk(full_text, slide_blocks, config)
    oneshot_prompt = ONE_SHOT_DETAILED_PROMPT_TEMPLATE.format(
        video_title=video_title,
        full_transcript=full_text,
        code_policy=code_policy,
    )
    if slides_context_block:
        oneshot_prompt = f"{oneshot_prompt.rstrip()}\n\n{slides_context_block}\n"

    detailed_notes = call_llm(
        client,
        settings["model"],
        SYSTEM_PROMPT_ONE_SHOT,
        oneshot_prompt,
        config,
        progress_label="One-shot 詳細筆記",
        max_tokens_override=settings["max_tokens"],
        temperature_override=settings["temperature"],
        max_retries_override=settings["max_retries"],
    )
    detailed_notes = _unwrap_outer_markdown_fence(detailed_notes).strip()
    detailed_notes = cleanup_english_detailed_notes(
        detailed_notes,
        video_title,
        config,
        client,
        settings["model"],
    )

    total_duration = format_timestamp(segments[-1]["end"]) if segments else "00:00"
    summary_section = ""
    if generate_summary:
        print("\n  📋 生成課程摘要...")
        qa_part = QA_SECTION if generate_qa else ""
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            video_title=video_title,
            full_notes=detailed_notes,
            qa_section=qa_part,
        )
        summary_section = _generate_validated_english_summary(
            client,
            settings["model"],
            SYSTEM_PROMPT_ONE_SHOT,
            summary_prompt,
            config,
            progress_label="課程摘要",
        )

    if request_delay:
        time.sleep(request_delay)

    final_md = f"""# {video_title}

> **Duration**: {total_duration}  
> **Generated by**: {settings["model"]}  

---

{summary_section}

---

## Detailed Notes

{detailed_notes}
"""

    return final_md, detailed_notes, total_duration, settings["model"]


def _normalize_slide_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            lines.append("")
            continue
        if any(pat.search(line) for pat in SLIDE_NOISE_LINE_PATTERNS):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tokenize_slide_text(text: str) -> set[str]:
    tokens = []
    for token in SLIDE_TOKEN_RE.findall(text.lower()):
        if len(token) < 2:
            continue
        if token in SLIDE_RETRIEVAL_STOPWORDS:
            continue
        tokens.append(token)
    return set(tokens)


def _build_slide_blocks(slides_text: str, config: dict) -> list[dict]:
    llm_cfg = config.get("llm", {})
    block_chars = int(llm_cfg.get("slides_block_chars", 1400))
    block_overlap = int(llm_cfg.get("slides_block_overlap", 120))
    min_block_chars = int(llm_cfg.get("slides_min_block_chars", 140))
    pages = [p for p in slides_text.split("\f") if p.strip()]

    blocks: list[dict] = []

    def add_block(label: str, text: str, page: int | None, part: int | None) -> None:
        normalized = _normalize_slide_text(text)
        if not normalized:
            return
        if len(normalized) < min_block_chars:
            return
        tokens = _tokenize_slide_text(normalized)
        if not tokens:
            return
        blocks.append({
            "label": label,
            "text": normalized,
            "tokens": tokens,
            "page": page,
            "part": part,
        })

    if pages:
        for page_idx, page_text in enumerate(pages, 1):
            normalized = _normalize_slide_text(page_text)
            if not normalized:
                continue
            if len(normalized) <= block_chars:
                add_block(f"Slide Page {page_idx}", normalized, page=page_idx, part=1)
            else:
                subchunks = chunk_text(normalized, block_chars, block_overlap)
                for sub_idx, sub in enumerate(subchunks, 1):
                    add_block(f"Slide Page {page_idx}.{sub_idx}", sub, page=page_idx, part=sub_idx)
    else:
        if len(slides_text) <= block_chars:
            add_block("Slide Segment 1", slides_text, page=None, part=1)
        else:
            subchunks = chunk_text(slides_text, block_chars, block_overlap)
            for idx, sub in enumerate(subchunks, 1):
                add_block(f"Slide Segment {idx}", sub, page=None, part=idx)

    return blocks


def _score_slide_block(query_tokens: set[str], block_tokens: set[str], block_text: str) -> tuple[int, int]:
    overlap = query_tokens & block_tokens
    if not overlap:
        return 0, 0
    overlap_count = len(overlap)
    lexical = sum(min(len(token), 12) for token in overlap)
    length_bonus = min(len(block_text) // 180, 6) * 2
    score = lexical + overlap_count * 4 + length_bonus
    return score, overlap_count


def _expand_slide_snippet(slide_blocks: list[dict], center_idx: int, min_chars: int, max_chars: int) -> str:
    if not slide_blocks:
        return ""
    center = slide_blocks[center_idx]
    page = center.get("page")
    center_text = center["text"].strip()
    if not center_text:
        return ""

    selected_indices = [center_idx]
    selected_index_set = {center_idx}
    total = len(center_text)

    def try_add(idx: int, prepend: bool, require_same_page: bool) -> bool:
        nonlocal total
        if idx < 0 or idx >= len(slide_blocks):
            return False
        if idx in selected_index_set:
            return False

        candidate = slide_blocks[idx]
        if require_same_page and page is not None and candidate.get("page") != page:
            return False

        segment = candidate["text"].strip()
        if not segment:
            return False
        if total + len(segment) > max_chars:
            return False

        if prepend:
            selected_indices.insert(0, idx)
        else:
            selected_indices.append(idx)
        selected_index_set.add(idx)
        total += len(segment)
        return True

    # Pass 1: expand around the center within the same page first.
    step = 1
    while total < min_chars and (center_idx - step >= 0 or center_idx + step < len(slide_blocks)):
        right_idx = center_idx + step
        left_idx = center_idx - step
        took = False
        took = try_add(right_idx, prepend=False, require_same_page=True) or took
        if total < min_chars:
            took = try_add(left_idx, prepend=True, require_same_page=True) or took
        step += 1
        if not took and left_idx < 0 and right_idx >= len(slide_blocks):
            break

    # Pass 2: if same-page content is still too short, expand to nearby pages.
    if total < min_chars:
        step = 1
        while total < min_chars and (center_idx - step >= 0 or center_idx + step < len(slide_blocks)):
            right_idx = center_idx + step
            left_idx = center_idx - step
            took = False
            took = try_add(right_idx, prepend=False, require_same_page=False) or took
            if total < min_chars:
                took = try_add(left_idx, prepend=True, require_same_page=False) or took
            step += 1
            if not took and left_idx < 0 and right_idx >= len(slide_blocks):
                break

    collected = [
        slide_blocks[idx]["text"].strip()
        for idx in selected_indices
        if slide_blocks[idx]["text"].strip()
    ]
    text = "\n\n".join(collected)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def _build_slides_context_block_for_chunk(chunk_text_input: str, slide_blocks: list[dict], config: dict) -> str:
    if not slide_blocks:
        return ""

    llm_cfg = config.get("llm", {})
    top_k = int(llm_cfg.get("slides_top_k_per_chunk", 3))
    per_snippet_chars = int(llm_cfg.get("slides_snippet_max_chars", 900))
    min_snippet_chars = int(llm_cfg.get("slides_min_snippet_chars", 450))
    min_accepted_snippet_chars = int(
        llm_cfg.get("slides_min_accepted_snippet_chars", max(220, min_snippet_chars // 2))
    )
    total_chars_cap = int(llm_cfg.get("slides_context_chars_per_chunk", 2600))
    min_score = int(llm_cfg.get("slides_retrieval_min_score", 10))
    min_overlap_terms = int(llm_cfg.get("slides_min_overlap_terms", 2))
    prefer_page_diversity = bool(llm_cfg.get("slides_prefer_page_diversity", False))

    query_tokens = _tokenize_slide_text(chunk_text_input)
    if not query_tokens:
        return ""

    scored = []
    for idx, block in enumerate(slide_blocks):
        score, overlap_count = _score_slide_block(query_tokens, block["tokens"], block["text"])
        if score > 0 and overlap_count >= min_overlap_terms:
            scored.append((score, overlap_count, idx, block))

    if not scored:
        return ""

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if scored[0][0] < min_score:
        return ""

    candidate_pool = scored[: max(top_k * 8, 12)]
    candidates = []
    for score, overlap_count, idx, block in candidate_pool:
        snippet = _expand_slide_snippet(
            slide_blocks,
            center_idx=idx,
            min_chars=min_snippet_chars,
            max_chars=per_snippet_chars,
        )
        if not snippet:
            continue
        snippet_len = len(snippet.strip())
        if snippet_len < min_accepted_snippet_chars:
            continue
        candidates.append((score, overlap_count, snippet_len, idx, block, snippet))

    # Fallback: avoid returning empty context when slides are globally sparse.
    if not candidates:
        fallback_pool = scored[: max(top_k * 3, 6)]
        for score, overlap_count, idx, block in fallback_pool:
            snippet = _expand_slide_snippet(
                slide_blocks,
                center_idx=idx,
                min_chars=min_snippet_chars,
                max_chars=per_snippet_chars,
            )
            if not snippet:
                continue
            snippet_len = len(snippet.strip())
            if snippet_len < 120:
                continue
            candidates.append((score, overlap_count, snippet_len, idx, block, snippet))
            if len(candidates) >= top_k:
                break

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

    selected_blocks = []
    used_chars = 0
    used_pages: set[int | None] = set()
    used_indices: set[int] = set()
    for score, overlap_count, snippet_len, idx, block, snippet in candidates:
        if idx in used_indices:
            continue
        page = block.get("page")
        if prefer_page_diversity and page in used_pages and len(selected_blocks) < top_k - 1:
            continue
        formatted = (
            f"[{block['label']} | relevance {score} | overlap {overlap_count}]\n"
            f"{snippet}"
        )
        if used_chars + len(formatted) > total_chars_cap and selected_blocks:
            continue
        selected_blocks.append(formatted)
        used_chars += len(formatted)
        used_pages.add(page)
        used_indices.add(idx)
        if len(selected_blocks) >= top_k:
            break

    if not selected_blocks:
        return ""

    return (
        "Relevant slide snippets for this transcript chunk "
        "(already retrieved by keyword overlap, use only when relevant):\n\n"
        + "\n\n---\n\n".join(selected_blocks)
        + "\n"
    )


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
        "detect_code": notes_cfg.get("detect_code", True),
        "glossary_text": _build_glossary_text(notes_cfg),
    }


def _load_slides_context(video_title: str, config: dict) -> str:
    paths_cfg = config.get("paths", {})
    slides_dir_name = paths_cfg.get("slides_dir", "slides")
    root = get_project_root()
    slides_dir = root / slides_dir_name

    if not slides_dir.exists():
        return ""

    txt_path = slides_dir / f"{video_title}.txt"
    md_path = slides_dir / f"{video_title}.md"
    pdf_path = slides_dir / f"{video_title}.pdf"

    raw = ""
    if txt_path.exists():
        raw = txt_path.read_text(encoding="utf-8", errors="replace")
    elif md_path.exists():
        raw = md_path.read_text(encoding="utf-8", errors="replace")
    elif pdf_path.exists() and shutil.which("pdftotext"):
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
            )
            raw = result.stdout
        except Exception:
            raw = ""

    raw = raw.strip()
    if not raw:
        return ""

    return raw


def generate_zh_notes_from_english_markdown(
    english_markdown: str,
    video_title: str,
    total_duration: str,
    config: dict,
    client: OpenAI,
    default_model: str,
) -> str:
    settings = _get_zh_settings(config, default_model)
    zh_model = settings["model"]
    english_summary, english_notes = _split_english_markdown_sections(english_markdown)
    zh_source_chunks = chunk_markdown_by_sections(
        english_notes,
        settings["chunk_size"],
        settings["chunk_overlap"],
    )
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
            code_policy=_build_code_policy(
                _chunk_has_code_evidence(en_chunk),
                allow_code_blocks=settings["detect_code"],
                language="zh",
            ),
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
        zh_note = _normalize_zh_chunk(zh_note, video_title)
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
            zh_note = _normalize_zh_chunk(zh_note, video_title)

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
    if english_summary:
        zh_prompt = ZH_SUMMARY_FROM_ENGLISH_PROMPT_TEMPLATE.format(
            video_title=video_title,
            english_summary=english_summary,
        )
    else:
        zh_qa_part = ZH_QA_SECTION if settings["generate_qa"] else ""
        zh_prompt = ZH_SUMMARY_PROMPT_TEMPLATE.format(
            video_title=video_title,
            full_notes=combined_notes_zh,
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


def build_english_notes_from_transcript(
    transcript_json_path: Path,
    config: dict,
    client: OpenAI,
) -> tuple[str, str, str, str]:
    """處理一份逐字稿，生成英文 Markdown 筆記。"""
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
    use_slides_context = notes_cfg.get("use_slides_context", True)
    detect_code = notes_cfg.get("detect_code", True)

    video_title = transcript_json_path.stem
    system_prompt = get_system_prompt(language)

    with open(transcript_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    slide_blocks = []
    if use_slides_context:
        slides_context = _load_slides_context(video_title, config)
        if slides_context:
            print(f"  ✓ 載入 slides 內容（{len(slides_context)} 字）")
            slide_blocks = _build_slide_blocks(slides_context, config)
            print(f"  ✓ slides 已建立檢索索引（{len(slide_blocks)} 段）")
        else:
            print("  ℹ 未找到可用 slides，改為僅使用逐字稿")
    else:
        print("  ⏭ 已關閉 slides 參考（notes.use_slides_context = false）")

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
        code_policy = _build_code_policy(
            _chunk_has_code_evidence(chunk),
            allow_code_blocks=detect_code,
            language="en",
        )
        slides_context_block = _build_slides_context_block_for_chunk(chunk, slide_blocks, config)

        user_prompt = CHUNK_PROMPT_TEMPLATE.format(
            chunk_num=i,
            total_chunks=total_chunks,
            video_title=video_title,
            transcript_chunk=chunk,
            slides_context_block=slides_context_block,
            code_policy=code_policy,
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
    combined_notes = cleanup_english_detailed_notes(
        combined_notes,
        video_title,
        config,
        client,
        model,
    )
    total_duration = format_timestamp(segments[-1]["end"]) if segments else "00:00"

    summary_section = ""
    if generate_summary:
        print("\n  📋 生成課程摘要...")

        qa_part = QA_SECTION if generate_qa else ""
        summary_prompt = TRANSCRIPT_SUMMARY_PROMPT_TEMPLATE.format(
            video_title=video_title,
            full_transcript=full_text,
            qa_section=qa_part,
        )
        fallback_summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            video_title=video_title,
            full_notes=combined_notes,
            qa_section=qa_part,
        )

        summary_section = _generate_validated_english_summary(
            client,
            model,
            SYSTEM_PROMPT_ONE_SHOT,
            summary_prompt,
            config,
            progress_label="課程摘要",
            fallback_prompt=fallback_summary_prompt,
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

    return final_md, combined_notes, total_duration, model


def process_transcript(transcript_json_path: Path, config: dict,
                       client: OpenAI) -> tuple[str, str]:
    """處理一份逐字稿，生成英文與中文 Markdown 筆記。"""
    notes_cfg = config.get("notes", {})
    generate_zh_notes = notes_cfg.get("generate_zh_summary", True)
    video_title = transcript_json_path.stem

    final_md, _combined_notes, total_duration, model = build_english_notes_from_transcript(
        transcript_json_path,
        config,
        client,
    )

    zh_summary_md = ""
    if generate_zh_notes:
        zh_summary_md = generate_zh_notes_from_english_markdown(
            final_md,
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
    notes_cfg = config.get("notes", {})
    generate_zh_notes = notes_cfg.get("generate_zh_summary", True)

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
            if not generate_zh_notes:
                print(f"⏭ 跳過（英文已存在，且已關閉中文）：{stem}")
                success_count += 1
                continue

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

                markdown_zh = generate_zh_notes_from_english_markdown(
                    english_markdown,
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
            markdown_en, _english_source, duration, model_name = build_english_notes_from_transcript(
                transcript_path,
                config,
                client,
            )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_en)
            print(f"  ✓ 英文筆記已儲存：{output_path.name}")

        except Exception as e:
            print(f"  ✗ 錯誤：{e}")
            fail_count += 1
            continue

        if generate_zh_notes:
            try:
                markdown_zh = generate_zh_notes_from_english_markdown(
                    markdown_en,
                    stem,
                    duration,
                    config,
                    client,
                    model_name,
                )
                with open(zh_summary_path, "w", encoding="utf-8") as f:
                    f.write(markdown_zh)
                print(f"  ✓ 中文筆記已儲存：{zh_summary_path.name}")
            except Exception as e:
                print(f"  ⚠ 中文筆記生成失敗（英文已完成）：{e}")
        else:
            print("  ⏭ 已關閉中文筆記生成（notes.generate_zh_summary = false）")

        success_count += 1

    print(f"\n{'=' * 50}")
    print(f"完成！成功：{success_count}，失敗：{fail_count}")
    print(f"筆記儲存於：{notes_dir}")


if __name__ == "__main__":
    main()
