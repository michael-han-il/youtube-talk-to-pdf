# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencc-python-reimplemented",
#     "mlx-lm>=0.20; sys_platform == 'darwin' and platform_machine == 'arm64'",
# ]
# ///
"""
Turn a time range of a *_timestamped.txt transcript (from transcribe.py) into
readable Markdown: proper punctuation, paragraphs, and optional time headings.
A local LLM does the punctuation; nothing leaves the computer.

Backends (picked automatically with --backend auto):
    mlx     Apple Silicon Mac, via mlx-lm (default model Qwen2.5-7B-Instruct 4-bit)
    ollama  any OS with Ollama running (default model qwen2.5:7b; `ollama pull qwen2.5:7b`)
    none    no model: basic punctuation from the transcript's segment breaks

Usage:
    uv run format_transcript.py "transcripts/Talk_timestamped.txt" --start 40:10 --end 1:46:55
    uv run format_transcript.py FILE --language en                 # default: detect from the text
    uv run format_transcript.py FILE --fix-errors                  # also fix obvious misheard names/terms
    uv run format_transcript.py FILE --headings 0                  # no time headings
    uv run format_transcript.py FILE --backend ollama --model qwen2.5:14b
    uv run format_transcript.py FILE --model mlx-community/Qwen2.5-3B-Instruct-4bit   # 8 GB Macs

Output: <title>_<start>-<end>.md next to the input file (or --out PATH).

Safety check: each chunk the model returns is compared with the original
wording. If the model changed too much, that chunk falls back to the original
text with basic punctuation, and it is listed at the end of the run.
"""

from __future__ import annotations

import argparse
import difflib
import json
import platform
import re
import sys
import time
import urllib.request
from pathlib import Path

APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
DEFAULT_MODELS = {"mlx": "mlx-community/Qwen2.5-7B-Instruct-4bit", "ollama": "qwen2.5:7b"}
OLLAMA_URL = "http://localhost:11434"
OPENCC_TRAD = {"std": "s2t", "tw": "s2tw", "hk": "s2hk"}

LINE_RE = re.compile(r"^\[(\d+(?::\d{2}){1,2})\]\s*(.*)$")
PUNCT = set("，。、！？；：「」『』（）《》〈〉…—,.!?;:\"'()[]·-–")
HALF_TO_FULL = str.maketrans({",": "，", "?": "？", "!": "！", ";": "；", ":": "："})
CJK = r"぀-ヿ㐀-䶿一-鿿豈-﫿　-〿＀-￯"
NO_SPACE_LANGS = {"zh", "ja"}

LANG_NAMES = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "de": "German",
              "fr": "French", "es": "Spanish", "pt": "Portuguese", "it": "Italian", "nl": "Dutch",
              "ru": "Russian", "id": "Indonesian", "ms": "Malay", "vi": "Vietnamese", "th": "Thai",
              "hi": "Hindi", "ar": "Arabic", "tr": "Turkish", "pl": "Polish"}

PROMPT_ZH = (
    "你是中文逐字稿的校對員。以下是一段演講的語音辨識稿，字詞之間的空格代表停頓。\n"
    "任務：加上正確的全形標點符號，並依語意分段，段落之間空一行。\n"
    "規則：\n"
    "1. {rule1}\n"
    "2. 使用{script}中文。\n"
    "3. 只輸出處理後的文字，不要任何說明、標題或引號。"
)
ZH_STRICT = "不可增加、刪除、改寫或調換任何字詞，只能加標點符號和分段。"
ZH_FIX = "可以修正明顯的同音或近音錯字（例如人名、地名、專有名詞），但不可增加、刪除或改寫語句。"

PROMPT_GENERIC = (
    "You are proofreading a speech-recognition transcript of a talk in {lang}. "
    "Spaces may mark pauses.\n"
    "Task: add correct punctuation and capitalisation for {lang}, and split the text into "
    "paragraphs by topic, with one blank line between paragraphs.\n"
    "Rules:\n"
    "1. {rule1}\n"
    "2. Keep the text in {lang}; do not translate.\n"
    "3. Output only the processed text, with no notes, headings or quotation marks around it."
)
GEN_STRICT = "Do not add, delete, reword or reorder any words. Only add punctuation, capitalisation and paragraph breaks."
GEN_FIX = ("You may correct obviously misheard words (names, places, technical terms), "
           "but do not add, delete or rephrase sentences.")


# ---------------------------------------------------------------- parsing

def parse_time(s: str) -> float:
    secs = 0
    for p in s.strip().split(":"):
        secs = secs * 60 + int(p)
    return secs


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_time_slug(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def read_segments(path: Path) -> list[tuple[float, str]]:
    segs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line.strip())
        if m and m.group(2).strip():
            segs.append((parse_time(m.group(1)), m.group(2).strip()))
    if not segs:
        raise SystemExit(f"No [mm:ss] lines found in {path}")
    return segs


def detect_language(text: str) -> str:
    letters = [c for c in text if not c.isspace() and c not in PUNCT]
    if not letters:
        return "en"
    kana = sum("぀" <= c <= "ヿ" for c in letters)
    hangul = sum("가" <= c <= "힯" for c in letters)
    han = sum("一" <= c <= "鿿" for c in letters)
    n = len(letters)
    if kana / n > 0.05:
        return "ja"
    if hangul / n > 0.3:
        return "ko"
    if han / n > 0.3:
        return "zh"
    return "other"


def chunk_segments(segs, max_chars: int, heading_every: float):
    """Group segments into chunks of about max_chars, never crossing a
    heading boundary (so headings land between chunks)."""
    chunks, cur, cur_len = [], [], 0
    next_heading = segs[0][0] + heading_every if heading_every else None
    for t, text in segs:
        crosses = next_heading is not None and t >= next_heading
        if cur and (cur_len + len(text) > max_chars or crosses):
            chunks.append({"start": cur[0][0], "segs": cur, "heading": False})
            cur, cur_len = [], 0
        if crosses:
            while t >= next_heading:
                next_heading += heading_every
            chunks.append(None)  # heading marker
        cur.append((t, text))
        cur_len += len(text)
    if cur:
        chunks.append({"start": cur[0][0], "segs": cur, "heading": False})
    out, pending = [], bool(heading_every)
    for c in chunks:
        if c is None:
            pending = True
            continue
        c["heading"] = pending
        pending = False
        out.append(c)
    return out


# ---------------------------------------------------------------- text utils

def bare(text: str, norm) -> str:
    """Letters only, same script, lower case: for comparing wording."""
    return "".join(ch for ch in norm(text) if ch not in PUNCT and not ch.isspace()).lower()


def basic_punctuate(segs, lang: str) -> str:
    if lang == "zh":
        parts = [s.translate(HALF_TO_FULL).rstrip("，、 ") for _, s in segs]
        text = "，".join(p for p in parts if p)
        return text if text.endswith(("。", "？", "！")) else text + "。"
    if lang == "ja":
        text = "、".join(s.rstrip("、 ") for _, s in segs if s)
        return text if text.endswith(("。", "？", "！")) else text + "。"
    text = " ".join(s.strip() for _, s in segs if s.strip())
    return text if text.endswith((".", "?", "!")) else text + "."


def clean_model_output(text: str, lang: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.strip().strip("`").strip()
    text = re.sub(r"^(markdown|text)\s*\n", "", text)
    if lang == "zh":
        text = text.translate(HALF_TO_FULL)
    joiner = "" if lang in NO_SPACE_LANGS else " "
    paras = [re.sub(r"\s*\n\s*", joiner, p).strip() for p in re.split(r"\n\s*\n", text)]
    # Remove stray spaces between CJK characters only.
    paras = [re.sub(rf"(?<=[{CJK}])\s+(?=[{CJK}])", "", p) for p in paras]
    return "\n\n".join(p for p in paras if p)


# ---------------------------------------------------------------- backends

def ollama_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2):
            return True
    except Exception:
        return False


def make_generator(backend: str, model_name: str, system: str):
    if backend == "mlx":
        from mlx_lm import generate, load

        print(f"Loading {model_name} (first run downloads it)…")
        model, tokenizer = load(model_name)

        def gen(text: str) -> str:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            return generate(model, tokenizer, prompt=prompt, max_tokens=len(text) * 2 + 200, verbose=False)
        return gen

    if backend == "ollama":
        print(f"Using Ollama model {model_name} (run `ollama pull {model_name}` first if needed)…")

        def gen(text: str) -> str:
            body = json.dumps({
                "model": model_name, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                "options": {"temperature": 0, "num_ctx": 8192},
            }).encode()
            req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["message"]["content"]
        return gen

    return None


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="Format a transcript time range into punctuated Markdown")
    ap.add_argument("input", help="*_timestamped.txt from transcribe.py")
    ap.add_argument("--start", default="0:00", help="Start time, mm:ss or h:mm:ss (inclusive)")
    ap.add_argument("--end", help="End time, mm:ss or h:mm:ss (inclusive). Default: end of file")
    ap.add_argument("--language", default="auto", help="zh, en, ja, … (default: detect from the text)")
    ap.add_argument("--backend", choices=["auto", "mlx", "ollama", "none"], default="auto",
                    help="auto = mlx on Apple Silicon, else Ollama if running, else none")
    ap.add_argument("--model", help="Model for the backend (default: Qwen2.5 7B)")
    ap.add_argument("--fix-errors", action="store_true",
                    help="Also let the model fix obviously misheard words (names, places, terms)")
    ap.add_argument("--script", choices=["trad", "simp"], default="trad", help="Chinese only: output script")
    ap.add_argument("--variant", choices=["std", "tw", "hk"], default="tw",
                    help="Chinese Traditional only: tw (default), hk, or std")
    ap.add_argument("--headings", type=float, default=5,
                    help="Add a [mm:ss] heading about every N minutes (0 = none, default 5)")
    ap.add_argument("--chunk-chars", type=int, default=0,
                    help="Characters per model call (default: 400 for Chinese/Japanese, 1200 otherwise)")
    ap.add_argument("--title", help="Document title (default: from file name)")
    ap.add_argument("--out", help="Output .md path")
    ap.add_argument("--dry-run", action="store_true", help="Same as --backend none")
    args = ap.parse_args()

    src = Path(args.input).expanduser()
    segs = read_segments(src)
    t0 = parse_time(args.start)
    t1 = parse_time(args.end) if args.end else float("inf")
    segs = [(t, s) for t, s in segs if t0 <= t <= t1]
    if not segs:
        raise SystemExit("No transcript lines in that time range.")

    lang = args.language.lower() if args.language != "auto" else detect_language(" ".join(s for _, s in segs[:200]))
    if lang == "other":
        lang = "en"  # treated as a generic space-separated language below
    lang_name = LANG_NAMES.get(lang, "the original language")

    # Script normalisation (Chinese only).
    norm = lambda s: s  # noqa: E731  used for comparisons
    out_conv = lambda s: s  # noqa: E731  applied to output text
    if lang == "zh":
        from opencc import OpenCC
        to_simp = OpenCC("t2s").convert
        norm = to_simp
        if args.script == "simp":
            out_conv = to_simp
        else:
            s2t = OpenCC(OPENCC_TRAD[args.variant]).convert
            out_conv = lambda s: s2t(to_simp(s))  # noqa: E731

    backend = "none" if args.dry_run else args.backend
    if backend == "auto":
        backend = "mlx" if APPLE_SILICON else ("ollama" if ollama_running() else "none")
        if backend == "none":
            print("No local model available (not Apple Silicon, and Ollama isn't running). "
                  "Using basic punctuation. Install Ollama and run `ollama pull qwen2.5:7b` for better results.")
    if backend == "mlx" and not APPLE_SILICON:
        raise SystemExit("mlx-lm needs an Apple Silicon Mac. Use --backend ollama or --backend none.")
    if backend == "ollama" and not ollama_running():
        raise SystemExit(f"Ollama isn't reachable at {OLLAMA_URL}. Start the Ollama app, or use --backend none.")

    if lang == "zh":
        system = PROMPT_ZH.format(rule1=ZH_FIX if args.fix_errors else ZH_STRICT,
                                  script="簡體" if args.script == "simp" else "繁體")
    else:
        system = PROMPT_GENERIC.format(lang=lang_name, rule1=GEN_FIX if args.fix_errors else GEN_STRICT)
    generate_fn = make_generator(backend, args.model or DEFAULT_MODELS.get(backend, ""), system)

    title = args.title or re.sub(r"_timestamped$", "", src.stem)
    end_label = args.end or fmt_time(segs[-1][0])
    out = Path(args.out) if args.out else src.with_name(
        f"{title}_{fmt_time_slug(t0)}-{fmt_time_slug(parse_time(end_label))}.md")

    chunk_chars = args.chunk_chars or (400 if lang in NO_SPACE_LANGS else 1200)
    chunks = chunk_segments(segs, chunk_chars, args.headings * 60)
    print(f"Language {lang}, backend {backend}: {len(segs)} lines, "
          f"{sum(len(s) for _, s in segs)} characters, {len(chunks)} chunks")

    threshold = 0.85 if args.fix_errors else 0.95
    joiner = "" if lang in NO_SPACE_LANGS else " "
    body: list[str] = []
    fallbacks: list[str] = []
    started = time.time()
    for i, c in enumerate(chunks, 1):
        raw = " ".join((s.translate(HALF_TO_FULL).rstrip("，") if lang == "zh" else s) for _, s in c["segs"])
        result = None
        if generate_fn:
            try:
                cand = clean_model_output(out_conv(generate_fn(raw)), lang)
            except Exception as e:  # network hiccup, OOM, …: keep going
                cand, err = "", e
                print(f"\n  model error at {fmt_time(c['start'])}: {err}")
            ratio = difflib.SequenceMatcher(None, bare(raw, norm), bare(cand, norm)).ratio() if cand else 0.0
            if cand and ratio >= threshold:
                result = cand
            else:
                fallbacks.append(f"{fmt_time(c['start'])} (match {ratio:.0%})")
        if result is None:
            result = out_conv(basic_punctuate(c["segs"], lang))

        paras = result.split("\n\n")
        if c["heading"] or not body:
            body.append(f"### [{fmt_time(c['start'])}]")
            body.extend(paras)
        else:
            body[-1] += joiner + paras[0]  # chunk boundaries are arbitrary: continue the paragraph
            body.extend(paras[1:])

        elapsed = time.time() - started
        print(f"\r  chunk {i}/{len(chunks)}  ({elapsed:.0f}s, ~{elapsed / i * (len(chunks) - i):.0f}s left)",
              end="", flush=True)
    print()

    if lang == "zh":
        note = f"> 片段：{fmt_time(t0)} – {end_label}　·　由語音辨識產生，標點與分段由本機模型整理，可能仍有錯字。"
    else:
        note = (f"> Excerpt {fmt_time(t0)} – {end_label} · Generated by speech recognition; punctuation and "
                f"paragraphs added by a local model, so some errors may remain.")
    out.write_text("\n".join([f"# {title}", "", note, ""]) + "\n" + "\n\n".join(body) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    if fallbacks and generate_fn:
        print(f"{len(fallbacks)} chunk(s) kept original wording with basic punctuation "
              f"(model changed too much): {', '.join(fallbacks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
