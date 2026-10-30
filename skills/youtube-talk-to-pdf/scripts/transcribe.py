# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yt-dlp[default]>=2025.11.12",
#     "opencc-python-reimplemented",
#     "mlx-whisper>=0.4; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "faster-whisper>=1.1; sys_platform != 'darwin' or platform_machine != 'arm64'",
# ]
# ///
"""
Transcribe a YouTube talk (or a local audio/video file) with Whisper.

Backends (picked automatically):
    Apple Silicon Mac          mlx-whisper (GPU, fastest)
    Intel Mac / Linux / Win    faster-whisper (CPU, or NVIDIA GPU if present)

Usage:
    uv run transcribe.py "https://youtu.be/VIDEO_ID" --language zh
    uv run transcribe.py URL --language en
    uv run transcribe.py URL                         # auto-detect the language
    uv run transcribe.py URL --fast                  # turbo model: quicker, weaker on non-English
    uv run transcribe.py URL --language zh --script simp      # Chinese: Simplified output
    uv run transcribe.py URL --language zh --variant hk       # Chinese: Hong Kong forms (default tw)
    uv run transcribe.py ./talk.m4a --language ja             # local file
    uv run transcribe.py URL --cookies-from-browser chrome    # if YouTube returns 403

Outputs (in ./transcripts/ under the current folder, or --out DIR):
    <title>.txt               plain text in paragraphs, no timestamps
    <title>_timestamped.txt   [mm:ss] text, one segment per line
    <title>.srt               subtitle file

Needs ffmpeg on PATH. For YouTube, yt-dlp also needs a JavaScript runtime such as Deno.
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

MODELS = {
    "mlx": {"accurate": "mlx-community/whisper-large-v3-mlx", "fast": "mlx-community/whisper-large-v3-turbo"},
    "faster-whisper": {"accurate": "large-v3", "fast": "large-v3-turbo"},
}

# Short punctuated prompts nudge Whisper to emit punctuation in that language.
PROMPTS = {
    ("zh", "trad"): "以下是普通話的句子，請加上標點符號。",
    ("zh", "simp"): "以下是普通话的句子，请加上标点符号。",
    ("yue", "trad"): "以下係廣東話嘅句子，請加上標點符號。",
    ("ja", None): "以下は講演の書き起こしです。句読点を付けてください。",
    ("ko", None): "다음은 강연 녹취록입니다. 문장 부호를 넣어 주세요.",
    ("en", None): "Hello, welcome. This is a transcript of a talk, with punctuation.",
}

OPENCC_TRAD = {"std": "s2t", "tw": "s2tw", "hk": "s2hk"}
CJK_LANGS = {"zh", "yue", "ja", "ko"}  # joined without spaces between segments (ko keeps its own spaces)
SENTENCE_END = tuple("。！？!?…」』\".")


# ---------------------------------------------------------------- formatting

def fmt_mmss(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_segments(segments: list[dict], convert) -> list[dict]:
    """Strip, convert script, and drop empty or immediately-repeated segments
    (a common Whisper hallucination on long audio)."""
    out: list[dict] = []
    prev = None
    for seg in segments:
        text = convert(seg["text"].strip())
        if not text or text == prev:
            continue
        out.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": text})
        prev = text
    return out


def needs_space(left: str, right: str, spaced_language: bool) -> bool:
    if spaced_language or left[-1:] == " ":
        return not left.endswith(" ")
    # CJK text: only put a space next to Latin letters/digits.
    return (left[-1].isascii() and left[-1].isalnum()) or (right[0].isascii() and right[0].isalnum())


def to_paragraphs(segments: list[dict], spaced_language: bool, gap: float = 2.0, max_chars: int = 220) -> str:
    """Join segments into paragraphs: break after a pause longer than `gap`
    seconds, or once a paragraph is long and the last segment ended a sentence."""
    if spaced_language:
        max_chars *= 3  # Latin-script text uses ~3x the characters for the same content
    paras: list[str] = []
    cur = ""
    last_end = None
    for seg in segments:
        text = seg["text"]
        long_pause = last_end is not None and seg["start"] - last_end > gap
        too_long = len(cur) >= max_chars and cur.endswith(SENTENCE_END)
        if cur and (long_pause or too_long):
            paras.append(cur)
            cur = ""
        if cur and needs_space(cur, text, spaced_language):
            cur += " "
        cur += text
        last_end = seg["end"]
    if cur:
        paras.append(cur)
    return "\n\n".join(paras) + "\n"


def to_timestamped(segments: list[dict]) -> str:
    return "".join(f"[{fmt_mmss(s['start'])}] {s['text']}\n" for s in segments)


def to_srt(segments: list[dict]) -> str:
    return "\n".join(
        f"{i}\n{fmt_srt_time(s['start'])} --> {fmt_srt_time(s['end'])}\n{s['text']}\n"
        for i, s in enumerate(segments, 1)
    )


def safe_filename(name: str, limit: int = 80) -> str:
    name = re.sub(r'[\\/:*?"<>|\n\r\t]+', " ", name).strip().strip(".")
    name = re.sub(r"\s+", " ", name)
    return name[:limit].rstrip() or "transcript"


# ---------------------------------------------------------------- pipeline

def download_audio(url: str, workdir: Path, cookies_browser: str | None = None) -> tuple[Path, str]:
    import yt_dlp

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "audio.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
        "noprogress": True,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        print(
            "\nDownload failed. Things to try, in order:\n"
            "  1. Install a JavaScript runtime for yt-dlp (Deno):  brew install deno  /  winget install DenoLand.Deno\n"
            "  2. Refresh yt-dlp to the latest version:            uv run --refresh transcribe.py <url>\n"
            "  3. Use your browser's YouTube login:                add  --cookies-from-browser chrome\n"
            "  4. Download the audio another way, then run:        uv run transcribe.py path/to/audio.m4a\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    title = info.get("title") or info.get("id") or "transcript"
    audio = next(workdir.glob("audio.*"))
    return audio, title


def run_mlx(audio: Path, model: str, language: str | None, prompt: str | None) -> tuple[list[dict], str]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(audio), path_or_hf_repo=model, language=language, task="transcribe",
        initial_prompt=prompt, condition_on_previous_text=False, verbose=False,
    )
    return result.get("segments", []), result.get("language") or language or ""


def run_faster_whisper(audio: Path, model: str, language: str | None, prompt: str | None) -> tuple[list[dict], str]:
    from faster_whisper import WhisperModel

    wm = WhisperModel(model, device="auto", compute_type="auto")
    seg_iter, info = wm.transcribe(
        str(audio), language=language, task="transcribe", initial_prompt=prompt,
        condition_on_previous_text=False, vad_filter=True, beam_size=5,
    )
    total = info.duration or 0
    segments = []
    for s in seg_iter:  # generator: transcription happens while iterating
        segments.append({"start": s.start, "end": s.end, "text": s.text})
        if total:
            print(f"\r  {s.end / total:5.1%} of audio", end="", flush=True)
    print()
    return segments, info.language or language or ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe a talk with Whisper (mlx-whisper or faster-whisper)")
    ap.add_argument("source", help="YouTube URL or path to a local audio/video file")
    ap.add_argument("--language", default="auto",
                    help="Whisper language code: zh, en, ja, ko, yue, de, … (default: auto-detect)")
    ap.add_argument("--script", choices=["trad", "simp"], default="trad",
                    help="Chinese only: Traditional (default) or Simplified output")
    ap.add_argument("--variant", choices=["std", "tw", "hk"], default="tw",
                    help="Chinese Traditional only: tw (Taiwan, default), hk, or std (OpenCC classic forms such as 喫/爲)")
    ap.add_argument("--backend", choices=["auto", "mlx", "faster-whisper"], default="auto",
                    help="auto = mlx on Apple Silicon, faster-whisper elsewhere")
    ap.add_argument("--fast", action="store_true", help="Use the large-v3-turbo model (faster, weaker on non-English)")
    ap.add_argument("--model", help="Override the model (mlx-community repo, or faster-whisper model name/path)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-timestamps", action="store_true", help="Only write the plain .txt")
    g.add_argument("--timestamps-only", action="store_true", help="Only write _timestamped.txt and .srt")
    ap.add_argument("--out", default="transcripts", help="Output folder (default: ./transcripts)")
    ap.add_argument("--cookies-from-browser", metavar="BROWSER",
                    help="Use YouTube cookies from chrome/safari/firefox/brave/edge if downloads get 403")
    ap.add_argument("--keep-audio", action="store_true", help="Also save the downloaded audio")
    args = ap.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "mlx" if APPLE_SILICON else "faster-whisper"
    if backend == "mlx" and not APPLE_SILICON:
        raise SystemExit("mlx-whisper needs an Apple Silicon Mac. Use --backend faster-whisper.")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Install it (macOS: brew install ffmpeg; Windows: winget install ffmpeg).")

    language = None if args.language in ("auto", "") else args.language.lower()
    model = args.model or MODELS[backend]["fast" if args.fast else "accurate"]
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    def prompt_for(lang: str | None) -> str | None:
        if lang in ("zh", "yue"):
            return PROMPTS.get((lang, args.script)) or PROMPTS[("zh", args.script)]
        return PROMPTS.get((lang, None))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = Path(args.source).expanduser()
        if src.exists():
            audio, title = src, src.stem
        else:
            print("Downloading audio…")
            audio, title = download_audio(args.source, tmp, args.cookies_from_browser)
        base = safe_filename(title)
        print(f"Title   : {title}")
        print(f"Backend : {backend}   Model: {model}  (first run downloads the weights)")
        print(f"Language: {language or 'auto-detect'}")
        print("Transcribing… this can take a while for long videos.")

        t0 = time.time()
        runner = run_mlx if backend == "mlx" else run_faster_whisper
        raw_segments, detected = runner(audio, model, language, prompt_for(language))
        print(f"Done in {time.time() - t0:.0f}s (language: {detected or 'unknown'})")

        if args.keep_audio and audio.parent == tmp:
            shutil.copy(audio, out_dir / f"{base}{audio.suffix}")

    lang = (language or detected or "").lower()
    convert = lambda s: s  # noqa: E731
    if lang in ("zh", "yue"):
        from opencc import OpenCC
        to_simp = OpenCC("t2s").convert
        if args.script == "simp":
            convert = to_simp
        else:
            to_trad = OpenCC(OPENCC_TRAD[args.variant]).convert
            convert = lambda s: to_trad(to_simp(s))  # noqa: E731  normalise mixed forms

    segments = clean_segments(raw_segments, convert)
    if not segments:
        print("No speech detected.", file=sys.stderr)
        return 1

    spaced = lang not in CJK_LANGS or lang == "ko"
    written = []
    if not args.timestamps_only:
        p = out_dir / f"{base}.txt"
        p.write_text(to_paragraphs(segments, spaced), encoding="utf-8")
        written.append(p)
    if not args.no_timestamps:
        p = out_dir / f"{base}_timestamped.txt"
        p.write_text(to_timestamped(segments), encoding="utf-8")
        written.append(p)
        p = out_dir / f"{base}.srt"
        p.write_text(to_srt(segments), encoding="utf-8")
        written.append(p)

    print("Wrote:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
