# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "markdown",
#     "opencc-python-reimplemented",
# ]
# ///
"""
Pass 2 of the slides workflow: place the slide images from extract_slides.py
into the formatted Markdown transcript, then export a PDF.

Each slide is timed by its file name (0052-14.jpg = 52:14). The original
*_timestamped.txt transcript says what was being said at that moment; the
script finds that wording in the formatted Markdown and places the slide at
the nearest sentence or paragraph break, never mid-sentence.

Usage:
    uv run insert_slides.py \\
        --md "transcripts/第29次中山講座_40m10s-1h46m55s.md" \\
        --transcript "transcripts/第29次中山講座_timestamped.txt"

    Options: --slides DIR (default: slides/ next to the .md)   --no-pdf   --font-size 14

Outputs next to the .md:
    <name>_with_slides.md     Markdown with the images (keep slides/ beside it)
    <name>_with_slides.pdf    PDF with images embedded (exported with Chrome, Edge or Brave)
"""

from __future__ import annotations

import argparse
import bisect
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LINE_RE = re.compile(r"^\[(\d+(?::\d{2}){1,2})\]\s*(.*)$")
SLIDE_RE = re.compile(r"^(\d{4})-(\d{2})(?:_\d+)?\.(jpg|jpeg|png)$", re.I)
PUNCT = set("，。、！？；：「」『』（）《》〈〉…—,.!?;:\"'()[]·　 ")
SENT_END = "。！？!?…"
LATIN_END = ".!?"  # count as sentence ends only when followed by a space
CLOSERS = "」』）)\"'"

BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "msedge",
]


def parse_time(s: str) -> int:
    t = 0
    for p in s.split(":"):
        t = t * 60 + int(p)
    return t


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------- alignment

class BareText:
    """Characters of a text with punctuation/spaces removed, normalised to
    Simplified + lower case, with a map back to the original positions."""

    def __init__(self, norm):
        self.norm = norm
        self.chars: list[str] = []
        self.where: list[tuple[int, int]] = []  # (line index, char index)

    def add(self, text: str, line_idx: int):
        for ci, ch in enumerate(text):
            if ch in PUNCT or ch.isspace():
                continue
            self.chars.append(ch)
            self.where.append((line_idx, ci))

    def finish(self):
        s = "".join(self.chars)
        self.text = self.norm(s).lower()
        if len(self.text) != len(self.chars):  # conversion changed length; fall back
            self.text = s.lower()
        return self


def bare(text: str, norm) -> str:
    return norm("".join(c for c in text if c not in PUNCT and not c.isspace())).lower()


def longest_increasing(anchors):
    """Keep the longest run of anchors whose md positions increase with time."""
    if not anchors:
        return []
    tails, tails_idx, prev = [], [], [-1] * len(anchors)
    for i, (_, pos) in enumerate(anchors):
        k = bisect.bisect_left(tails, pos)
        if k == len(tails):
            tails.append(pos)
            tails_idx.append(i)
        else:
            tails[k] = pos
            tails_idx[k] = i
        prev[i] = tails_idx[k - 1] if k else -1
    out, i = [], tails_idx[-1]
    while i != -1:
        out.append(anchors[i])
        i = prev[i]
    return out[::-1]


def build_anchors(segments, md_bare: str, norm):
    """(time, md position) pairs for transcript lines whose wording appears
    exactly once in the Markdown. Keys grow until unique, so repetitive
    phrasing and Latin-script text (more letters per word) still anchor."""
    anchors = []
    for t, text in segments:
        b = bare(text, norm)
        if len(b) < 6:
            continue
        for key_len in (8, 16, 32, len(b)):
            key = b[:key_len]
            first = md_bare.find(key)
            if first == -1:
                break  # wording changed here; longer keys won't match either
            if md_bare.find(key, first + 1) == -1:
                anchors.append((t, first))
                break
            if key_len >= len(b):
                break
    return longest_increasing(anchors)


def locate(t: float, segments, anchors, md_bare: str, norm) -> int:
    """Position in md_bare that corresponds to video time t."""
    times = [a[0] for a in anchors]
    poss = [a[1] for a in anchors]
    # Linear interpolation between surrounding anchors.
    k = bisect.bisect_right(times, t)
    if k == 0:
        guess = poss[0]
    elif k == len(times):
        guess = poss[-1]
    else:
        t0, t1, p0, p1 = times[k - 1], times[k], poss[k - 1], poss[k]
        guess = int(p0 + (p1 - p0) * (t - t0) / max(t1 - t0, 1e-6))
    # Refine: find the words spoken at time t near the guess.
    seg_times = [s[0] for s in segments]
    j = max(bisect.bisect_right(seg_times, t) - 1, 0)
    for text in (segments[j][1], " ".join(s[1] for s in segments[j:j + 2])):
        b = bare(text, norm)
        for key in (b[:32], b[:16], b[:10], b[:6]):
            if len(key) < 4:
                continue
            best, idx = None, md_bare.find(key, max(guess - 400, 0))
            while idx != -1 and idx <= guess + 400:
                if best is None or abs(idx - guess) < abs(best - guess):
                    best = idx
                idx = md_bare.find(key, idx + 1)
            if best is not None:
                return best
    return max(0, min(guess, len(md_bare) - 1))


# ---------------------------------------------------------------- placement

def sentence_breaks(line: str) -> list[int]:
    """Character offsets in a paragraph where a sentence ends (split points)."""
    out = []
    i = 0
    while i < len(line):
        latin = line[i] in LATIN_END and i + 1 < len(line) and line[i + 1] == " "
        if latin:
            j = i + 1
            while j < len(line) and line[j] == " ":
                j += 1
            if j < len(line):
                out.append(j)
            i = j
        elif line[i] in SENT_END:
            j = i + 1
            while j < len(line) and (line[j] in SENT_END or line[j] in CLOSERS):
                j += 1
            if j < len(line):
                out.append(j)
            i = j
        else:
            i += 1
    return out


def choose_break(line: str, ci: int, min_piece: int, para_pull: int) -> int:
    """Pick where in this paragraph to put a slide spoken at char ci.
    Returns 0 (before paragraph), len(line) (after), or a sentence break."""
    n = len(line)
    if ci <= para_pull:
        return 0
    if n - ci <= para_pull // 2:
        return n
    # The start of the sentence being spoken, so the slide precedes its discussion.
    cands = [b for b in sentence_breaks(line) if min_piece <= b <= n - min_piece]
    before = [b for b in cands if b <= ci]
    if before:
        return before[-1]
    after = [b for b in cands if b > ci]
    # No full sentence before it in this paragraph: use paragraph start
    # unless the next sentence break is much closer.
    if after and after[0] - ci < ci:
        return after[0]
    return 0


LABEL = {"zh": "投影片", "ja": "スライド", "ko": "슬라이드"}


def figure_md(rel: str, t: float, label: str) -> str:
    return f"![{label} {fmt_time(t)}]({rel})"


def insert_slides(md_lines, slides, segments, norm, label, min_piece=30, para_pull=60):
    # Body paragraphs: everything that is not a heading, quote, image or blank.
    def is_body(line):
        s = line.strip()
        return bool(s) and not s.startswith(("#", ">", "![", "<!--"))

    bt = BareText(norm)
    for li, line in enumerate(md_lines):
        if is_body(line):
            bt.add(line, li)
    bt.finish()
    if not bt.text:
        raise SystemExit("The Markdown file has no body text.")

    anchors = build_anchors(segments, bt.text, norm)
    if len(anchors) < 3:
        raise SystemExit("Could not match the transcript to the Markdown. Is it the same talk?")
    print(f"Matched {len(anchors)} transcript lines to the Markdown for alignment")

    # Decide (line, split offset) for every slide first, then apply from the bottom up.
    placements: dict[tuple[int, int], list[tuple[float, str]]] = {}
    for t, rel in slides:
        pos = locate(t, segments, anchors, bt.text, norm)
        li, ci = bt.where[pos]
        cut = choose_break(md_lines[li], ci, min_piece, para_pull)
        placements.setdefault((li, cut), []).append((t, rel))

    lines = list(md_lines)
    for (li, cut) in sorted(placements, reverse=True):
        figs = [figure_md(rel, t, label) for t, rel in sorted(placements[(li, cut)])]
        line = lines[li]
        block = []
        for f in figs:
            block += [f, ""]
        if cut == 0:
            lines[li:li] = block
        elif cut >= len(line):
            lines[li + 1:li + 1] = [""] + block[:-1]
        else:
            lines[li:li + 1] = [line[:cut].rstrip(), ""] + block + [line[cut:].lstrip()]
    # Collapse runs of blank lines.
    out = []
    for line in lines:
        if line.strip() == "" and out and out[-1].strip() == "":
            continue
        out.append(line)
    return out


# ---------------------------------------------------------------- PDF

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
html { -webkit-print-color-adjust: exact; }
body { font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, "PingFang TC", "Hiragino Sans",
       "Apple SD Gothic Neo", "Noto Sans CJK TC", "Microsoft JhengHei", "Yu Gothic", "Malgun Gothic", sans-serif;
       font-size: %(size)spt; line-height: 1.85; color: #111; max-width: 100%%; }
h1 { font-size: 1.6em; margin: 0 0 .4em; }
h3 { font-size: 1em; color: #666; font-weight: 600; margin: 1.6em 0 .4em;
     border-top: 1px solid #ddd; padding-top: .6em; break-after: avoid; }
blockquote { color: #555; border-left: 3px solid #ccc; margin: 0 0 1.2em; padding: .2em 0 .2em .9em; font-size: .85em; }
p { margin: 0 0 .9em; text-align: justify; }
figure { margin: 1.1em 0 1.3em; break-inside: avoid; text-align: center; }
figure img { max-width: 100%%; max-height: 120mm; border: 1px solid #ccc; }
figcaption { font-size: .75em; color: #666; margin-top: .3em; }
"""


def to_html(md_text: str, title: str, font_size: float, lang: str) -> str:
    import markdown

    body = markdown.markdown(md_text, extensions=["extra"])
    body = re.sub(
        r'<p>\s*<img alt="([^"]*)" src="([^"]+)"\s*/?>\s*</p>',
        lambda m: f'<figure><img src="{m.group(2)}" alt="{m.group(1)}"><figcaption>{m.group(1)}</figcaption></figure>',
        body,
    )
    return (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
            f"<title>{html.escape(title)}</title><style>{CSS % {'size': font_size}}</style></head>"
            f"<body>{body}</body></html>")


def export_pdf(html_path: Path, pdf_path: Path) -> bool:
    browser = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if not browser:
        print("No Chrome/Edge/Brave found for PDF export. Open the .html file in a browser and "
              "use Print → Save as PDF instead.", file=sys.stderr)
        return False
    with tempfile.TemporaryDirectory() as profile:
        cmd = [browser, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
               "--no-pdf-header-footer", "--print-to-pdf-no-header",
               f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            cmd.insert(1, "--no-sandbox")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not pdf_path.exists():
        print(f"PDF export failed:\n{r.stderr[-800:]}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="Insert slide images into a formatted transcript and export PDF")
    ap.add_argument("--md", required=True, help="Formatted Markdown from format_transcript.py")
    ap.add_argument("--transcript", required=True, help="Original *_timestamped.txt from transcribe.py")
    ap.add_argument("--slides", help="Folder of slide images (default: slides/ next to the .md)")
    ap.add_argument("--out", help="Output .md path (default: <name>_with_slides.md)")
    ap.add_argument("--no-pdf", action="store_true", help="Skip the PDF export")
    ap.add_argument("--font-size", type=float, default=13, help="PDF body font size in pt (default 13)")
    ap.add_argument("--label", help="Caption word before each slide time (default by language: Slide / 投影片 / …)")
    args = ap.parse_args()

    from opencc import OpenCC
    norm = OpenCC("t2s").convert

    md_path = Path(args.md).expanduser()
    slides_dir = Path(args.slides).expanduser() if args.slides else md_path.parent / "slides"
    out_md = Path(args.out).expanduser() if args.out else md_path.with_name(md_path.stem + "_with_slides.md")

    segments = []
    for line in Path(args.transcript).expanduser().read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line.strip())
        if m and m.group(2).strip():
            segments.append((parse_time(m.group(1)), m.group(2).strip()))
    if not segments:
        raise SystemExit("No [mm:ss] lines in the transcript file.")

    slides = []
    for p in sorted(slides_dir.glob("*")):
        m = SLIDE_RE.match(p.name)
        if m:
            t = int(m.group(1)) * 60 + int(m.group(2))
            slides.append((t, os.path.relpath(p, out_md.parent).replace(os.sep, "/")))
    if not slides:
        raise SystemExit(f"No slide images (like 0052-14.jpg) found in {slides_dir}")
    print(f"{len(slides)} slides from {slides_dir}")

    md_lines = md_path.read_text(encoding="utf-8").splitlines()
    sample = "".join(md_lines[:200])
    if re.search(r"[\u3040-\u30ff]", sample):
        lang = "ja"
    elif re.search(r"[\uac00-\ud7af]", sample):
        lang = "ko"
    elif len(re.findall(r"[\u4e00-\u9fff]", sample)) > len(sample) * 0.2:
        lang = "zh"
    else:
        lang = "en"
    label = args.label or LABEL.get(lang, "Slide")
    new_lines = insert_slides(md_lines, slides, segments, norm, label)
    out_md.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")

    if not args.no_pdf:
        title = next((l[2:].strip() for l in new_lines if l.startswith("# ")), out_md.stem)
        html_path = out_md.with_suffix(".html")
        html_path.write_text(to_html("\n".join(new_lines), title, args.font_size, {"zh": "zh-Hant"}.get(lang, lang)), encoding="utf-8")
        pdf_path = out_md.with_suffix(".pdf")
        if export_pdf(html_path, pdf_path):
            print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
