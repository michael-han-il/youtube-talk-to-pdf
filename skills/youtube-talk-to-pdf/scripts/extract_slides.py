# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yt-dlp[default]>=2025.11.12",
#     "scenedetect[opencv]>=0.6.4",
#     "torch",
#     "transformers>=4.40",
#     "transformers<4.50; sys_platform == 'darwin' and platform_machine == 'x86_64'",
#     "pillow",
#     "imagehash",
#     "numpy",
#     "numpy<2; sys_platform == 'darwin' and platform_machine == 'x86_64'",
# ]
# ///
"""
Pass 1 of the slides workflow: find the shots in a lecture video that show the
presentation slides (not the speaker), save one clear image per distinct slide,
and keep a trimmed copy of the video for the chosen time range.

Usage:
    uv run extract_slides.py "https://youtu.be/VIDEO_ID" --start 40:10 --end 1:46:55
    uv run extract_slides.py path/to/video.mp4 --start 40:10 --end 1:46:55
    uv run extract_slides.py URL --start 40:10 --end 1:46:55 --cookies-from-browser chrome

Runs on Apple Silicon (GPU via MPS), NVIDIA GPUs (CUDA) or plain CPU (Intel Macs,
Windows, Linux). On Intel Macs PyTorch is capped at 2.2, which the pins above handle.

Outputs (in ./transcripts/ under the current folder by default):
    <title>_<start>-<end>.mp4          the video trimmed to the range
    slides/<mmmm-ss>.jpg               one image per distinct slide, named by
                                       its time in the full video (0052-14 = 52:14)
    slides/_contact_sheet.jpg          all slides on one page, for quick review
    slides/_slides.json                times and scores for each saved slide

Then: delete any wrong images from slides/, and run insert_slides.py.

Needs ffmpeg and, for YouTube, the Deno JavaScript runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CLIP_MODEL = "openai/clip-vit-base-patch16"

# Zero-shot labels. Scores are summed per group.
SLIDE_PROMPTS = [
    "a presentation slide",
    "a projected slide with text",
    "a slide with a chart, table or map",
    "a slide with an old photograph and a caption",
    "a screenshot of a PowerPoint slide",
]
OTHER_PROMPTS = [
    "a person speaking at a podium",
    "a man giving a lecture",
    "a close-up of a speaker's face",
    "an audience sitting in a hall",
    "a wide shot of a stage with a speaker",
]


# ---------------------------------------------------------------- helpers

def parse_time(s: str) -> float:
    parts = [float(p) for p in s.strip().split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def fmt_time(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_slug(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def slide_name(seconds: float) -> str:
    """0052-14 for 52:14, 0106-30 for 1:06:30 (total minutes, seconds)."""
    s = int(seconds)
    return f"{s // 60:04d}-{s % 60:02d}"


def safe_filename(name: str, limit: int = 80) -> str:
    name = re.sub(r'[\\/:*?"<>|\n\r\t]+', " ", name).strip().strip(".")
    return re.sub(r"\s+", " ", name)[:limit].rstrip() or "video"


# ---------------------------------------------------------------- steps

def download_video(url: str, workdir: Path, height: int, cookies_browser: str | None) -> tuple[Path, str]:
    import yt_dlp

    opts = {
        "format": f"bv*[height<={height}][ext=mp4]+ba[ext=m4a]/b[height<={height}][ext=mp4]/bv*[height<={height}]+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "full.%(ext)s"),
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
            "\nDownload failed. Try: install Deno (brew install deno / winget install DenoLand.Deno);\n"
            "uv run --refresh extract_slides.py …;\n"
            "or add --cookies-from-browser chrome;  or pass a local video file instead of the URL.",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    title = info.get("title") or info.get("id") or "video"
    video = next(p for p in workdir.glob("full.*") if p.suffix in (".mp4", ".mkv", ".webm"))
    return video, title


def trim_video(src: Path, dst: Path, start: float, end: float | None) -> None:
    """Frame-accurate trim (re-encodes). Uses the Mac's hardware encoder when available."""
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src)]
    if end is not None:
        base += ["-t", f"{end - start:.3f}"]
    tail = ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst)]
    encoders = []
    if sys.platform == "darwin":
        encoders.append(["-c:v", "h264_videotoolbox", "-b:v", "2500k"])
    encoders.append(["-c:v", "h264_nvenc", "-b:v", "2500k"])
    encoders.append(["-c:v", "libx264", "-preset", "veryfast", "-crf", "22"])
    last_err = ""
    for vcodec in encoders:
        r = subprocess.run(base + vcodec + tail, capture_output=True, text=True)
        if r.returncode == 0:
            return
        last_err = r.stderr
    raise SystemExit(f"ffmpeg could not trim the video:\n{last_err[-800:]}")


def detect_shots(video: Path, threshold: float) -> list[tuple[float, float]]:
    from scenedetect import ContentDetector, SceneManager, open_video

    v = open_video(str(video))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=12))
    sm.detect_scenes(v, show_progress=True)
    scenes = sm.get_scene_list(start_in_scene=True)
    return [(a.get_seconds(), b.get_seconds()) for a, b in scenes]


def grab_best_frame(cap, start: float, end: float, samples: int = 5):
    """Sample frames from the settled middle of a shot; return the sharpest as RGB."""
    import cv2

    best, best_sharp = None, -1.0
    span = end - start
    for k in range(samples):
        t = start + span * (0.2 + 0.6 * k / max(samples - 1, 1))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if sharp > best_sharp:
            best, best_sharp = frame, sharp
    if best is None:
        return None
    return cv2.cvtColor(best, cv2.COLOR_BGR2RGB)


def split_slide_shot(cap, start: float, end: float, change: int, step: float = 1.0):
    """A slide shot can contain several slides with no cut to the speaker in
    between (the presenter just clicks). Sample about once a second and start a
    new slide whenever the picture changes by more than `change` (image-hash
    distance). Returns [(time, sharpest RGB frame)] per slide."""
    import cv2
    import imagehash
    from PIL import Image

    groups = []  # [first_time, ref_hash, best_frame, best_sharpness]
    t = start + min(0.4, (end - start) / 4)
    while t < end - 0.2:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h = imagehash.phash(Image.fromarray(rgb))
            sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if not groups or h - groups[-1][1] > change:
                groups.append([t, h, rgb, sharp])
            elif sharp > groups[-1][3]:
                groups[-1][2], groups[-1][3] = rgb, sharp
        t += step
    return [(g[0], g[2]) for g in groups]


class SlideClassifier:
    def __init__(self, model_name: str = CLIP_MODEL):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Loading {model_name} on {self.device} (first run downloads it)…")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.prompts = SLIDE_PROMPTS + OTHER_PROMPTS

    def slide_scores(self, images) -> list[float]:
        out = []
        for i in range(0, len(images), 16):
            batch = images[i:i + 16]
            inputs = self.proc(text=self.prompts, images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                probs = self.model(**inputs).logits_per_image.softmax(dim=-1).cpu()
            out.extend(probs[:, :len(SLIDE_PROMPTS)].sum(dim=-1).tolist())
        return out


def contact_sheet(items: list[tuple[Path, float]], dst: Path, cols: int = 4, thumb_w: int = 400) -> None:
    from PIL import Image, ImageDraw

    if not items:
        return
    thumbs = []
    for path, t in items:
        im = Image.open(path).convert("RGB")
        im.thumbnail((thumb_w, thumb_w))
        thumbs.append((im, t, path.stem))
    th = max(im.height for im, _, _ in thumbs)
    label_h = 24
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (thumb_w + 10) + 10, rows * (th + label_h + 10) + 10), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, t, stem) in enumerate(thumbs):
        x = 10 + (i % cols) * (thumb_w + 10)
        y = 10 + (i // cols) * (th + label_h + 10)
        sheet.paste(im, (x, y))
        draw.text((x, y + th + 4), f"{fmt_time(t)}   ({stem}.jpg)", fill="black")
    sheet.save(dst, quality=85)


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="Extract presentation slides from a lecture video")
    ap.add_argument("source", help="YouTube URL or local video file")
    ap.add_argument("--start", default="0:00", help="Start time, mm:ss or h:mm:ss")
    ap.add_argument("--end", help="End time, mm:ss or h:mm:ss (default: end of video)")
    ap.add_argument("--out", default="transcripts", help="Output folder (default: ./transcripts)")
    ap.add_argument("--height", type=int, default=720, help="Max download resolution (default 720)")
    ap.add_argument("--cookies-from-browser", metavar="BROWSER", help="chrome/safari/firefox… if YouTube returns 403")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Slide score needed to keep a shot, 0–1 (default 0.5; lower keeps more)")
    ap.add_argument("--cut-sensitivity", type=float, default=27.0,
                    help="PySceneDetect threshold; lower finds more cuts (default 27)")
    ap.add_argument("--min-shot", type=float, default=1.5, help="Ignore shots shorter than N seconds")
    ap.add_argument("--dedup", type=int, default=6,
                    help="Max image-hash distance to treat two slides as the same (default 6; 0 = keep repeats)")
    ap.add_argument("--slide-change", type=int, default=8,
                    help="Image-hash change that counts as a new slide within one shot (default 8; lower splits more)")
    ap.add_argument("--no-trimmed-video", action="store_true", help="Don't keep the trimmed video")
    ap.add_argument("--keep-full-video", action="store_true", help="Also keep the full downloaded video")
    ap.add_argument("--overwrite", action="store_true", help="Replace images already in slides/")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Install it (macOS: brew install ffmpeg; Windows: winget install ffmpeg).")

    out_dir = Path(args.out).expanduser()
    slides_dir = out_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    existing = list(slides_dir.glob("*.jpg"))
    if existing and not args.overwrite:
        raise SystemExit(f"{slides_dir} already has {len(existing)} images. Use --overwrite to replace them.")
    for p in existing:
        p.unlink()

    start = parse_time(args.start)
    end = parse_time(args.end) if args.end else None

    import cv2
    import imagehash
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = Path(args.source).expanduser()
        if src.exists():
            full, title = src, src.stem
        else:
            print("Downloading video…")
            full, title = download_video(args.source, tmp, args.height, args.cookies_from_browser)
            if args.keep_full_video:
                shutil.copy(full, out_dir / f"{safe_filename(title)}{full.suffix}")

        clip_name = f"{safe_filename(title)}_{fmt_slug(start)}-{fmt_slug(end) if end else 'end'}.mp4"
        clip = (tmp / clip_name) if args.no_trimmed_video else (out_dir / clip_name)
        print(f"Trimming {fmt_time(start)} – {fmt_time(end) if end else 'end'}…")
        trim_video(full, clip, start, end)
        if not args.no_trimmed_video:
            print(f"  saved {clip}")

        print("Finding camera cuts…")
        shots = [(a, b) for a, b in detect_shots(clip, args.cut_sensitivity) if b - a >= args.min_shot]
        print(f"  {len(shots)} shots of {args.min_shot}s or longer")

        cap = cv2.VideoCapture(str(clip))
        frames, kept_shots = [], []
        for i, (a, b) in enumerate(shots, 1):
            f = grab_best_frame(cap, a, b)
            if f is not None:
                frames.append(Image.fromarray(f))
                kept_shots.append((a, b))
            print(f"\r  grabbing frames {i}/{len(shots)}", end="", flush=True)
        cap.release()
        print()

        clf = SlideClassifier()
        scores = clf.slide_scores(frames)

        saved, hashes, manifest = [], [], []
        cap = cv2.VideoCapture(str(clip))
        slide_shots = [(a, b, sc) for (a, b), sc in zip(kept_shots, scores) if sc >= args.threshold]
        print(f"  {len(slide_shots)} shots look like slides; checking each for slide changes…")
        for a, b, score in slide_shots:
            for t_rel, rgb in split_slide_shot(cap, a, b, args.slide_change):
                img = Image.fromarray(rgb)
                h = imagehash.phash(img)
                if args.dedup and any(h - prev <= args.dedup for prev in hashes):
                    continue  # already saved earlier in the talk
                hashes.append(h)
                t_abs = start + t_rel
                path = slides_dir / f"{slide_name(t_abs)}.jpg"
                n = 2
                while path.exists():  # two slides in the same second
                    path = slides_dir / f"{slide_name(t_abs)}_{n}.jpg"
                    n += 1
                img.save(path, quality=92)
                saved.append((path, t_abs))
                manifest.append({"file": path.name, "time": round(t_abs, 2), "time_str": fmt_time(t_abs),
                                 "shot": f"{fmt_time(start + a)}–{fmt_time(start + b)}",
                                 "slide_score": round(score, 3)})
        cap.release()

    (slides_dir / "_slides.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet(saved, slides_dir / "_contact_sheet.jpg")

    print(f"\nSaved {len(saved)} slide images to {slides_dir}")
    print(f"Review {slides_dir / '_contact_sheet.jpg'}, delete any wrong images, then run insert_slides.py.")
    if not saved:
        print("No slides found — try --threshold 0.35 or --cut-sensitivity 20.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
