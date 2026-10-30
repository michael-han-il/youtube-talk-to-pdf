---
name: youtube-talk-to-pdf
description: Turns a recorded talk, lecture or conference session (YouTube URL or local video) into a readable PDF transcript with the presentation slides placed where they were shown. Use when the user asks to "transcribe this talk", "make a PDF of this lecture with the slides", "get the slides from this YouTube video", "turn this conference video into notes", or wants a timestamped/plain transcript of a talk in any language (Chinese, English, Japanese, …). Has extra Chinese support: Traditional or Simplified output, Taiwan/Hong Kong character standards, full-width punctuation, and Mandarin/Cantonese prompts. Runs fully locally on Apple Silicon (fastest), Intel Macs, Windows or Linux.
---

# YouTube talk → transcript + slides PDF

Four bundled scripts in `scripts/`, each a self-contained `uv` script (dependencies declared inline, installed on first run):

| Step | Script | Produces |
|---|---|---|
| 1 | `transcribe.py` | `<title>.txt`, `<title>_timestamped.txt`, `<title>.srt` |
| 2 | `format_transcript.py` | `<title>_<start>-<end>.md` — punctuated paragraphs, `### [mm:ss]` headings |
| 3 | `extract_slides.py` | trimmed `.mp4`, `slides/<mmmm-ss>.jpg`, `slides/_contact_sheet.jpg` |
| — | **user reviews slides** | deletes wrong images |
| 4 | `insert_slides.py` | `<…>_with_slides.md`, `.html`, `.pdf` |

Everything runs on the user's own computer. Nothing is uploaded anywhere except the YouTube download itself.

## Usage boundaries

- Use for personal study, accessibility (e.g. a relative who reads better than listens), or talks the user has rights to. Remind the user once, briefly, that the output copies the speaker's talk and slides and shouldn't be redistributed without permission.
- Do not paste the transcript or long passages of it into the chat. Point the user to the files. Short quotes for a specific question are fine.
- Don't help bypass DRM, paywalls or private videos. The cookie option only reuses the user's own logged-in browser session.

## Step 0 — Gather inputs

Collect what's missing (one AskUserQuestion call, skip anything already given):

1. **Source**: a YouTube URL or local video/audio path.
2. **Time range** (optional): e.g. `40:10`–`1:46:55` to skip intros/Q&A. Default is the whole video.
3. **Language**: Whisper code (`zh`, `en`, `ja`, `ko`, `yue`, …). Default `auto`. For Chinese also ask **Traditional or Simplified**, and for Traditional the standard (`tw` default, `hk`, `std`).
4. **Output folder**: where the user wants the files (a connected folder in Cowork, or the current directory in a CLI).

## Step 1 — Decide where the scripts run

The scripts need the user's real machine: GPU access, several GB of model downloads, and YouTube network access.

- **Claude Code / CLI on the user's machine**: run the commands directly with Bash from the output folder. Use `${CLAUDE_PLUGIN_ROOT}/skills/youtube-talk-to-pdf/scripts/<name>.py` (or this skill's base directory + `scripts/`) as the script path.
- **Cowork with a connected folder**: the Cowork shell on the user's computer is a Linux VM. It has no Apple GPU and usually can't reach YouTube, so don't run the pipeline there. Instead copy the four scripts into a `talk-to-pdf-tools/` subfolder of the connected folder (write them with the file tools or `device_commit_files`), then give the user the Terminal commands below, with `cd` into their folder. Run the cheap checks yourself where possible (listing outputs, reading the contact sheet image after they stage it).
- **No access to the user's machine**: hand the user the scripts as files plus the commands.

Prerequisites to confirm (ask the user to run, or check with `which`):

| Need | macOS | Windows | Linux |
|---|---|---|---|
| uv | `brew install uv` | `winget install astral-sh.uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| ffmpeg | `brew install ffmpeg` | `winget install ffmpeg` | `apt install ffmpeg` |
| Deno (yt-dlp YouTube support) | `brew install deno` | `winget install DenoLand.Deno` | `curl -fsSL https://deno.land/install.sh \| sh` |
| PDF export | Chrome, Edge or Brave | Chrome or Edge | `google-chrome`/`chromium` |
| Formatting LLM (non-Apple-Silicon only, optional) | Ollama + `ollama pull qwen2.5:7b` | same | same |

Backends are picked automatically:

| Machine | Transcription | Formatting LLM | Slide classifier |
|---|---|---|---|
| Apple Silicon Mac | mlx-whisper (GPU) | mlx-lm Qwen2.5-7B 4-bit | CLIP on MPS |
| Intel Mac | faster-whisper (CPU, int8) | Ollama if running, else basic punctuation | CLIP on CPU (PyTorch 2.2, pinned automatically) |
| Windows / Linux + NVIDIA | faster-whisper (CUDA) | Ollama, else basic | CLIP on CUDA |
| Windows / Linux CPU | faster-whisper (CPU) | Ollama, else basic | CLIP on CPU |

Tell CPU-only users that transcription of an hour of audio with `large-v3` can take an hour or more, and suggest `--fast` (turbo model) when the talk is in English.

## Step 2 — Run the pipeline

Use `S=talk-to-pdf-tools` (or the plugin path) and run from the output folder. Example for a Chinese talk, Traditional, range 40:10–1:46:55:

```bash
uv run $S/transcribe.py "<URL>" --language zh
uv run $S/format_transcript.py "transcripts/<title>_timestamped.txt" --start 40:10 --end 1:46:55
uv run $S/extract_slides.py "<URL>" --start 40:10 --end 1:46:55
```

Then **stop and ask the user to review** `transcripts/slides/_contact_sheet.jpg` and delete wrong images from `transcripts/slides/`. If the folder is reachable, stage the contact sheet and look at it yourself to point out likely false positives, such as audience shots or a wide stage shot. Don't delete anything without the user's say-so.

```bash
uv run $S/insert_slides.py --md "transcripts/<title>_40m10s-1h46m55s.md" \
                           --transcript "transcripts/<title>_timestamped.txt"
```

Notes:
- Pass the same `--start/--end` to `format_transcript.py` and `extract_slides.py`. `insert_slides.py` takes slide times from file names (`0052-14.jpg` = 52:14 in the full video) and aligns them using the original timestamped transcript. Never edit that file between steps.
- `transcribe.py` and `extract_slides.py` each download from YouTube. If the user already has the video file, pass its path instead of the URL to both.
- `--fix-errors` on `format_transcript.py` lets the model correct misheard names and terms. Default is punctuation-only. Chunks where the model changed too much automatically keep the original wording, and the script lists them.

## Step 3 — Report

Tell the user, in plain language, which files were created and where (folder + names), how many slides were inserted, and any chunks that fell back to basic punctuation. Keep it to a few lines. Don't restate the transcript.

## Tuning and troubleshooting

See `references/troubleshooting.md` for: YouTube 403 errors, too many or too few slides, slide builds counted as separate slides, misplaced slides, PDF export problems, low-memory Macs, and running Ollama.
