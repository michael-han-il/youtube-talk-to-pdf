# Troubleshooting and tuning

## YouTube download fails (HTTP 403, "Sign in to confirm", no formats)
1. Install Deno (yt-dlp needs a JavaScript runtime for YouTube): `brew install deno` / `winget install DenoLand.Deno`.
2. Refresh yt-dlp: rerun with `uv run --refresh <script>.py …`.
3. Use the user's own browser login: add `--cookies-from-browser chrome` (or `safari`, `firefox`, `edge`, `brave`). macOS may prompt once for keychain access.
4. Last resort: the user downloads the video another way and passes the file path to `transcribe.py` and `extract_slides.py`.

## Transcription
- Wrong language detected: pass `--language` explicitly.
- Repeated lines or "Thank you for watching" hallucinations: already reduced (`condition_on_previous_text=False`, duplicate removal, VAD on faster-whisper). Re-run with `--language` set if it persists.
- Chinese mixes Traditional/Simplified or uses 喫/爲: output is normalised with OpenCC. `--variant tw` (default) gives 吃/為, `hk` gives Hong Kong forms, and `--script simp` gives Simplified.
- Slow on CPU: `--fast` (large-v3-turbo). This is good for English but noticeably weaker for Chinese, Japanese and Korean.

## Formatting (format_transcript.py)
- "N chunks kept original wording": the model rewrote too much, so those chunks have basic punctuation. Try a bigger model (`--model mlx-community/Qwen2.5-14B-Instruct-4bit`, or `--backend ollama --model qwen2.5:14b`), or accept as is.
- 8 GB Apple Silicon Macs: `--model mlx-community/Qwen2.5-3B-Instruct-4bit`.
- Not Apple Silicon: install Ollama, start it, `ollama pull qwen2.5:7b`. With `--backend auto` it is used automatically when running. Without it, basic punctuation is used.
- Check the range before a long run: `--dry-run`.

## Slides (extract_slides.py)
| Problem | Fix |
|---|---|
| Slides missed | `--overwrite --threshold 0.35` (lower = keep more) |
| Speaker/audience shots kept | `--overwrite --threshold 0.65`, or just delete them in review |
| Consecutive slides merged into one | `--overwrite --slide-change 5` |
| One slide with bullet builds saved many times | `--overwrite --slide-change 12` |
| Same slide shown again later is skipped | intended (first appearance only); `--dedup 0` keeps repeats |
| Camera cuts not detected | `--cut-sensitivity 20` |
| Slide text hard to read | `--height 1080` |

`slides/_slides.json` lists each saved slide's time, the shot it came from and its slide score, which helps when tuning `--threshold`.

The trimmed video is saved by default. `--no-trimmed-video` skips it, and `--keep-full-video` also keeps the full download.

## Slide placement (insert_slides.py)
- Slides are placed before the sentence being spoken when they appeared, or between paragraphs when that's within ~60 characters. They never go mid-sentence.
- "Could not match the transcript to the Markdown": the `--transcript` file isn't the one the Markdown was made from, or `--fix-errors` changed too much. Use the same `_timestamped.txt` that `format_transcript.py` read.
- A slide is clearly in the wrong place: rename its file to a different time (e.g. `0052-14.jpg` → `0053-40.jpg`) and rerun step 4. Re-running is cheap.
- Caption word: `--label "Slide"`. Larger print: `--font-size 15`.

## PDF export
- Uses Chrome, Edge, Brave or Chromium headless. If none is found, the script leaves an `.html` next to the `.md`. Open it in any browser and use Print → Save as PDF.
- Images missing in the PDF: keep `slides/` next to the `.md`/`.html`, and don't move the HTML before exporting.
