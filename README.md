# youtube-talk-to-pdf

Turns a recorded talk (YouTube URL or local video) into:

- a transcript, with and without timestamps, plus an `.srt` subtitle file
- a clean Markdown version of any time range, with punctuation, paragraphs and time headings
- one image per presentation slide, plus the video trimmed to that range
- a PDF with the slides placed in the text where they were shown

## Install
In Claude Code (v2.1.275 or later), inside a session:

```
/plugin install youtube-talk-to-pdf --marketplace michael-han-il/youtube-talk-to-pdf
```

Or from your shell:

```
claude plugin marketplace add michael-han-il/youtube-talk-to-pdf
claude plugin install youtube-talk-to-pdf@michael-han-il
```

Updates: `claude plugin update youtube-talk-to-pdf@michael-han-il`.

## Why use it
It works best on conference recordings, lectures, keynotes and other events where the speaker presents slides on a projector or screen. In those videos, much of the content is on the slides rather than in what's said.

YouTube's own transcript is text only. Auto-generated captions often have little or no punctuation, and they leave out everything shown on screen. This plugin adds:

- **Slide detection:** finds the shots where the camera shows the projected slides rather than the speaker, splits them when the presenter clicks to a new slide, and keeps one clear image of each slide
- **Slides placed in the text:** each slide goes beside the part of the talk where it was shown, between sentences rather than mid-sentence
- **A readable PDF:** punctuated paragraphs, time headings to find each passage in the video, and the slides included, ready to read, print or share

## Extra Chinese language support
Works in any language Whisper supports, with additional handling for Chinese talks:

- **Traditional or Simplified** output, normalised with OpenCC so the result doesn't mix the two
- **Taiwan (default), Hong Kong or classic OpenCC** character standards for Traditional (e.g. 吃/為 vs 喫/爲)
- **Full-width Chinese punctuation** and paragraphing using Chinese-language formatting instructions
- **Mandarin and Cantonese** prompts for transcription, and an option to fix misheard names and terms

Everything runs locally with `uv` scripts. Models download on first use: about 3 GB for Whisper large-v3, 4 GB for the formatting model, and 0.6 GB for CLIP.

## Requirements
- `uv`, `ffmpeg`, Deno (for YouTube downloads), and Chrome/Edge/Brave (for the PDF export)
- **Apple Silicon Mac**: everything on the GPU (mlx-whisper, mlx-lm, PyTorch MPS). Best experience.
- **Intel Mac, Windows, Linux**: faster-whisper for transcription (CUDA if available, otherwise CPU); [Ollama](https://ollama.com) with `qwen2.5:7b` for punctuation (optional; falls back to basic punctuation); CLIP on CPU/CUDA.

## Using it
Ask Claude something like *"Make a PDF of this lecture with the slides: <URL>, from 40:10 to 1:46:55, it's in Chinese, Traditional characters."* The skill gathers what's missing, runs (or hands you) the four commands, and pauses so you can review the extracted slides before they go into the PDF.

## Please use responsibly
The output copies a speaker's talk and slides. Use it for personal study, accessibility, or content you have rights to, and respect the video's licence and YouTube's Terms of Service.

## License
[MIT](LICENSE) © 2026 Mike Han. Issues and suggestions: <https://github.com/michael-han-il/youtube-talk-to-pdf/issues>
