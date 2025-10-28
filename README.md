# AI Viral Clipper

This repository provides a fully automated pipeline for turning long-form YouTube
videos (or local video files) into three viral-ready vertical clips tailored for
TikTok/Reels.

## Features

- Downloads videos via `yt-dlp`.
- Transcribes audio with OpenAI Whisper (verbose JSON output saved to `clips/transcription.json`).
- Uses GPT to select three 20-60 second highlights with reasons and viral titles.
- Automatically crops and resizes footage to 9:16 vertical format.
- Generates synced captions and bold titles for each clip.
- Outputs finished clips as `clips/clip_1.mp4`, `clips/clip_2.mp4`, and `clips/clip_3.mp4`.

## Installation

Install the Python dependencies:

```bash
pip install openai moviepy yt-dlp torch numpy pillow
```

Install FFmpeg if you do not already have it (Debian/Ubuntu example):

```bash
sudo apt install ffmpeg
```

Set your OpenAI API key in the environment:

```bash
export OPENAI_API_KEY="sk-..."
```

## Usage

### Quick start

1. Clone or download this repository and open a terminal in the project
   directory.
2. Install the dependencies listed above and make sure FFmpeg is available on
   your `PATH`.
3. Set the `OPENAI_API_KEY` environment variable so Whisper and GPT can run.
4. Launch the script with either a YouTube link or a local video file:

   ```bash
   # Example: YouTube link
   python ai_viral_clipper.py --url "https://www.youtube.com/watch?v=..."

   # Example: local file (pass the absolute or relative path to --url)
   python ai_viral_clipper.py --url /path/to/video.mp4
   ```

During processing, progress messages will be printed to the console. When the
script completes, the three highlight clips and the verbose Whisper
transcription will be saved in the `clips/` directory. The script also prints
the reason each highlight was chosen along with the generated title.

## Troubleshooting

- **`Can't list 'ai_viral_clipper.py'` when running `python -m compileall`:**
  Make sure you run the command from the same directory that contains the
  script. If you cloned the repository into `C:\Users\1sams\Aiclipping`, open a
  terminal in that folder (or `cd` into it) before executing the command:

  ```powershell
  cd C:\Users\1sams\Aiclipping
  python -m compileall ai_viral_clipper.py
  ```

- **Fonts not found for titles/captions:** Install a TrueType font such as
  DejaVu Sans and provide its path via the `DEJAVU_FONTS` environment variable or
  edit the script to point to a font installed on your system.
