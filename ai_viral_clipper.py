"""
AI Viral Clipper Script
=======================

This script downloads a YouTube video or processes a local video file, transcribes
it with OpenAI Whisper, detects the three most engaging highlights with GPT, and
outputs vertically formatted clips suitable for TikTok/Reels.

Installation Instructions
-------------------------
Install the required dependencies before running the script:

    pip install openai moviepy yt-dlp torch numpy pillow

FFmpeg is required by MoviePy/yt-dlp. Install it via your package manager (e.g.,
`sudo apt install ffmpeg` on Debian/Ubuntu).

Usage
-----
Run from the terminal:

    python ai_viral_clipper.py --url <YouTubeURL or /path/to/video>

The script will create a `clips/` directory with three highlight clips.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

from moviepy.editor import (  # type: ignore
    CompositeVideoClip,
    VideoFileClip,
)
from moviepy.video.VideoClip import ImageClip
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from yt_dlp import YoutubeDL


@dataclass
class TranscriptSegment:
    """Represents a single whisper transcript segment."""

    start: float
    end: float
    text: str


@dataclass
class Highlight:
    """Information about a highlight returned by GPT."""

    start: float
    end: float
    reason: str
    title: str


def download_video(source: str, workdir: str) -> str:
    """Download the video from YouTube or copy a local file into workdir."""

    # If the provided source is a local file, copy it into the working directory.
    if os.path.isfile(source):
        destination = os.path.join(workdir, os.path.basename(source))
        shutil.copy(source, destination)
        return destination

    # Otherwise, assume a URL and download using yt-dlp.
    ydl_opts = {
        "outtmpl": os.path.join(workdir, "input.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        return ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp4"


def transcribe_video(client: OpenAI, video_path: str) -> Tuple[List[TranscriptSegment], Dict]:
    """Transcribe video using OpenAI Whisper and return segments and raw response."""

    with open(video_path, "rb") as fh:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=fh,
            response_format="verbose_json",
        )

    segments = [
        TranscriptSegment(start=float(seg["start"]), end=float(seg["end"]), text=seg["text"].strip())
        for seg in response["segments"]
    ]
    return segments, response


def format_segments_for_prompt(segments: List[TranscriptSegment]) -> str:
    """Create a concise transcript string for GPT prompt."""

    lines = []
    for seg in segments:
        start = seg.start
        minutes = int(start // 60)
        seconds = start % 60
        timestamp = f"{minutes:02d}:{seconds:05.2f}"
        lines.append(f"[{timestamp}] {seg.text}")
    return "\n".join(lines)


def detect_highlights(client: OpenAI, transcript: str) -> List[Highlight]:
    """Use GPT to detect three engaging highlights from transcript."""

    format_instruction = (
        """Output strictly JSON: {"highlights":[{"start":number,"end":number,"title":string,"reason":string}, ...]}"""
    )

    system_prompt = (
        "You are an editing assistant. Read a transcript with timestamps and select "
        "three non-overlapping, engaging highlights (funny, emotional, shocking, etc.). "
        "Each highlight must be 20-60 seconds long. Return JSON with keys 'start', 'end', "
        "'reason', and 'title'. Timestamps should be in seconds. Titles must be short and viral. "
        f"{format_instruction}"
    )

    user_prompt = (
        "Transcript with timestamps:```\n"
        f"{transcript}\n"
        "```\nRespond with a JSON list of three highlight objects."
    )

    response = client.responses.create(
        model="gpt-4o-mini",
        temperature=0.5,
        response_format={"type": "json_object"},
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    if hasattr(response, "output_text"):
        content = response.output_text
    elif response.output:
        content = response.output[0].content[0].text
    else:
        content = "{}"

    data = json.loads(content)

    if isinstance(data, dict) and "highlights" in data:
        items = data["highlights"]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    highlights = []
    for item in items:
        start = float(item["start"])
        end = float(item["end"])
        reason = item["reason"]
        title = item["title"]
        highlights.append(Highlight(start=start, end=end, reason=reason, title=title))
    return highlights[:3]


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def build_caption_images(
    segments: List[TranscriptSegment],
    start: float,
    end: float,
    video_size: Tuple[int, int],
    font_path: str = "",
) -> List[ImageClip]:
    """Create caption ImageClips positioned at the bottom based on segments."""

    caption_clips = []
    width, height = video_size
    padding = 40
    box_width = int(width * 0.9)
    font_size = max(36, width // 18)

    # Attempt to use a default system font; fallback to PIL default if not available.
    try:
        font = ImageFont.truetype(font_path or "DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Collect segments overlapping with the highlight window.
    filtered: List[TranscriptSegment] = []
    for seg in segments:
        if seg.end <= start or seg.start >= end:
            continue
        filtered.append(
            TranscriptSegment(start=max(seg.start, start), end=min(seg.end, end), text=seg.text)
        )

    if not filtered:
        return []

    groups: List[Tuple[float, float, str]] = []
    current_text: List[str] = []
    group_start = filtered[0].start
    last_end = filtered[0].start

    for seg in filtered:
        if current_text and seg.start - last_end > 1.2:
            groups.append((group_start, last_end, " ".join(current_text)))
            current_text = []
            group_start = seg.start

        current_text.append(seg.text)
        last_end = seg.end

    if current_text:
        groups.append((group_start, last_end, " ".join(current_text)))

    for g_start, g_end, text in groups:
        duration = max(0.5, g_end - g_start)
        clip = create_caption_clip(text.strip(), duration, box_width, font, width, height, padding)
        offset = max(0.0, g_start - start)
        caption_clips.append(clip.set_start(offset))

    return caption_clips


def create_caption_clip(
    text: str,
    duration: float,
    box_width: int,
    font: ImageFont.FreeTypeFont,
    video_width: int,
    video_height: int,
    padding: int,
) -> ImageClip:
    """Create a caption clip with semi-transparent background using PIL."""

    lines = wrap_text(text, font, box_width)
    line_height = calculate_line_height(font)
    text_height = line_height * len(lines)
    box_height = text_height + padding

    img = Image.new("RGBA", (box_width + padding, box_height), (0, 0, 0, 160))
    draw = ImageDraw.Draw(img)

    y_text = padding // 2
    for line in lines:
        line_width = measure_text_width(font, line)
        x_text = (img.width - line_width) / 2
        draw.text((x_text, y_text), line, font=font, fill=(255, 255, 255, 255))
        y_text += line_height

    caption = ImageClip(np.array(img))
    caption = caption.set_duration(duration)
    caption = caption.set_position(("center", video_height - box_height - 100))
    return caption


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Wrap text to fit within max_width using the provided font."""

    words = text.split()
    lines: List[str] = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        if measure_text_width(font, test_line) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def measure_text_width(font: ImageFont.FreeTypeFont, text: str) -> float:
    """Measure text width with compatibility across Pillow font types."""

    if hasattr(font, "getlength"):
        return float(font.getlength(text))
    return float(font.getsize(text)[0])


def calculate_line_height(font: ImageFont.FreeTypeFont) -> int:
    """Return a reasonable line height for the provided font."""

    if hasattr(font, "getbbox"):
        bbox = font.getbbox("Hg")
        return bbox[3] - bbox[1]
    return font.getsize("Hg")[1]


def create_title_clip(title: str, duration: float, video_width: int) -> ImageClip:
    """Create a bold title clip for the top of the video."""

    font_size = max(60, video_width // 12)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    padding = 40
    lines = wrap_text(title.upper(), font, int(video_width * 0.9))
    line_height = calculate_line_height(font)
    text_height = line_height * len(lines)
    box_height = text_height + padding

    img = Image.new("RGBA", (int(video_width * 0.95), box_height), (255, 255, 0, 220))
    draw = ImageDraw.Draw(img)
    y_text = padding // 2
    for line in lines:
        line_width = measure_text_width(font, line)
        x_text = (img.width - line_width) / 2
        draw.text((x_text, y_text), line, font=font, fill=(0, 0, 0, 255))
        y_text += line_height

    title_clip = ImageClip(np.array(img)).set_duration(duration)
    title_clip = title_clip.set_position(("center", 60))
    return title_clip


def generate_highlight_clips(
    video_path: str,
    highlights: List[Highlight],
    segments: List[TranscriptSegment],
    output_dir: str,
) -> None:
    """Create highlight clips with captions and titles."""

    ensure_directory(output_dir)

    with VideoFileClip(video_path) as base_clip:
        for idx, highlight in enumerate(highlights, start=1):
            start = max(0.0, highlight.start)
            end = min(base_clip.duration, highlight.end)
            if end <= start:
                continue

            # Enforce highlight duration constraints between 20-60 seconds when possible.
            min_duration = 20.0
            max_duration = 60.0
            if end - start < min_duration:
                end = min(base_clip.duration, start + min_duration)
            if end - start > max_duration:
                end = start + max_duration

            duration = end - start
            if duration <= 0:
                continue

            clip = base_clip.subclip(start, end)

            target_w, target_h = 1080, 1920
            clip = resize_to_vertical(clip, target_w, target_h)

            overlay_clips = build_caption_images(segments, start, end, (clip.w, clip.h))

            title_clip = create_title_clip(highlight.title, duration, clip.w).set_start(0)
            composite = CompositeVideoClip([clip, title_clip, *overlay_clips])
            composite = composite.set_audio(clip.audio)

            output_path = os.path.join(output_dir, f"clip_{idx}.mp4")
            composite.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=clip.fps,
                temp_audiofile=os.path.join(tempfile.gettempdir(), f"temp-audio-{idx}.m4a"),
                remove_temp=True,
                threads=4,
            )

            composite.close()
            clip.close()

            print(f"Highlight {idx}: {highlight.reason}\nTitle: {highlight.title}\nSaved: {output_path}\n")


def resize_to_vertical(clip: VideoFileClip, target_w: int, target_h: int) -> VideoFileClip:
    """Resize and crop clip to vertical 9:16 while preserving content."""

    target_ratio = target_w / target_h
    clip_ratio = clip.w / clip.h

    if clip_ratio > target_ratio:
        scaled_clip = clip.resize(height=target_h)
        extra_width = scaled_clip.w - target_w
        x_center = scaled_clip.w / 2
        cropped = scaled_clip.crop(x_center=x_center, y_center=scaled_clip.h / 2, width=target_w, height=target_h)
    else:
        scaled_clip = clip.resize(width=target_w)
        extra_height = scaled_clip.h - target_h
        y_center = scaled_clip.h / 2
        cropped = scaled_clip.crop(x_center=scaled_clip.w / 2, y_center=y_center, width=target_w, height=target_h)

    return cropped


def main() -> None:
    """Entry point for the script."""

    parser = argparse.ArgumentParser(description="AI Viral Video Clipper")
    parser.add_argument("--url", required=True, help="YouTube URL or local video path")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI()

    with tempfile.TemporaryDirectory() as workdir:
        video_path = download_video(args.url, workdir)
        print(f"Video prepared at {video_path}")

        segments, whisper_response = transcribe_video(client, video_path)
        print("Transcription completed.")

        transcript_for_prompt = format_segments_for_prompt(segments)
        highlights = detect_highlights(client, transcript_for_prompt)

        if len(highlights) < 3:
            print("Warning: Less than three highlights detected. Processing available ones.")

        output_dir = os.path.join(os.getcwd(), "clips")
        generate_highlight_clips(video_path, highlights, segments, output_dir)

        transcription_path = os.path.join(output_dir, "transcription.json")
        ensure_directory(output_dir)
        with open(transcription_path, "w", encoding="utf-8") as fh:
            json.dump(whisper_response, fh, indent=2)
        print(f"Verbose transcription saved to {transcription_path}")


if __name__ == "__main__":
    main()
