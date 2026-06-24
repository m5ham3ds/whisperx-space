---
title: WhisperX Quran - Transcription & Alignment
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.0
python_version: '3.13'
app_file: app.py
pinned: false
short_description: Arabic Quran Alignment with Transcription & Word Timestamps
---

# 🎙️ WhisperX Quran - Transcription & Alignment

## Features
- 📝 **Transcription Mode** — Auto-extract Arabic text from audio (no text needed)
- 🔗 **Alignment Mode** — Align existing Arabic text to audio timestamps
- 📁 Direct audio file upload support
- 🔗 Direct download URL support (audio & video)
- 🎬 Auto video-to-audio extraction via FFmpeg
- ⏱️ Word-level timestamps
- 🧹 Automatic cleanup of temporary files

## Supported Formats
**Audio:** MP3, WAV, FLAC, AAC, OGG, M4A, WMA, OPUS, AIFF, AU
**Video:** MP4, AVI, MKV, MOV, WMV, FLV, WEBM, M4V, 3GP, TS, M2TS

## API Usage
```python
from gradio_client import Client

client = Client("qalam249/whisperx")

# Transcription mode (no text)
result = client.predict(
    audio_file="audio.mp3",
    direct_url=None,
    arabic_text="",
    api_name="/predict"
)

# Alignment mode (with text)
result = client.predict(
    audio_file="audio.mp3",
    direct_url=None,
    arabic_text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    api_name="/predict"
)
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
