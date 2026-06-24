import gradio as gr
import whisperx
import torchaudio
import subprocess
import tempfile
import os
import requests
import uuid
import torch
import sys
import re
from urllib.parse import urlparse

# ========== Fix PyTorch 2.6 weights_only issue ==========
try:
    from torch._weights_only_unpickler import WeightsUnpickler
    _original_find_class = WeightsUnpickler.find_class

    def _patched_find_class(self, module, name):
        try:
            return _original_find_class(self, module, name)
        except Exception:
            import importlib
            mod = importlib.import_module(module)
            return getattr(mod, name)

    WeightsUnpickler.find_class = _patched_find_class
except ImportError:
    pass

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

import lightning_fabric.utilities.cloud_io as cloud_io

def _patched_pl_load(path_or_url, map_location=None, weights_only=None):
    return torch.load(path_or_url, map_location=map_location, weights_only=False)

cloud_io._load = _patched_pl_load

import pytorch_lightning.core.saving as pl_saving
_original_load_from_checkpoint = pl_saving._load_from_checkpoint

def _patched_load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load_from_checkpoint(cls, checkpoint_path, *args, **kwargs)

pl_saving._load_from_checkpoint = _patched_load_from_checkpoint

# ========== Global Settings ==========
device = "cpu"

# ========== Load Models ==========
print("Loading Whisper model for Arabic transcription...")
whisper_model = whisperx.load_model(
    "large-v2",
    device,
    compute_type="int8",
    language="ar"
)
print("Whisper model loaded successfully")

print("Loading Arabic alignment model...")
align_model, metadata = whisperx.load_align_model(
    language_code="ar",
    device=device
)
print("Alignment model loaded successfully")

# ========== Helper Functions ==========

def get_filename_from_url(url):
    parsed = urlparse(url)
    return os.path.basename(parsed.path)

def detect_file_type(filepath):
    """Detect if file is audio/video or HTML/etc by reading header."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(32)

        if header.startswith(b'\x00\x00\x00 ftyp') or header.startswith(b'\x00\x00\x00\x20ftyp'):
            return "MP4/M4A"
        elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3') or header.startswith(b'\xff\xf2'):
            return "MP3"
        elif header.startswith(b'RIFF'):
            return "WAV/RIFF"
        elif header.startswith(b'\x1aE\xdf\xa3'):
            return "MKV/WebM"
        elif header.startswith(b'OggS'):
            return "OGG"
        elif header.startswith(b'fLaC'):
            return "FLAC"
        elif b'<html' in header.lower() or b'<!doctype' in header.lower():
            return "HTML (not audio/video)"
        elif b'<?xml' in header:
            return "XML (not audio/video)"
        elif b'{\"' in header or b'{' in header[:5]:
            return "JSON (not audio/video)"
        else:
            return f"Unknown (header: {header[:8].hex()})"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def is_video_file(filepath):
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ts', '.m2ts'}
    return os.path.splitext(filepath)[1].lower() in video_extensions

def is_audio_file(filepath):
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus', '.aiff', '.au'}
    return os.path.splitext(filepath)[1].lower() in audio_extensions

def extract_audio_from_video(video_path, output_dir):
    """Extract audio from video using FFmpeg."""
    output_file = os.path.join(output_dir, f"extracted_audio_{uuid.uuid4().hex[:8]}.wav")
    result = subprocess.run(
        [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-y",
            output_file
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown FFmpeg error"
        raise Exception(f"FFmpeg video extraction failed (exit {result.returncode}): {stderr[:500]}")
    return output_file

def convert_audio(input_file):
    """Convert any audio to WAV 16kHz mono."""
    output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    result = subprocess.run(
        [
            "ffmpeg",
            "-i", input_file,
            "-ar", "16000",
            "-ac", "1",
            "-y",
            output_file
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown FFmpeg error"
        if os.path.exists(input_file):
            file_size = os.path.getsize(input_file)
            file_type = detect_file_type(input_file)
            raise Exception(f"FFmpeg conversion failed (exit {result.returncode}). File size: {file_size} bytes, detected type: {file_type}. Error: {stderr[:500]}")
        else:
            raise Exception(f"FFmpeg conversion failed (exit {result.returncode}). Input file not found. Error: {stderr[:500]}")
    return output_file

def build_chunks(words_data, max_words=5, max_duration=2.5):
    chunks = []
    current_words = []
    chunk_start = None
    for word in words_data:
        if chunk_start is None:
            chunk_start = word["start"]
        current_words.append(word)
        duration = word["end"] - chunk_start
        if len(current_words) >= max_words or duration >= max_duration:
            chunks.append({
                "arabic": " ".join(w["word"] for w in current_words),
                "start": round(current_words[0]["start"], 3),
                "end": round(current_words[-1]["end"], 3)
            })
            current_words = []
            chunk_start = None
    if current_words:
        chunks.append({
            "arabic": " ".join(w["word"] for w in current_words),
            "start": round(current_words[0]["start"], 3),
            "end": round(current_words[-1]["end"], 3)
        })
    return chunks

def perform_transcription(audio_path):
    converted_file = None
    try:
        converted_file = convert_audio(audio_path)
        audio = whisperx.load_audio(converted_file)
        result = whisper_model.transcribe(audio, batch_size=1, language="ar")
        result_aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device
        )
        full_text = " ".join(segment["text"].strip() for segment in result_aligned["segments"])
        words = []
        for segment in result_aligned["segments"]:
            for word in segment.get("words", []):
                if not word.get("word"):
                    continue
                words.append({
                    "word": word["word"].strip(),
                    "start": round(float(word.get("start", 0)), 3),
                    "end": round(float(word.get("end", 0)), 3)
                })
        chunks = build_chunks(words_data=words, max_words=5, max_duration=2.5)
        waveform, sample_rate = torchaudio.load(converted_file)
        duration = waveform.shape[1] / sample_rate
        return {
            "mode": "transcription",
            "transcribed_text": full_text,
            "duration": round(duration, 3),
            "word_count": len(words),
            "chunk_count": len(chunks),
            "words": words,
            "chunks": chunks
        }
    finally:
        if converted_file and os.path.exists(converted_file):
            os.remove(converted_file)

def perform_alignment(audio_path, arabic_text):
    converted_file = None
    try:
        converted_file = convert_audio(audio_path)
        waveform, sample_rate = torchaudio.load(converted_file)
        duration = waveform.shape[1] / sample_rate
        audio = whisperx.load_audio(converted_file)
        segments = [{
            "start": 0.0,
            "end": duration,
            "text": arabic_text.strip()
        }]
        result = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device
        )
        words = []
        for segment in result["segments"]:
            for word in segment.get("words", []):
                if not word.get("word"):
                    continue
                words.append({
                    "word": word["word"].strip(),
                    "start": round(float(word.get("start", 0)), 3),
                    "end": round(float(word.get("end", 0)), 3)
                })
        chunks = build_chunks(words_data=words, max_words=5, max_duration=2.5)
        return {
            "mode": "alignment",
            "duration": round(duration, 3),
            "word_count": len(words),
            "chunk_count": len(chunks),
            "words": words,
            "chunks": chunks
        }
    finally:
        if converted_file and os.path.exists(converted_file):
            os.remove(converted_file)

# ========== Enhanced Download Functions ==========

def is_social_media_url(url):
    """Check if URL belongs to a platform best handled by yt-dlp."""
    social_domains = [
        'tiktok.com', 'youtube.com', 'youtu.be',
        'instagram.com', 'facebook.com', 'x.com', 'twitter.com',
        'vimeo.com', 'dailymotion.com', 'soundcloud.com'
    ]
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return any(domain in host for domain in social_domains)

def download_with_requests(url, output_dir):
    """Download using requests, raise exception if not a valid audio/video."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'audio/*, video/*, */*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': url,
    }
    session = requests.Session()
    response = session.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
    response.raise_for_status()

    # Extract filename from Content-Disposition or URL
    content_disp = response.headers.get('content-disposition', '')
    if 'filename=' in content_disp:
        filename = content_disp.split('filename=')[-1].strip('"')
    else:
        filename = get_filename_from_url(url)
        if not filename:
            # Generate from content-type
            content_type = response.headers.get('content-type', '')
            ext = '.mp3'
            if 'video' in content_type:
                ext = '.mp4'
            elif 'audio' in content_type:
                ext = '.mp3' if 'mpeg' in content_type else '.wav'
            filename = f"downloaded_{uuid.uuid4().hex[:8]}{ext}"

    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # Check file type – if HTML/XML/JSON, raise exception to trigger fallback
    file_type = detect_file_type(output_path)
    if "HTML" in file_type or "XML" in file_type or "JSON" in file_type:
        os.remove(output_path)
        raise Exception(f"Downloaded HTML/XML/JSON instead of media (detected: {file_type})")

    return output_path

def download_with_ytdlp(url, output_dir):
    """Download using yt-dlp, returns path to downloaded file."""
    import yt_dlp

    # Unique template to avoid collisions
    output_template = os.path.join(output_dir, '%(id)s.%(ext)s')

    ydl_opts = {
        'outtmpl': output_template,
        'format': 'bestaudio[ext=m4a]/bestaudio/best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': False,
        'nocheckcertificate': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'headers': {
            'Referer': url,
            'Origin': urlparse(url).netloc,
        },
        'extract_audio': True,   # ensures audio extraction when possible
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise Exception("yt-dlp could not extract any information from the URL")

        # The downloaded file should be named using the template with the video id
        video_id = info.get('id', 'unknown')
        ext = info.get('ext', 'mp4')
        possible_file = os.path.join(output_dir, f"{video_id}.{ext}")
        if os.path.exists(possible_file):
            return possible_file

        # Fallback: search for any file in the directory (should be the only one)
        files = os.listdir(output_dir)
        if files:
            return os.path.join(output_dir, files[0])
        else:
            raise Exception("yt-dlp did not produce any output file")

def download_file(url, output_dir):
    """
    Main download dispatcher.
    - For social media URLs, use yt-dlp directly (more reliable).
    - For others, try requests, fallback to yt-dlp on failure.
    """
    try:
        if is_social_media_url(url):
            return download_with_ytdlp(url, output_dir)
        else:
            # Try requests first
            return download_with_requests(url, output_dir)
    except Exception as req_err:
        # Fallback to yt-dlp
        print(f"Requests failed ({req_err}), falling back to yt-dlp...")
        return download_with_ytdlp(url, output_dir)

# ========== Main Processing Function ==========

def process_audio(audio_file, direct_url, arabic_text):
    downloaded_file = None
    extracted_audio = None
    temp_dir = None

    try:
        has_file = audio_file is not None and audio_file != ""
        has_url = direct_url is not None and direct_url.strip() != ""
        has_text = arabic_text is not None and arabic_text.strip() != ""

        if not has_file and not has_url:
            return {"error": "Please provide either an audio file or a direct download URL"}

        audio_path = None

        # Case 1: uploaded file
        if has_file:
            if not is_audio_file(audio_file):
                return {"error": f"Uploaded file is not a recognized audio format. Supported: mp3, wav, flac, aac, ogg, m4a, wma, opus"}
            audio_path = audio_file

        # Case 2: URL
        elif has_url:
            url = direct_url.strip()
            temp_dir = tempfile.mkdtemp()

            try:
                downloaded_file = download_file(url, temp_dir)
            except Exception as e:
                return {"error": f"Failed to download from URL: {str(e)}"}

            if not os.path.exists(downloaded_file):
                return {"error": "Download succeeded but file was not found on disk"}

            file_size = os.path.getsize(downloaded_file)
            if file_size == 0:
                return {"error": "Downloaded file is empty (0 bytes)"}

            file_type = detect_file_type(downloaded_file)
            if "HTML" in file_type or "XML" in file_type or "JSON" in file_type:
                return {"error": f"Downloaded file is not audio/video. Detected type: {file_type}. The URL may require authentication or be expired."}

            # Video: extract audio
            if is_video_file(downloaded_file):
                try:
                    extracted_audio = extract_audio_from_video(downloaded_file, temp_dir)
                    audio_path = extracted_audio
                except Exception as e:
                    return {"error": f"Video processing failed: {str(e)}"}

            # Audio: use directly
            elif is_audio_file(downloaded_file):
                audio_path = downloaded_file

            # Unknown: try anyway
            else:
                audio_path = downloaded_file

        if not audio_path or not os.path.exists(audio_path):
            return {"error": "Failed to prepare audio file for processing"}

        # Choose mode
        if has_text:
            result = perform_alignment(audio_path, arabic_text)
        else:
            result = perform_transcription(audio_path)

        return result

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

    finally:
        # Cleanup
        if extracted_audio and os.path.exists(extracted_audio):
            try:
                os.remove(extracted_audio)
            except:
                pass
        if downloaded_file and os.path.exists(downloaded_file):
            try:
                os.remove(downloaded_file)
            except:
                pass
        if temp_dir and os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except:
                pass

# ========== Gradio Interface ==========

with gr.Blocks(title="WhisperX Quran - Transcription & Alignment") as demo:
    gr.Markdown("""
    # 🎙️ WhisperX Quran - Transcription & Alignment
    ### Arabic Speech-to-Text with Word-Level Timestamps

    **Two Modes:**
    - 📝 **Transcription Mode** — Leave text empty → auto-extract Arabic text from audio
    - 🔗 **Alignment Mode** — Enter Arabic text → align it to audio timestamps

    **Input Sources:**
    - ✅ Direct audio file upload (MP3, WAV, FLAC, AAC, OGG, M4A, etc.)
    - ✅ Direct download URL for audio files
    - ✅ Direct download URL for video files (auto-extracts audio via FFmpeg)
    - ✅ Social media links (TikTok, YouTube, Instagram) via yt-dlp

    **⚠️ Important for TikTok:** Use the **page URL** (e.g., `https://www.tiktok.com/@user/video/123...`) not the direct download link. Direct TikTok links expire quickly.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="📁 Upload Audio File (Optional)"
            )
            url_input = gr.Textbox(
                label="🔗 URL (Optional)",
                placeholder="Direct audio/video URL or TikTok/YouTube page link",
                lines=1
            )
            text_input = gr.Textbox(
                label="📝 Arabic Text (Optional for Alignment)",
                placeholder="أدخل النص العربي هنا للموائمة... (اتركه فارغاً لاستخراج النص تلقائياً)",
                lines=10
            )
            submit_btn = gr.Button("▶️ Start Processing", variant="primary", size="lg")

        with gr.Column(scale=1):
            output = gr.JSON(
                label="📊 Results",
                value={"status": "Waiting for input..."}
            )

    gr.Markdown("""
    ---
    **💡 How to use:**

    **Mode 1 — Transcription (No text needed):**
    1. Upload an audio file OR paste a URL
    2. Leave the text box **empty**
    3. Click **Start Processing**
    4. The system will auto-transcribe the Arabic audio and return word timestamps

    **Mode 2 — Alignment (With text):**
    1. Upload an audio file OR paste a URL
    2. Enter the Arabic text to align
    3. Click **Start Processing**
    4. The system will align each word to its exact timestamp in the audio

    **Note:** If both file and URL are provided, the uploaded file takes priority.
    """)

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, url_input, text_input],
        outputs=output
    )

demo.launch()