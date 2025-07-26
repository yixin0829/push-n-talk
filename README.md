# Push-to-Talk Speech Transcription App

A sophisticated push-to-talk application that transcribes speech to text using OpenAI's Whisper API, with optional AI-powered text refinement using GPT models. The application now supports **real-time streaming transcription** for immediate speech-to-text conversion as you speak.

## 🚀 Features

- **Real-time Streaming Transcription**: Live speech-to-text using OpenAI's Realtime API with WebSockets
- **Push-to-Talk & Toggle Modes**: Flexible recording modes with customizable hotkeys
- **AI Text Refinement**: Optional GPT-powered text improvement and formatting
- **Multiple Insertion Methods**: Insert text via clipboard or direct keystroke simulation
- **Audio Feedback**: Customizable sound effects for recording events
- **Fallback Support**: Automatic fallback from streaming to file-based transcription
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Configurable**: Extensive configuration options for all aspects of the application

## 🆕 Streaming Transcription

The application now supports **real-time streaming transcription** using OpenAI's Realtime API, providing immediate speech-to-text conversion as you speak:

### How It Works

1. **Real-time Processing**: Audio is streamed directly to OpenAI's WebSocket endpoint while you speak
2. **Immediate Feedback**: Transcription appears in real-time with minimal latency
3. **Intelligent Fallback**: If streaming fails, automatically falls back to file-based transcription
4. **Voice Activity Detection**: Automatically detects when you start and stop speaking

### Configuration Options

```python
config = PushToTalkConfig()

# Enable streaming transcription (default: True)
config.enable_streaming_transcription = True

# Choose streaming model (faster vs more accurate)
config.streaming_model = "gpt-4o-transcribe"        # High accuracy
# config.streaming_model = "gpt-4o-mini-transcribe"  # Faster, lower latency

# Language detection (None = auto-detect)
config.streaming_language = None  # Auto-detect
# config.streaming_language = "en"  # Force English

# Fallback behavior
config.fallback_to_file_transcription = True  # Recommended for reliability
```

### Models Available

- **`gpt-4o-transcribe`**: High accuracy, standard latency (recommended)
- **`gpt-4o-mini-transcribe`**: Lower accuracy, minimal latency (for speed-critical applications)

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Microphone access
- Audio drivers (PyAudio)
