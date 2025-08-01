"""
PushToTalk - A Speech-to-Text Push-to-Talk Application

A Python application that provides push-to-talk speech-to-text functionality
with OpenAI Whisper transcription, GPT text refinement, and automatic text
insertion into the active window across platforms.

Main Components:
- AudioRecorder: Records audio using pyaudio
- Transcriber: Converts speech to text using OpenAI Whisper
- TextRefiner: Improves transcription using Refinement Models
- TextInserter: Inserts text into active windows using cross-platform libraries
- HotkeyService: Handles push-to-talk hotkey detection
- PushToTalkApp: Main application orchestrator
"""

__version__ = "1.0.0"
__author__ = "Assistant"
__description__ = "Speech-to-Text Push-to-Talk Application"

from dotenv import load_dotenv
from .push_to_talk import PushToTalkApp, PushToTalkConfig, main
from .audio_recorder import AudioRecorder
from .transcription import Transcriber
from .text_refiner import TextRefiner
from .text_inserter import TextInserter
from .hotkey_service import HotkeyService

load_dotenv()

__all__ = [
    "PushToTalkApp",
    "PushToTalkConfig",
    "AudioRecorder",
    "Transcriber",
    "TextRefiner",
    "TextInserter",
    "HotkeyService",
    "main",
]
