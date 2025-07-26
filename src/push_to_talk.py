import os
import sys
import time
import logging
import threading
import signal
import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import json

from src.audio_recorder import AudioRecorder
from src.transcription import Transcriber
from src.text_refiner import TextRefiner
from src.text_inserter import TextInserter
from src.hotkey_service import HotkeyService
from src.audio_feedback import play_start_feedback, play_stop_feedback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('push_to_talk.log')
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class PushToTalkConfig:
    """Configuration class for PushToTalk application."""
    # OpenAI settings
    openai_api_key: str = ""
    whisper_model: str = "gpt-4o-transcribe"
    gpt_model: str = "gpt-4.1-nano"
    
    # Audio settings
    sample_rate: int = 24000  # 24kHz required for streaming transcription
    chunk_size: int = 1024
    channels: int = 1
    
    # Hotkey settings
    hotkey: str = "ctrl+shift+space"
    toggle_hotkey: str = "ctrl+shift+t"  # New toggle hotkey
    
    # Text insertion settings
    insertion_method: str = "clipboard"  # "clipboard" or "sendkeys"
    insertion_delay: float = 0.005
    
    # Feature flags
    enable_text_refinement: bool = True
    enable_logging: bool = True
    enable_audio_feedback: bool = True
    enable_streaming_transcription: bool = True  # Enable streaming transcription by default
    
    # Streaming transcription settings
    streaming_model: str = "gpt-4o-transcribe"  # Model for streaming (gpt-4o-transcribe or gpt-4o-mini-transcribe)
    streaming_language: Optional[str] = None  # Language for streaming transcription (None = auto-detect)
    fallback_to_file_transcription: bool = True  # Fallback to file-based transcription if streaming fails
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PushToTalkConfig':
        """Load configuration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.warning(f"Failed to load config from {filepath}: {e}")
            return cls()

class PushToTalkApp:
    def __init__(self, config: Optional[PushToTalkConfig] = None):
        """
        Initialize the PushToTalk application.
        
        Args:
            config: Configuration object. If None, default config is used.
        """
        self.config = config or PushToTalkConfig()
        
        # Validate OpenAI API key
        if not self.config.openai_api_key:
            self.config.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide in config.")
        
        # Initialize components
        self.audio_recorder = AudioRecorder(
            sample_rate=self.config.sample_rate,
            chunk_size=self.config.chunk_size,
            channels=self.config.channels
        )
        
        self.transcriber = Transcriber(
            api_key=self.config.openai_api_key,
            model=self.config.whisper_model
        )
        
        self.text_refiner = TextRefiner(
            api_key=self.config.openai_api_key,
            model=self.config.gpt_model
        ) if self.config.enable_text_refinement else None
        
        self.text_inserter = TextInserter(
            insertion_delay=self.config.insertion_delay
        )
        
        self.hotkey_service = HotkeyService(
            hotkey=self.config.hotkey,
            toggle_hotkey=self.config.toggle_hotkey
        )
        
        # Audio feedback will be used directly via function calls when enabled
        
        # State management
        self.is_running = False
        self.processing_lock = threading.Lock()
        
        # Streaming transcription state
        self.streaming_websocket = None
        self.streaming_task = None
        self.current_transcription = ""
        self.is_streaming_active = False
        
        # Setup hotkey callbacks
        self.hotkey_service.set_callbacks(
            on_start_recording=self._on_start_recording,
            on_stop_recording=self._on_stop_recording
        )
        
        logger.info("PushToTalk application initialized")
    
    def start(self):
        """Start the PushToTalk application."""
        if self.is_running:
            logger.warning("Application is already running")
            return
        
        logger.info("Starting PushToTalk application...")
        
        # Start hotkey service
        if not self.hotkey_service.start_service():
            logger.error("Failed to start hotkey service")
            return
        
        self.is_running = True
        logger.info(f"PushToTalk is running.")
        logger.info(f"Push-to-talk: Press and hold '{self.config.hotkey}' to record.")
        logger.info(f"Toggle mode: Press '{self.config.toggle_hotkey}' to start/stop recording.")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def stop(self):
        """Stop the PushToTalk application."""
        if not self.is_running:
            logger.warning("Application is not running")
            return
        
        logger.info("Stopping PushToTalk application...")
        
        self.is_running = False
        self.hotkey_service.stop_service()
        
        # No cleanup needed for audio feedback utility functions
        
        logger.info("PushToTalk application stopped")
    
    def run(self):
        """Run the application until stopped."""
        self.start()
        
        try:
            # Keep the main thread alive
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _on_start_recording(self):
        """Callback for when recording starts."""
        with self.processing_lock:
            # Play audio feedback if enabled
            if self.config.enable_audio_feedback:
                play_start_feedback()
            
            # Start streaming transcription if enabled
            if self.config.enable_streaming_transcription:
                self._start_streaming_transcription()
                
                # Start recording with streaming callback
                if not self.audio_recorder.start_recording(streaming_callback=self._on_audio_chunk):
                    logger.error("Failed to start audio recording")
                    return
            else:
                # Start regular recording without streaming
                if not self.audio_recorder.start_recording():
                    logger.error("Failed to start audio recording")
                    return
    
    def _on_stop_recording(self):
        """Callback for when recording stops."""
        # Play audio feedback immediately when hotkey is released
        if self.config.enable_audio_feedback:
            play_stop_feedback()
        
        def process_recording():
            with self.processing_lock:
                if self.config.enable_streaming_transcription:
                    self._process_streaming_recording()
                else:
                    self._process_recorded_audio()
        
        # Process in a separate thread to avoid blocking the hotkey service
        processing_thread = threading.Thread(target=process_recording, daemon=True)
        processing_thread.start()
    
    def _start_streaming_transcription(self):
        """Start streaming transcription session."""
        try:
            self.current_transcription = ""
            self.is_streaming_active = True
            
            # Start the transcriber's streaming session
            success = self.transcriber.start_streaming_transcription(
                callback=self._on_transcription_update,
                model=self.config.streaming_model,
                language=self.config.streaming_language
            )
            
            if success:
                # Start the WebSocket connection in a background task
                self.streaming_task = threading.Thread(target=self._run_streaming_connection, daemon=True)
                self.streaming_task.start()
                logger.info("Streaming transcription started")
            else:
                logger.error("Failed to start streaming transcription")
                self.is_streaming_active = False
                
        except Exception as e:
            logger.error(f"Error starting streaming transcription: {e}")
            self.is_streaming_active = False
    
    def _run_streaming_connection(self):
        """Run the WebSocket connection in a separate thread."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the WebSocket connection
            loop.run_until_complete(self.transcriber._setup_streaming_connection())
            
        except Exception as e:
            logger.error(f"Error in streaming connection: {e}")
        finally:
            self.is_streaming_active = False
    
    def _on_audio_chunk(self, audio_data: bytes):
        """Handle audio chunk for streaming transcription."""
        if self.is_streaming_active:
            try:
                # Send audio chunk to transcriber's streaming queue
                self.transcriber.add_audio_chunk(audio_data)
            except Exception as e:
                logger.error(f"Error sending audio chunk: {e}")
    
    def _on_transcription_update(self, text: str, is_final: bool):
        """Handle transcription updates from streaming."""
        if is_final:
            # Final transcription segment
            if text.strip():
                self.current_transcription += " " + text.strip() if self.current_transcription else text.strip()
                logger.info(f"Streaming transcription update: {text}")
        else:
            # Partial transcription (real-time feedback)
            logger.debug(f"Partial transcription: {text}")
    
    def _process_streaming_recording(self):
        """Process the recorded audio using streaming transcription results."""
        try:
            # Stop recording and get audio file (for fallback if needed)
            logger.info("Processing streaming recording...")
            audio_file = self.audio_recorder.stop_recording()
            
            # Stop streaming transcription and get final result
            final_transcription = self.transcriber.stop_streaming_transcription()
            
            # Use accumulated streaming transcription or fallback to file-based
            transcribed_text = None
            if final_transcription and final_transcription.strip():
                transcribed_text = final_transcription.strip()
                logger.info(f"Using streaming transcription: {transcribed_text}")
            elif self.config.fallback_to_file_transcription and audio_file:
                logger.info("Falling back to file-based transcription...")
                transcribed_text = self.transcriber.transcribe_audio(audio_file)
                if transcribed_text:
                    logger.info(f"Fallback transcription: {transcribed_text}")
            
            if not transcribed_text:
                logger.warning("No transcription available from streaming or fallback")
                return
            
            # Get active window info for logging
            window_title = self.text_inserter.get_active_window_title()
            if window_title:
                logger.info(f"Target window: {window_title}")
            
            # Refine text if enabled (if failed, fallback to the original transcription)
            final_text = transcribed_text
            if self.text_refiner and self.config.enable_text_refinement:
                logger.info("Refining transcribed text...")
                refined_text = self.text_refiner.refine_text(transcribed_text)
                if refined_text:
                    final_text = refined_text
                    logger.info(f"Refined: {final_text}")
            
            # Insert text into active window
            logger.info("Inserting text into active window...")
            success = self.text_inserter.insert_text(
                final_text, 
                method=self.config.insertion_method
            )
            
            if success:
                logger.info("Text insertion successful")
            else:
                logger.error("Text insertion failed")
                
        except Exception as e:
            logger.error(f"Error processing streaming recording: {e}")
        finally:
            # Reset streaming state
            self.is_streaming_active = False
            self.current_transcription = ""
    
    def _process_recorded_audio(self):
        """Process the recorded audio through the full pipeline."""
        try:
            # Stop recording and get audio file
            logger.info("Processing recorded audio...")
            audio_file = self.audio_recorder.stop_recording()
            
            if not audio_file:
                logger.warning("No audio file to process")
                return
            
            # Get active window info for logging
            window_title = self.text_inserter.get_active_window_title()
            if window_title:
                logger.info(f"Target window: {window_title}")
            
            # Transcribe audio
            logger.info("Transcribing audio...")
            transcribed_text = self.transcriber.transcribe_audio(audio_file)
            
            if not transcribed_text:
                logger.warning("Transcription failed or returned empty text")
                return
            
            logger.info(f"Transcribed: {transcribed_text}")
            
            # Refine text if enabled (if failed, fallback to the original transcription)
            final_text = transcribed_text
            if self.text_refiner and self.config.enable_text_refinement:
                logger.info("Refining transcribed text...")
                refined_text = self.text_refiner.refine_text(transcribed_text)
                if refined_text:
                    final_text = refined_text
                    logger.info(f"Refined: {final_text}")
            
            # Insert text into active window
            logger.info("Inserting text into active window...")
            success = self.text_inserter.insert_text(
                final_text, 
                method=self.config.insertion_method
            )
            
            if success:
                logger.info("Text insertion successful")
            else:
                logger.error("Text insertion failed")
                
        except Exception as e:
            logger.error(f"Error processing recorded audio: {e}")
    
    def change_hotkey(self, new_hotkey: str) -> bool:
        """
        Change the push-to-talk hotkey combination.
        
        Args:
            new_hotkey: New push-to-talk hotkey combination
            
        Returns:
            True if hotkey was changed successfully
        """
        if self.hotkey_service.change_hotkey(new_hotkey):
            self.config.hotkey = new_hotkey
            return True
        return False
    
    def change_toggle_hotkey(self, new_toggle_hotkey: str) -> bool:
        """
        Change the toggle hotkey combination.
        
        Args:
            new_toggle_hotkey: New toggle hotkey combination
            
        Returns:
            True if toggle hotkey was changed successfully
        """
        if self.hotkey_service.change_toggle_hotkey(new_toggle_hotkey):
            self.config.toggle_hotkey = new_toggle_hotkey
            return True
        return False
    
    def toggle_text_refinement(self) -> bool:
        """
        Toggle text refinement on/off.
        
        Returns:
            New state of text refinement (True if enabled)
        """
        self.config.enable_text_refinement = not self.config.enable_text_refinement
        logger.info(f"Text refinement {'enabled' if self.config.enable_text_refinement else 'disabled'}")
        return self.config.enable_text_refinement
    
    def toggle_audio_feedback(self) -> bool:
        """
        Toggle audio feedback on/off.
        
        Returns:
            New state of audio feedback (True if enabled)
        """
        self.config.enable_audio_feedback = not self.config.enable_audio_feedback
        
        # Audio feedback is now handled via utility functions - no service to manage
        
        logger.info(f"Audio feedback {'enabled' if self.config.enable_audio_feedback else 'disabled'}")
        return self.config.enable_audio_feedback
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current application status.
        
        Returns:
            Dictionary containing status information
        """
        return {
            "is_running": self.is_running,
            "hotkey": self.config.hotkey,
            "toggle_hotkey": self.config.toggle_hotkey,
            "recording_mode": self.hotkey_service.get_recording_mode(),
            "is_recording": self.hotkey_service.is_recording,
            "is_toggle_recording": self.hotkey_service.is_toggle_recording(),
            "text_refinement_enabled": self.config.enable_text_refinement,
            "audio_feedback_enabled": self.config.enable_audio_feedback,
            "streaming_transcription_enabled": self.config.enable_streaming_transcription,
            "is_streaming_active": self.is_streaming_active,
            "fallback_to_file_transcription": self.config.fallback_to_file_transcription,
            "insertion_method": self.config.insertion_method,
            "hotkey_service_running": self.hotkey_service.is_service_running(),
            "models": {
                "whisper": self.config.whisper_model,
                "gpt": self.config.gpt_model,
                "streaming": self.config.streaming_model
            }
        }
    
    def toggle_streaming_transcription(self) -> bool:
        """
        Toggle streaming transcription on/off.
        
        Returns:
            New state of streaming transcription (True if enabled)
        """
        self.config.enable_streaming_transcription = not self.config.enable_streaming_transcription
        logger.info(f"Streaming transcription {'enabled' if self.config.enable_streaming_transcription else 'disabled'}")
        return self.config.enable_streaming_transcription

def main():
    """Main entry point for the application."""
    # Load config if it exists
    config_file = "push_to_talk_config.json"
    if os.path.exists(config_file):
        config = PushToTalkConfig.load_from_file(config_file)
        logger.info(f"Loaded configuration from {config_file}")
    else:
        config = PushToTalkConfig()
        config.save_to_file(config_file)
        logger.info(f"Created default configuration file: {config_file}")
    
    # Create and run application
    try:
        app = PushToTalkApp(config)
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 