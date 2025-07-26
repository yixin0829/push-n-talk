#!/usr/bin/env python3
"""
PushToTalk - Example Usage

This script demonstrates different ways to use the PushToTalk application.
"""

import os
import sys
import time
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from push_to_talk import PushToTalkApp, PushToTalkConfig

def main():
    """Example usage of the PushToTalk application."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = PushToTalkConfig()
        
        # Customize configuration if needed
        config.hotkey = "ctrl+shift+space"  # Push-to-talk hotkey
        config.toggle_hotkey = "ctrl+shift+t"  # Toggle recording mode
        config.enable_text_refinement = True  # Enable AI text refinement
        config.enable_audio_feedback = True  # Enable audio feedback
        config.enable_streaming_transcription = True  # Enable real-time streaming transcription
        config.streaming_model = "gpt-4o-transcribe"  # Use the fast streaming model
        config.fallback_to_file_transcription = True  # Fallback to file-based if streaming fails
        
        # Optional: Set language for transcription (None = auto-detect)
        # config.streaming_language = "en"  # English
        
        logger.info("Starting PushToTalk application with streaming transcription...")
        logger.info(f"Streaming enabled: {config.enable_streaming_transcription}")
        logger.info(f"Streaming model: {config.streaming_model}")
        logger.info(f"Fallback enabled: {config.fallback_to_file_transcription}")
        
        # Create and start the application
        app = PushToTalkApp(config)
        
        logger.info("Application initialized successfully!")
        logger.info(f"Push-to-talk hotkey: {config.hotkey}")
        logger.info(f"Toggle recording hotkey: {config.toggle_hotkey}")
        logger.info("Press and hold the push-to-talk key to start recording with real-time transcription.")
        logger.info("Press Ctrl+C to quit.")
        
        # Display current status
        status = app.get_status()
        logger.info(f"Application status: {status}")
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Application stopped")

def streaming_transcription_demo():
    """Demonstrate streaming transcription capabilities."""
    
    logger = logging.getLogger(__name__)
    
    # Create a config optimized for streaming
    config = PushToTalkConfig()
    config.enable_streaming_transcription = True
    config.streaming_model = "gpt-4o-mini-transcribe"  # Faster, lower latency option
    config.enable_text_refinement = False  # Disable for faster response
    config.fallback_to_file_transcription = False  # Pure streaming mode
    
    logger.info("Running streaming transcription demo...")
    logger.info("This demo uses pure streaming mode with no fallback.")
    logger.info("Speech will be transcribed in real-time as you speak.")
    
    app = PushToTalkApp(config)
    
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Demo ended")

if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment to run the streaming demo instead
    # streaming_transcription_demo() 