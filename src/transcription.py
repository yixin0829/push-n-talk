import os
import logging
import time
import json
import base64
import asyncio
import threading
import websockets
import queue
import numpy as np
from typing import Optional, Callable
from openai import OpenAI
import tempfile

logger = logging.getLogger(__name__)

def bytes_to_16bit_pcm_base64(audio_bytes: bytes) -> str:
    """Convert raw audio bytes to base64-encoded 16-bit PCM."""
    try:
        # Convert bytes to base64 directly - PyAudio already gives us 16-bit PCM
        return base64.b64encode(audio_bytes).decode('ascii')
    except Exception as e:
        logger.error(f"Error converting audio to base64: {e}")
        return ""

class Transcriber:
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        """
        Initialize the transcriber with OpenAI API.
        
        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY environment variable
            model: Whisper model to use (default: whisper-1, for streaming: gpt-4o-transcribe)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
        # Streaming transcription state
        self.streaming_transcription = ""
        self.final_transcription = ""
        self.is_streaming = False
        self.streaming_lock = threading.Lock()
        self.transcription_callback: Optional[Callable[[str, bool], None]] = None
        
        # Audio streaming queue for WebSocket
        self.audio_queue = queue.Queue()
        self.websocket_instance = None
        
    def transcribe_audio(self, audio_file_path: str, language: Optional[str] = None) -> Optional[str]:
        """
        Transcribe audio file to text using OpenAI API.
        
        Args:
            audio_file_path: Path to the audio file
            language: Language code (optional, auto-detect if None)
            
        Returns:
            Transcribed text or None if transcription failed
        """
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return None
            
        try:
            start_time = time.time()
            logger.debug(f"Starting transcription for: {audio_file_path}")
            
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language,
                    response_format="text"
                )
            
            transcription_time = time.time() - start_time
            
            # Clean up temporary file
            try:
                os.remove(audio_file_path)
                logger.debug(f"Cleaned up temporary audio file: {audio_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up audio file {audio_file_path}: {e}")
            
            if isinstance(transcript, str):
                text = transcript.strip()
            else:
                text = transcript.text.strip() if hasattr(transcript, 'text') else str(transcript).strip()
            
            logger.info(f"Transcription successful: {len(text)} characters in {transcription_time:.2f}s")
            return text if text else None
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            # Clean up temporary file even on error
            try:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
            except:
                pass
            return None
    
    def start_streaming_transcription(self, 
                                     callback: Optional[Callable[[str, bool], None]] = None,
                                     model: str = "gpt-4o-transcribe",
                                     language: Optional[str] = None) -> bool:
        """
        Start streaming transcription session.
        
        Args:
            callback: Optional callback function called with (text, is_final) for each transcription update
            model: Model to use for streaming (gpt-4o-transcribe or gpt-4o-mini-transcribe)
            language: Language code (optional, auto-detect if None)
            
        Returns:
            True if streaming started successfully, False otherwise
        """
        if self.is_streaming:
            logger.warning("Streaming transcription already in progress")
            return False
        
        with self.streaming_lock:
            self.streaming_transcription = ""
            self.final_transcription = ""
            self.transcription_callback = callback
            self.is_streaming = True
            
            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Store streaming model and language for the WebSocket session
        self.streaming_model = model
        self.streaming_language = language
        
        logger.info(f"Starting streaming transcription with model: {model}")
        return True
    
    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk to the streaming queue."""
        if self.is_streaming:
            try:
                self.audio_queue.put(audio_data, block=False)
            except queue.Full:
                logger.warning("Audio queue is full, dropping audio chunk")
    
    async def _setup_streaming_connection(self):
        """Set up WebSocket connection for streaming transcription."""
        try:
            # OpenAI Realtime API WebSocket URL
            ws_url = "wss://api.openai.com/v1/realtime?intent=transcription"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            }
            
            async with websockets.connect(ws_url, additional_headers=headers, max_size=None) as websocket:
                logger.info("WebSocket connection established for streaming transcription")
                self.websocket_instance = websocket
                
                # Send session configuration
                await self._send_session_config(websocket)
                
                # Start concurrent tasks for sending audio and receiving messages
                await asyncio.gather(
                    self._send_audio_stream(websocket),
                    self._receive_streaming_messages(websocket)
                )
                
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            with self.streaming_lock:
                self.is_streaming = False
            self.websocket_instance = None
    
    async def _send_session_config(self, websocket):
        """Send session configuration to the WebSocket."""
        session_config = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": self.streaming_model
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            },
        }
        
        # Add language if specified
        if self.streaming_language:
            session_config["input_audio_transcription"]["language"] = self.streaming_language
        
        await websocket.send(json.dumps(session_config))
        logger.debug("Session configuration sent")
    
    async def _send_audio_stream(self, websocket):
        """Send audio chunks from queue to WebSocket with proper pacing."""
        try:
            while self.is_streaming:
                try:
                    # Get audio chunk with timeout
                    audio_data = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: self.audio_queue.get(timeout=0.1)
                        ), 
                        timeout=0.2
                    )
                    
                    # Send audio chunk
                    await self.send_audio_chunk(audio_data, websocket)
                    
                    # Add small delay for real-time pacing
                    await asyncio.sleep(0.01)
                    
                except (queue.Empty, asyncio.TimeoutError):
                    # No audio data available, continue waiting
                    await asyncio.sleep(0.01)
                    continue
                except Exception as e:
                    logger.error(f"Error sending audio stream: {e}")
                    break
            
            # Send end signal when streaming stops
            await self.send_audio_end_signal(websocket)
                    
        except Exception as e:
            logger.error(f"Error in audio streaming: {e}")
    
    async def _receive_streaming_messages(self, websocket):
        """Receive and process messages from the WebSocket."""
        try:
            while self.is_streaming:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    await self._handle_streaming_message(json.loads(message))
                except asyncio.TimeoutError:
                    continue  # Keep the loop alive
                except websockets.exceptions.ConnectionClosedError:
                    logger.info("WebSocket connection closed")
                    break
                    
        except Exception as e:
            logger.error(f"Error receiving streaming messages: {e}")
        finally:
            with self.streaming_lock:
                self.is_streaming = False
    
    async def _handle_streaming_message(self, message):
        """Handle incoming streaming transcription messages."""
        msg_type = message.get("type", "")
        
        if msg_type == "conversation.item.input_audio_transcription.delta":
            # Incremental transcription update
            delta = message.get("delta", "")
            with self.streaming_lock:
                self.streaming_transcription += delta
                
            # Call callback with partial result
            if self.transcription_callback and delta:
                try:
                    self.transcription_callback(self.streaming_transcription, False)
                except Exception as e:
                    logger.error(f"Error in transcription callback: {e}")
                    
        elif msg_type == "conversation.item.input_audio_transcription.completed":
            # Final transcription for this segment
            completed_text = message.get("transcript", "")
            with self.streaming_lock:
                if completed_text:
                    self.final_transcription += " " + completed_text if self.final_transcription else completed_text
                    self.streaming_transcription = ""  # Reset for next segment
                    
            # Call callback with final result for this segment
            if self.transcription_callback and completed_text:
                try:
                    self.transcription_callback(completed_text, True)
                except Exception as e:
                    logger.error(f"Error in transcription callback: {e}")
                    
            logger.debug(f"Completed transcription segment: {completed_text}")
            
        elif msg_type == "input_audio_buffer.speech_started":
            logger.debug("Speech detected")
            
        elif msg_type == "input_audio_buffer.speech_stopped":
            logger.debug("Speech stopped")
            
        elif msg_type == "transcription_session.created":
            logger.info("Transcription session created")
            
        elif msg_type == "transcription_session.updated":
            logger.info("Transcription session updated")
    
    async def send_audio_chunk(self, audio_data: bytes, websocket):
        """Send audio chunk to the WebSocket for streaming transcription."""
        try:
            # Convert audio bytes to base64-encoded 16-bit PCM
            audio_base64 = bytes_to_16bit_pcm_base64(audio_data)
            
            if audio_base64:  # Only send if conversion was successful
                # Create audio buffer message
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64,
                }
                
                await websocket.send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Error sending audio chunk: {e}")
    
    async def send_audio_end_signal(self, websocket):
        """Send end signal to indicate audio stream is complete."""
        try:
            message = {"type": "input_audio_buffer.end"}
            await websocket.send(json.dumps(message))
            logger.debug("Audio end signal sent")
        except Exception as e:
            logger.error(f"Error sending audio end signal: {e}")
    
    def stop_streaming_transcription(self) -> str:
        """
        Stop streaming transcription and return the final accumulated transcription.
        
        Returns:
            Final accumulated transcription text
        """
        with self.streaming_lock:
            self.is_streaming = False
            final_text = self.final_transcription.strip()
            
            # Include any remaining partial transcription
            if self.streaming_transcription.strip():
                final_text += " " + self.streaming_transcription.strip() if final_text else self.streaming_transcription.strip()
            
            # Reset state
            self.streaming_transcription = ""
            self.final_transcription = ""
            self.transcription_callback = None
            
        logger.info(f"Streaming transcription stopped. Final text: {final_text}")
        return final_text
    
    def get_current_transcription(self) -> tuple[str, str]:
        """
        Get current transcription state.
        
        Returns:
            Tuple of (final_transcription, streaming_transcription)
        """
        with self.streaming_lock:
            return self.final_transcription, self.streaming_transcription
    
    def is_streaming_active(self) -> bool:
        """Check if streaming transcription is currently active."""
        return self.is_streaming 