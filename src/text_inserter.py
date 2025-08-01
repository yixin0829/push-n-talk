import time
import logging
import sys
from typing import Optional

import pyautogui
import pyperclip

logger = logging.getLogger(__name__)


class TextInserter:
    def __init__(self, insertion_delay: float = 0.1):
        """
        Initialize the text inserter.

        Args:
            insertion_delay: Delay between keystrokes in seconds
        """
        self.insertion_delay = insertion_delay

    def insert_text(self, text: str, method: str = "clipboard") -> bool:
        """
        Insert text into the currently active window.

        Args:
            text: Text to insert
            method: Method to use for insertion ("clipboard" or "sendkeys")

        Returns:
            True if insertion was successful, False otherwise
        """
        if not text:
            logger.warning("Empty text provided for insertion")
            return False

        try:
            if method == "clipboard":
                return self._insert_via_clipboard(text)
            elif method == "sendkeys":
                return self._insert_via_sendkeys(text)
            else:
                logger.error(f"Unknown insertion method: {method}")
                return False

        except Exception as e:
            logger.error(f"Text insertion failed: {e}")
            return False

    def _insert_via_clipboard(self, text: str) -> bool:
        """Insert text by copying to clipboard and pasting."""

        try:
            original_clipboard = pyperclip.paste()
            pyperclip.copy(text)

            time.sleep(0.05)

            paste_keys = ["command", "v"] if sys.platform == "darwin" else ["ctrl", "v"]
            pyautogui.hotkey(*paste_keys)

            time.sleep(0.1)

            if original_clipboard:
                pyperclip.copy(original_clipboard)

            logger.info(f"Text inserted via clipboard: {len(text)} characters")
            return True

        except Exception as e:
            logger.error(f"Clipboard insertion failed: {e}")
            return False

    def _insert_via_sendkeys(self, text: str) -> bool:
        """Insert text by simulating individual keystrokes."""

        try:
            pyautogui.write(text, interval=self.insertion_delay)
            logger.info(f"Text inserted via sendkeys: {len(text)} characters")
            return True

        except Exception as e:
            logger.error(f"SendKeys insertion failed: {e}")
            return False

    def _get_clipboard_text(self) -> Optional[str]:
        """Get current clipboard text content."""
        try:
            return pyperclip.paste()
        except Exception:
            return None

    def _set_clipboard_text(self, text: str) -> None:
        """Set clipboard text content."""
        pyperclip.copy(text)

    def get_active_window_title(self) -> Optional[str]:
        """
        Get the title of the currently active window.

        Returns:
            Window title or None if no active window
        """
        try:
            window = pyautogui.getActiveWindow()
            if window:
                return window.title if window.title else None
            return None
        except Exception as e:
            logger.error(f"Failed to get active window title: {e}")
            return None
