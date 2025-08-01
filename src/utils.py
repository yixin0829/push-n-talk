import threading
import logging
import math
import array
import simpleaudio as sa

logger = logging.getLogger(__name__)


def _beep(frequency: int, duration_ms: int) -> None:
    """Play a beep using simpleaudio if available."""

    if sa is None:
        logger.warning("simpleaudio not installed; skipping audio feedback")
        return

    sample_rate = 44100
    num_samples = int(sample_rate * (duration_ms / 1000))
    amplitude = 32767
    buf = array.array("h")
    for i in range(num_samples):
        sample = amplitude * math.sin(2 * math.pi * frequency * i / sample_rate)
        buf.append(int(sample))

    try:
        play_obj = sa.play_buffer(buf.tobytes(), 1, 2, sample_rate)
        play_obj.wait_done()
    except Exception as e:  # pragma: no cover - runtime audio errors
        logger.error(f"Failed to play beep: {e}")


def play_start_feedback() -> None:
    """Play a high-pitched beep for recording start."""

    threading.Thread(target=_beep, args=(880, 150), daemon=True).start()


def play_stop_feedback() -> None:
    """Play a lower-pitched confirmation beep for recording stop."""

    threading.Thread(target=_beep, args=(660, 100), daemon=True).start()
