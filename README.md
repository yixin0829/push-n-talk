# PushToTalk - AI Refined Speech-to-Text Dictation

A Python application that provides push-to-talk speech-to-text functionality with AI speech to text transcription, smart text refinement, and automatic text insertion into the active window on Windows, macOS, and Linux. **Now features a persistent GUI configuration interface with real-time status management and easy application control.**

## Features

- **🎯 GUI Interface**: Integrated configuration control and application status monitoring in one window
- **🎤 Push-to-Talk Recording**: Hold a customizable hotkey to record audio
- **🤖 Speech-to-Text**: Uses OpenAI Whisper for accurate transcription
- **✨ Text Refinement**: Improves transcription quality using Refinement Models
- **📝 Auto Text Insertion**: Automatically inserts refined text into the active window
- **🔊 Audio Feedback**: Optional audio cues for recording start/stop
- **📋 Multiple Insertion Methods**: Support for clipboard and sendkeys insertion

## Roadmap

- [x] GUI for configuration
- [ ] Customizable glossary for transcription refinement
- [ ] Streaming transcription with ongoing audio
- [x] Cross-platform support (MacOS, Linux)

## Requirements

- Windows, macOS, or Linux OS
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- OpenAI API key (https://platform.openai.com/docs/api-reference/introduction)
- Microphone access (for recording)
- Administrator privileges (for global hotkey detection)

## Quick Start (GUI Application)

### For End Users (Recommended)

1. **Download and launch**:
   - Download `PushToTalk.exe` from releases
   - Double-click to launch the configuration interface

2. **One-window setup and control**:
   - **Welcome section** explains the application at the top
   - **Configure your settings** in the organized sections below
   - **Click "Start Application"** to begin - the GUI stays open
   - **Monitor status** with real-time indicators (green = running, gray = stopped)
   - **View active settings** displayed when running
   - **Easy control** with "Stop Application" button to terminate

3. **Daily usage**:
   - GUI provides persistent control and status monitoring
   - Use your configured hotkeys to record and transcribe
   - Start/stop the service anytime from the GUI
   - No separate console or technical setup required

### For Developers

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd push-to-talk
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Run the GUI application**:
   ```bash
   uv run python main_gui.py
   ```

4. **Or run the console version**:
   ```bash
   uv run python main_console.py
   ```

## GUI Configuration Interface

The application features a comprehensive, persistent configuration GUI with organized sections:

### 🏠 Welcome & Status
- **Real-Time Status**: Visual indicators show current application state
  - **Gray circle + "Ready to start"**: Application stopped
  - **Green circle + "Running - Use your configured hotkeys"**: Application running
- **Active Settings Display**: Shows current hotkeys and enabled features when running

### 🔑 API Settings
- **OpenAI API Key**: Secure entry with show/hide functionality
- **Model Selection**: Choose Whisper and Refinement Models
- **API Key Testing**: Validate your credentials

### 🎵 Audio Settings
- **Sample Rate**: 8kHz to 44.1kHz options (16kHz recommended)
- **Chunk Size**: Buffer size configuration
- **Channels**: Mono/stereo recording options
- **Helpful Recommendations**: Built-in guidance for optimal settings

### ⌨️ Hotkey Configuration
- **Push-to-Talk Hotkey**: Hold to record (default: Ctrl+Shift+Space)
- **Toggle Recording Hotkey**: Press once to start/stop (default: Ctrl+Shift+T)
- **Validation**: Prevents duplicate hotkey assignments
- **Examples**: Common hotkey combinations provided

### 📄 Text Insertion Settings
- **Insertion Method**: Choose between clipboard (fast) or sendkeys (compatible)
- **Insertion Delay**: Fine-tune timing for different applications
- **Method Guidance**: Recommendations for each approach



## How to Use

### Via GUI (Recommended)
1. **Launch**: Double-click `PushToTalk.exe` or run `uv run python main_gui.py`
2. **Configure**: Use the integrated setup interface with welcome guidance
3. **Start**: Click "Start Application" - GUI stays open with status indicators
4. **Monitor**: Watch real-time status and active settings display
5. **Use**: Background operation with your configured hotkeys
6. **Control**: Use "Stop Application" button to terminate, or restart anytime

### Via Console
1. **Run**: `uv run python main_console.py`
2. **Configure**: Edit `push_to_talk_config.json` manually or set environment variables
3. **Use**: Same hotkey functionality as GUI

## Building the Application

### Build GUI Executable
```bash
.\build.bat
```
This creates `dist\PushToTalk.exe` - a standalone GUI application.

### Build Console Executable
```bash
# First, modify push_to_talk.spec to
# 1. Replace main_gui.py with main_console.py
# 2. Set console=True
uv run pyinstaller push_to_talk.spec
```

## Configuration

The application supports both GUI and file-based configuration:

### Via GUI (Recommended)
- Launch the application to access the integrated configuration interface
- **All settings** validated and saved automatically to `push_to_talk_config.json`
- **Real-time status** shows application state with visual indicators

### File-Based Configuration
The application creates a `push_to_talk_config.json` file. Example configuration file:

```json
{
  "openai_api_key": "your_api_key_here",
  "stt_model": "gpt-4o-transcribe",
  "refinement_model": "gpt-4.1-nano",
  "sample_rate": 16000,
  "chunk_size": 1024,
  "channels": 1,
  "hotkey": "ctrl+shift+space",
  "toggle_hotkey": "ctrl+shift+t",
  "insertion_method": "sendkeys",
  "insertion_delay": 0.005,
  "enable_text_refinement": true,
  "enable_logging": true,
  "enable_audio_feedback": true
}
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `openai_api_key` | string | `""` | Your OpenAI API key for Whisper and GPT services. Required for transcription and text refinement. Can be set via GUI, config file, or `OPENAI_API_KEY` environment variable. |
| `stt_model` | string | `"gpt-4o-transcribe"` | STT Model for speech-to-text. Options: `gpt-4o-transcribe`, `whisper-1`. |
| `refinement_model` | string | `"gpt-4.1-nano"` | Refinement Model for text refinement. Options: `gpt-4.1-nano`, `gpt-4o-mini`, `gpt-4o`. |
| `sample_rate` | integer | `16000` | Audio sampling frequency in Hz. 16kHz is optimal for speech recognition with Whisper. |
| `chunk_size` | integer | `1024` | Audio buffer size in samples. Determines how much audio is read at once (affects latency vs performance). |
| `channels` | integer | `1` | Number of audio channels. Use `1` for mono recording (recommended for speech). |
| `hotkey` | string | `"ctrl+shift+space"` | Hotkey combination for push-to-talk. See [Hotkey Options](#hotkey-options) for examples. |
| `toggle_hotkey` | string | `"ctrl+shift+t"` | Hotkey combination for toggle recording mode. Press once to start, press again to stop. |
| `insertion_method` | string | `"sendkeys"` | Method for inserting text. Options: `sendkeys` (better for special chars), `clipboard` (faster). |
| `insertion_delay` | float | `0.005` | Delay in seconds before text insertion. Helps ensure target window is ready. |
| `enable_text_refinement` | boolean | `true` | Whether to use GPT to refine transcribed text. Disable for faster processing without refinement. |
| `enable_logging` | boolean | `true` | Whether to enable detailed logging to `push_to_talk.log` file and console. |
| `enable_audio_feedback` | boolean | `true` | Whether to play sophisticated audio cues when starting/stopping recording. Provides immediate feedback for hotkey interactions. |

#### Audio Quality Settings

- **sample_rate**:
  - `16000` (16kHz) - Recommended for speech (Whisper optimized)
  - `8000` (8kHz) - Lower quality but faster processing
  - `44100` (44.1kHz) - CD quality (overkill for speech, slower)

- **chunk_size**:
  - `512` - Lower latency, more CPU overhead
  - `1024` - Balanced (recommended)
  - `2048` - Higher latency, less CPU usage

- **channels**:
  - `1` - Mono recording (recommended for speech)
  - `2` - Stereo recording (unnecessary for speech-to-text)

### Hotkey Options

You can configure different hotkey combinations for both modes:

**Push-to-talk hotkey** (hold to record):
- `ctrl+shift+space` (default)
- `alt+space`
- `ctrl+alt+r`
- `f12`

**Toggle hotkey** (press once to start, press again to stop):
- `ctrl+shift+t` (default)
- `f11`
- `ctrl+alt+t`
- `alt+t`

Both hotkeys support any combination from the `keyboard` library.

### Text Insertion Methods

- **sendkeys** (default): Simulates individual keystrokes, better for special characters
- **clipboard**: Faster and more reliable, uses Ctrl+V

### Audio Feedback

The application includes clean and simple audio feedback:

- **Recording Start**: A crisp high-pitched beep (880 Hz) that signals recording has begun
- **Recording Stop**: A lower confirmation beep (660 Hz) that confirms recording completion
- **Non-Blocking**: Audio playback runs in separate threads to avoid interfering with recording or transcription
- **Configurable**: Can be toggled on/off via GUI or configuration JSON file
- **Cross-Platform**: Audio cues played using the `simpleaudio` package

## Architecture

```mermaid
flowchart TB
    %% GUI
    GUI -->|"Save Configuration"| PushToTalkApp
    GUI -.->|"Real-time Updates"| PushToTalkApp

    %% Main Flow
    PushToTalkApp -->|"Initialize"| HotkeyService
    HotkeyService -->|"Start/Stop Recording"| AudioRecorder
    AudioRecorder -->|"Audio File"| Transcriber
    Transcriber -->|"AI Transcription"| TextRefiner
    TextRefiner -->|"AI Refinement"| TextInserter
```

The application consists of several modular components:

### Core Components

- **ConfigurationGUI** (`src/config_gui.py`): User-friendly GUI for settings management
- **MainGUI** (`main_gui.py`): Entry point with welcome flow and startup management
- **AudioRecorder** (`src/audio_recorder.py`): Handles audio recording using PyAudio
- **Transcriber** (`src/transcription.py`): Converts speech to text using OpenAI Whisper
- **TextRefiner** (`src/text_refiner.py`): Improves transcription using Refinement Models
 - **TextInserter** (`src/text_inserter.py`): Inserts text into active windows using `pyautogui` and `pyperclip`
- **HotkeyService** (`src/hotkey_service.py`): Manages global hotkey detection
- **PushToTalkApp** (`src/push_to_talk.py`): Main application orchestrator with dynamic configuration updates

### User Experience Flow

1. **Launch** → Single window with integrated welcome and configuration
2. **Configure** → Comprehensive GUI with organized settings sections
3. **Start** → Click "Start Application" with immediate visual feedback
4. **Monitor** → Real-time status indicators and active settings display
5. **Operate** → Background push-to-talk with persistent GUI control
6. **Control** → Easy start/stop with "Stop Application" button
7. **Manage** → Multiple start/stop cycles without closing the interface

### Data Flow

1. User presses hotkey → Audio recording starts
2. User releases hotkey → Recording stops
3. Audio file is sent to OpenAI Whisper for transcription
4. Raw transcription is refined using Refinement Models (if enabled)
5. Refined text is inserted into the active window

## Dependencies

- **tkinter**: GUI interface (built into Python)
- **keyboard**: Global hotkey detection
- **pyaudio**: Audio recording
- **openai**: Speech-to-text and text refinement
- **pyautogui**: Cross-platform keyboard input
- **pyperclip**: Clipboard management
- **simpleaudio**: Audio feedback beeps
- **python-dotenv**: Environment variable management

## Troubleshooting

### GUI Application Issues

1. **Application won't start**:
   - Make sure you're running as Administrator for hotkey detection
   - Check that the executable isn't blocked by antivirus
   - Try running from command line to see error messages

2. **Status indicators not updating**:
   - The GUI should show real-time status changes when starting/stopping
   - If stuck, try restarting the application
   - Check `push_to_talk.log` for error details

3. **Start/Stop button not working**:
   - Ensure all required fields are filled (especially OpenAI API key)
   - Use "Test Configuration" to validate settings
   - Check that no other instance is running

4. **Settings not saving**:
   - Ensure the application has write permissions in its directory
   - Check that the configuration file isn't marked as read-only
   - Try running as Administrator

### Common Issues

1. **"No module named 'pyautogui'"** (Development):
   ```bash
   uv add pyautogui pyperclip simpleaudio
   ```

2. **"Could not find PyAudio"** (Development):
   - Install PyAudio: `uv add pyaudio`
   - On Windows, you may need Visual C++ build tools

3. **Hotkey not working**:
   - Run as administrator (required for global hotkey detection)
   - Check if another application is using the same hotkey
   - Try a different hotkey combination in the GUI
   - Ensure the application shows "Running" status in the GUI

4. **OpenAI API errors**:
   - Use the "Test Configuration" button in the GUI to validate settings
   - Verify your API key is valid and has sufficient credits
   - Check your OpenAI account has access to the models you're using
   - Ensure internet connectivity

5. **Text not inserting**:
   - Make sure the target window is active and has a text input field
   - Try switching insertion method in the GUI (sendkeys vs clipboard)
   - Check OS permissions for clipboard access
   - Increase insertion delay if text appears truncated

6. **GUI appearance issues**:
   - Try restarting the application
   - Check display scaling settings (recommended: 100-150%)
   - Ensure your operating system is up to date

### Logging

Logs are written to `push_to_talk.log`. The GUI application logs only to file for cleaner user experience, while console mode logs to both file and console.

Log levels:
- INFO: Normal operation events
- WARNING: Non-critical issues
- ERROR: Critical errors

## Advanced Usage

### Custom Text Refinement Prompts

You can customize the text refinement behavior:

```python
from src import PushToTalkApp, PushToTalkConfig

app = PushToTalkApp()
app.text_refiner.set_custom_prompt(
    "Your custom refinement instructions here..."
)
```

### Programmatic Control

```python
from src import PushToTalkApp, PushToTalkConfig

# Create custom config
config = PushToTalkConfig()
config.hotkey = "f12"
config.toggle_hotkey = "f11"
config.enable_text_refinement = False

# Run application
app = PushToTalkApp(config)

# Update configuration dynamically
new_config = PushToTalkConfig()
new_config.openai_api_key = "new_key"
app.update_configuration(new_config)

# Change hotkeys
app.change_hotkey("ctrl+alt+r")  # Change push-to-talk hotkey
app.change_toggle_hotkey("ctrl+alt+t")  # Change toggle hotkey

# Toggle features
app.toggle_audio_feedback()  # Toggle audio feedback
app.toggle_text_refinement()  # Toggle text refinement

# Check status
status = app.get_status()
print(f"Push-to-talk hotkey: {status['hotkey']}")
print(f"Toggle hotkey: {status['toggle_hotkey']}")
print(f"Recording mode: {status['recording_mode']}")
print(f"Audio feedback enabled: {status['audio_feedback_enabled']}")

app.run()
```

### GUI Integration

```python
from src.config_gui import show_configuration_gui
from src.push_to_talk import PushToTalkConfig

# Show persistent configuration GUI
result, config = show_configuration_gui()
if result == "close":
    # User closed the application
    print("Application closed by user")
```

## Performance Tips

1. **Optimize audio settings**: Lower sample rates (8000-16000 Hz) for faster processing
2. **Disable text refinement**: For faster transcription without GPT processing
3. **Use clipboard method**: Generally faster than sendkeys for text insertion
4. **Short recordings**: Keep recordings under 30 seconds for optimal performance
5. **Monitor via GUI**: Use the status indicators to verify application is running efficiently

## Security Considerations

- **API Key Security**: GUI stores API keys securely; avoid sharing configuration files
- **Administrator Rights**: Required for global hotkey detection
- **Microphone Access**: Application needs microphone permissions
- **Network Access**: Required for OpenAI API calls
- **File Permissions**: Ensure configuration files have appropriate access controls

## Version History

- **0.2.0**: Persistent GUI interface, real-time status management, improved user experience, packaging as executable
- **0.1.0**: Initial console-based release
