# Kiwi 🥝

> **A fast, completely local, privacy-first AI voice assistant for Linux.**  
> Powered by **Ollama (`qwen2.5:3b`)**, **Faster-Whisper**, **Piper Neural TTS**, and direct **Linux desktop controls** (KDE Plasma, PipeWire, WirePlumber).

No cloud APIs. No telemetry. No subscriptions. Everything runs locally on your machine.

---

## ✨ Features

- 🧠 **Local Brain**: Uses Ollama (`qwen2.5:3b` by default; fits comfortably on a 4 GB laptop GPU or runs on CPU).
- 🎙️ **Instant Ear (STT)**: Faster-Whisper with Silero Voice Activity Detection (VAD) and acoustic domain vocabulary priming.
- 🗣️ **Natural Neural Voice (TTS)**: Piper TTS streaming high-fidelity 32 kHz neural speech (Mei US High), featuring whole-clause prosody and automatic unit expansion (`°C`, `km/h`, `$`, `%`).
- ⚡ **Sub-50ms Fast-Path Router**: Deterministic regex routing executes common hardware commands (volume, brightness, music, weather, desktop) instantly in `< 1 ms` without waiting for LLM token generation.
- 🎵 **Smart Media Ducking**: Automatically drops Spotify, YouTube, and VLC volume to 20% whenever you converse with Kiwi, and smoothly restores volume when done.
- 🧠 **Persistent Memory**: Remembers facts and preferences across reboots and sessions via `memory.json`.
- 🛑 **Multi-Channel Barge-In**: Interrupt long answers instantly via any keypress in terminal (`Space`, `Enter`, `Escape`), a global desktop shortcut (`SIGUSR1`), or standard voice sleep commands.
- 🛠️ **21 Built-in Linux Tools**: Hardware volume, brightness, KDE Night Color toggle, KWin desktop minimize/restore, Spotify auto-launch & play, desktop notification reader, battery status, weather, and timers.

---

## 📋 Requirements

- **OS**: Linux (optimized for KDE Plasma & PipeWire; works on any desktop environment with standard utilities).
- **Python**: 3.10 – 3.12+
- **Ollama**: Installed and running locally.
- **Hardware**: Runs smoothly on modern multi-core CPUs; NVIDIA GPU (4 GB+ VRAM) supported for fast LLM inference.

### System Dependencies

**Arch Linux / Manjaro / CachyOS:**
```bash
sudo pacman -S portaudio wireplumber playerctl brightnessctl libnotify wl-clipboard networkmanager
```

**Fedora:**
```bash
sudo dnf install portaudio wireplumber playerctl brightnessctl libnotify wl-clipboard NetworkManager
```

**Ubuntu / Debian:**
```bash
sudo apt install portaudio19-dev wireplumber playerctl brightnessctl libnotify-bin wl-clipboard network-manager
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/kiwi-assistant.git
cd kiwi-assistant
```

### 2. Set Up Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # (or .venv/bin/activate.fish if using fish shell)
pip install -r requirements.txt
```

### 3. Pull the Ollama Model
Ensure Ollama is running (`ollama serve`), then pull the default model:
```bash
ollama pull qwen2.5:3b
```

### 4. Download the Neural Voice Model
Download the high-fidelity **Mei (US High 32kHz)** voice model into the `voices/` directory:
```bash
mkdir -p voices
curl -L -o voices/en_US-mei-high.onnx "https://huggingface.co/VINUK/piper-gpl-revamped-voices/resolve/main/en/en_US/mei/high_extened/en_US-mei-high.onnx"
curl -L -o voices/en_US-mei-high.onnx.json "https://huggingface.co/VINUK/piper-gpl-revamped-voices/resolve/main/en/en_US/mei/high_extened/en_US-mei-high.onnx.json"
```

*(Optional: You can also use standard Piper voices like `en_GB-alan-medium` via `python -m piper.download_voices en_GB-alan-medium --download-dir voices`)*

---

## 🎮 Usage

### 1. Hands-Free Voice Mode (Recommended)
Wake up Kiwi with wake phrases, issue one-breath commands, and converse freely without repeating the wake word:
```bash
python jarvis.py --voice
```
- **Wake phrases**: *"Hey Kiwi"*, *"Hello Kiwi"*, or *"Kiwi"*
- **Sleep phrases**: *"Thank you"*, *"Standby"*, *"Go to sleep"*, or *"Be quiet"*
- **Auto-Standby**: When you say *"Open Spotify and play music"*, Kiwi starts playback and immediately enters standby so song lyrics aren't heard as speech.

### 2. Push-To-Talk Mode
Press `Enter` to record your voice, press `Enter` again when finished speaking:
```bash
python jarvis.py --voice --ptt
```

### 3. Fast Text Mode
Type instructions in terminal to test tools and prompts without audio:
```bash
python jarvis.py
```

---

## 🛑 Stopping & Interrupting Speech (Barge-In)

When Kiwi is reading long answers, you can halt speech and cancel token generation in `< 5 ms`:

1. **Terminal Keypress**: Tap any key in the Kiwi terminal (**Space**, **Enter**, **Escape**, **s**, or **q**).
2. **Global Desktop Shortcut (From any window)**:
   - In KDE Plasma (*System Settings → Shortcuts → Custom Commands*), bind a shortcut (e.g. `Meta + Escape`, `Pause/Break`, or `F8`) to:
     ```bash
     pkill -USR1 -f "jarvis.py"
     ```
   - Pressing this key silences Kiwi immediately even if you're in a browser, terminal, or full-screen app.
3. **Voice Sleep Commands**: Say *"Stop"*, *"Shut up"*, *"Be quiet"*, or *"Standby"* during the active listening phase.
4. **Ctrl + C**: Standard terminal interrupt cleanly stops current speech without terminating the process.

---

## 🛠️ Built-in Tools & Capabilities (21 Tools)

| Category | Tool | Spoken Examples |
|---|---|---|
| **Music & Playback** | `spotify_play`, `media` | *"Play Spotify"*, *"Pause music"*, *"Next track"*, *"Resume"* |
| **Now Playing** | `media_info` | *"What song is playing?"*, *"Who is singing this?"* |
| **Persistent Memory** | `remember`, `recall`, `forget` | *"Remember that my birthday is July 4th"*, *"What do you remember?"*, *"Forget my birthday"* |
| **Notifications** | `read_notifications` | *"Any notifications?"*, *"Check my notifications"* |
| **Night Color (KDE)** | `night_light` | *"Turn on night light"*, *"Warm up display"*, *"Cool down screen"* |
| **Desktop Management** | `show_desktop` | *"Show desktop"*, *"Minimize everything"*, *"Bring back all windows"* |
| **Application Control** | `open_app`, `close_all_windows` | *"Open Spotify"*, *"Launch terminal"*, *"Close all windows"* |
| **Audio Volume** | `volume` | *"Set volume to 75%"*, *"Turn it up"*, *"Mute"* |
| **Backlight** | `brightness` | *"Brightness to 80%"*, *"Dim the screen"* |
| **System Diagnostics** | `system_status` | *"System status"*, *"How are the CPU and GPU doing?"* |
| **Weather** | `get_weather` | *"What's the weather?"*, *"Weather in Kathmandu"* |
| **Screen Lock** | `lock_screen` | *"Lock my screen"*, *"Lock my laptop"* |
| **Screenshot** | `take_screenshot` | *"Take a screenshot"*, *"Capture screen"* |
| **Web Search & URLs** | `web_search`, `open_url` | *"Search online for top restaurants"*, *"Open github.com"* |
| **Clock & Calendar** | `current_time` | *"What time is it?"*, *"What's today's date?"* |
| **Battery Diagnostics** | `battery_status` | *"What's the battery level?"*, *"Is it charging?"* |
| **Countdown Timers** | `set_timer` | *"Set a timer for 10 minutes"* |
| **Quick Notes** | `note` | *"Take a note: review pull request tomorrow"* |

---

## ⚙️ Configuration (`config.json`)

Configure Kiwi parameters in `config.json` without modifying any Python code:

```json
{
  "model": "qwen2.5:3b",
  "voice_file": "voices/en_US-mei-high.onnx",
  "tts_speed": 1.0,
  "stt_model": "small.en",
  "idle_timeout": 25.0,
  "auto_ducking": true,
  "duck_volume": 0.20,
  "wake_phrases": [
    "hey kiwi", "hello kiwi", "kiwi", "yo kiwi", "hi kiwi", "okay kiwi"
  ],
  "sleep_phrases": [
    "thank you", "go to sleep", "standby", "enter standby",
    "stop listening", "be quiet", "shut up", "stop talking", "goodnight", "bye"
  ]
}
```

CLI flags (e.g. `--model`, `--voice-file`, `--tts-speed`) override settings in `config.json`.

---

## 🧪 Testing Tools Directly

You can test any tool individually without running the voice loop or LLM:

```bash
# System diagnostics & information
python tools.py system_status
python tools.py battery_status
python tools.py current_time
python tools.py get_weather '{"location": "Tokyo"}'

# Persistent memory
python tools.py remember '{"fact": "favorite genre is synthwave", "key": "music"}'
python tools.py recall '{"query": "music"}'
python tools.py forget '{"key": "music"}'

# Hardware & media controls
python tools.py volume '{"action": "set", "amount": 80}'
python tools.py media_info
python tools.py read_notifications
```

---

## 📁 Project Structure

```
├── jarvis.py          # Main entry point & state machine (KiwiAssistant)
├── voice.py           # Faster-Whisper STT engine & Piper TTS streaming Mouth
├── tools.py           # 21 PC control tools, ducking engine, & persistent memory
├── config.json        # Central configuration (models, speed, wake/sleep words)
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignores large neural models, caches, and user data
└── voices/            # Local neural voice ONNX models
```

---

## 📜 License & Disclaimer

This is a personal hobby project built for personal Linux productivity and experimentation. Distributed under the MIT License. Feel free to fork, experiment, and customize it for your desktop setup!
# Kiwi-Local-Voice-Assistant
