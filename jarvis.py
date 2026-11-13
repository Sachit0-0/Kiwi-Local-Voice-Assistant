#!/usr/bin/env python3
"""Kiwi: a local personal assistant for Linux.

Brain = Ollama (qwen2.5:3b), Hands = tools.py, Voice = voice.py.

Modes:
    python jarvis.py                # Text mode (type commands)
    python jarvis.py --speak        # Text input, spoken replies
    python jarvis.py --voice        # Hands-free: wake with 'Hey Kiwi' or 'Hello Kiwi'
    python jarvis.py --voice --ptt  # Push-to-talk (press Enter to talk)
"""

import argparse
import datetime
import json
import os
import random
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

import tools

HERE = os.path.dirname(os.path.abspath(__file__))

STOP_FILE = "/tmp/kiwi_stop"
_SIGUSR1_TRIGGERED = False

def _on_sigusr1(signum, frame):
    global _SIGUSR1_TRIGGERED
    _SIGUSR1_TRIGGERED = True

try:
    signal.signal(signal.SIGUSR1, _on_sigusr1)
except Exception:
    pass


class TerminalNonBlocking:
    """Non-blocking terminal single-key detector without waiting for Enter."""

    def __init__(self):
        self.is_tty = sys.stdin.isatty()
        self.old_settings = None

    def __enter__(self):
        if self.is_tty:
            try:
                import termios, tty
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self.old_settings = None
        return self

    def __exit__(self, *args):
        if self.is_tty and self.old_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

    def check_key(self):
        if not self.is_tty:
            return False
        import select
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                sys.stdin.read(1)
                return True
        except Exception:
            pass
        return False


# =============================================================================
# Configuration (loaded from config.json with env/CLI overrides)
# =============================================================================

CONFIG_PATH = os.path.join(HERE, "config.json")
CONFIG = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            CONFIG = json.load(f)
    except Exception as e:
        print(f"[warning] failed to read config.json: {e}", file=sys.stderr)

MODEL         = os.environ.get("KIWI_MODEL", os.environ.get("JARVIS_MODEL", CONFIG.get("model", "qwen2.5:3b")))
HOST          = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not HOST.startswith("http"):
    HOST = "http://" + HOST

NUM_CTX        = 4096   # context window (fits comfortably on 4 GB GPU)
KEEP_ALIVE     = "30m"  # keep model hot in memory
MAX_HISTORY    = 20     # messages kept in conversation memory
MAX_TOOL_STEPS = 4      # prevent infinite tool-call loops
IDLE_TIMEOUT   = float(CONFIG.get("idle_timeout", 25.0))
AUTO_DUCKING   = bool(CONFIG.get("auto_ducking", True))
DEFAULT_SPEED  = float(CONFIG.get("tts_speed", 1.0))
DEFAULT_STT    = CONFIG.get("stt_model", "small.en")
DEFAULT_VOICE  = os.path.join(HERE, CONFIG.get("voice_file", "voices/en_US-mei-high.onnx"))

SYSTEM_PROMPT = (
    "You are Kiwi, a calm, witty, and concise AI assistant running locally on the user's Linux laptop. "
    "Your replies are spoken aloud, so keep them to one or two short, direct sentences of plain text. "
    "Never address the user as 'sir'. "
    "You have tools to control laptop hardware (volume, brightness, night_light, media, spotify_play, "
    "media_info, lock_screen, take_screenshot, show_desktop, open_app, close_all_windows), "
    "access information (get_weather, system_status, battery_status, current_time, wifi_status, read_notifications), "
    "and utilities (remember, recall, forget, web_search, open_url, set_timer, clipboard, note). "
    "When asked to play a song or open Spotify, call spotify_play. "
    "When asked to adjust night light or display warmth, call night_light directly — do not open Settings. "
    "When asked to remember a fact or recall what you remember, call remember or recall. "
    "Call a tool only when the user requests an action or live data. "
    "Never claim you performed an action unless a tool result confirms it. "
    "For greetings or questions about whether you can hear, reply in plain text without tools."
)

HELP = (
    "Commands: /reset clear conversation | /tools list tools | /help show this | /quit exit\n"
    "Wake phrases: 'Hey Kiwi', 'Hello Kiwi', 'Kiwi' | Sleep phrases: 'Thank you', 'Go to sleep', 'Standby'"
)


# =============================================================================
# Wake / Sleep detection & Conversational Variations
# =============================================================================

_WAKE_LIST = CONFIG.get("wake_phrases", ["hey kiwi", "hello kiwi", "kiwi", "yo kiwi", "hi kiwi", "okay kiwi"])
WAKE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in _WAKE_LIST) + r")\b",
    re.IGNORECASE,
)

_SLEEP_LIST = CONFIG.get("sleep_phrases", [
    "thank you", "go to sleep", "standby", "enter standby", "stop listening",
    "be quiet", "shut up", "stop talking", "goodnight", "bye"
])
SLEEP_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s).replace(r"\ ", r"\s+") for s in _SLEEP_LIST) + r"|^stop$)\b",
    re.IGNORECASE,
)

# Detects a trailing request to go to standby after performing an action
# e.g. "open spotify and standby", "turn off night light then go to sleep"
STANDBY_SUFFIX = re.compile(
    r"(?:[,;]?\s*(?:and|then|also)?\s*(?:stand\s*by|standby|go\s+to\s+sleep|sleep\s+now|sleep|stop\s+listening|goodnight))\s*[.!?]*$",
    re.IGNORECASE,
)

WAKE_RESPONSES = [
    "Online. What do you need?",
    "At your service.",
    "Ready. How can I help?",
    "Online and listening.",
    "I'm here. What's on your mind?",
    "Loud and clear. What do you need?",
    "Ready for instructions.",
    "Yes? What can I do for you?",
]

SLEEP_RESPONSES = [
    "Standing by.",
    "Going to sleep.",
    "Entering standby.",
    "Standing down.",
    "Catch you later.",
    "Sleeping now.",
    "Until next time.",
]


def extract_wake_remainder(text):
    """If text contains a wake phrase, return the remainder. Otherwise None."""
    m = WAKE_PATTERN.search(text)
    return text[m.end():].strip(" ,:?.!") if m else None


def is_sleep_trigger(text):
    """Return True if the user is asking Kiwi to sleep."""
    # Ignore YouTube / podcast outros ("thanks for watching", etc.)
    if re.search(r"\bthanks?\s+for\s+(watching|listening|subscribing|coming|tuning)\b",
                 text, re.IGNORECASE):
        return False
    # Long sentences without 'sleep'/'kiwi'/'standby' are probably ambient audio
    if (len(text.split()) > 6
            and not re.search(r"\b(sleep|stop\s+listening|kiwi|stand\s*by)\b", text, re.IGNORECASE)):
        return False
    return bool(SLEEP_PATTERN.search(text))


def is_spotify_play_trigger(text):
    """Return True if query starts/resumes music playback (triggers auto-standby)."""
    t = text.lower()
    if re.search(r"\b(what|who|which)\b", t):
        return False
    if re.search(r"\b(pause|stop)\b", t):
        return False
    return bool(
        "spotify" in t
        or re.search(r"\b(play\s+(a|some|your)?\s*(song|music|track)|start\s+(music|spotify)|open\s+and\s+play)\b", t)
    )


# =============================================================================
# Command Router  — sub-50 ms deterministic fast-path
# =============================================================================

# Named numeric words for volume / brightness level extraction
NUM_WORDS = {
    "zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "one hundred": 100, "full": 100, "max": 100, "maximum": 100,
    "half": 50, "min": 0, "minimum": 0,
}


def extract_level(text):
    """Extract a numeric percentage (0-100) from natural language."""
    m = re.search(r"\b(?:to|at|is|set to)\s*(?:about\s*)?(\d{1,3})\s*%?\b", text)
    if m and 0 <= int(m.group(1)) <= 100:
        return int(m.group(1))
    m2 = re.search(r"\b(\d{1,3})\s*%\b", text)
    if m2 and 0 <= int(m2.group(1)) <= 100:
        return int(m2.group(1))
    for word, val in NUM_WORDS.items():
        if re.search(r"\b(?:to|at|is|set to)\s*(?:about\s*)?" + word + r"\b", text):
            return val
    return None


class CommandRouter:
    """Fast, deterministic action router — no LLM required for common commands."""

    # Filler prefixes to strip before pattern matching
    _STRIP = [
        re.compile(r"^(?:hey\s+|yo\s+|hello\s+|hi\s+|ok\s+|okay\s+)?(?:kiwi|jarvis)[,:]?\s*", re.IGNORECASE),
        re.compile(r"^(?:no[,\s]+(?:i\s+meant\s+)?|and\s+|so\s+|actually\s+|well\s+)", re.IGNORECASE),
        re.compile(r"^(?:please\s+|could you\s+|can you\s+|would you\s+|just\s+)", re.IGNORECASE),
        re.compile(r"[?.!]+$"),
    ]

    _RE_HEAR_ME = re.compile(r"\b((can you\s+)?(hear|listen to)\s+(me|my voice)|are you (listening|there)|you hear me)\b", re.IGNORECASE)
    _RE_GREETING = re.compile(r"^(hello|hi|hey|greetings|how are you|how's it going)\b", re.IGNORECASE)
    _RE_UNMUTE = re.compile(r"\b(unmute|turn audio on)\b", re.IGNORECASE)
    _RE_MUTE = re.compile(r"\b(mute|silence|shut up|be quiet)\b", re.IGNORECASE)
    _RE_VOL_UP = re.compile(r"\b(volume up|turn (it )?up|louder|raise|increase).*(volume|audio|sound)?\b", re.IGNORECASE)
    _RE_VOL_DOWN = re.compile(r"\b(volume down|turn (it )?down|softer|quieter|lower|decrease).*(volume|audio|sound)?\b", re.IGNORECASE)
    _RE_BRIGHT_LOCK = re.compile(r"\b(lock|night|warm|cool)\b", re.IGNORECASE)
    _RE_BRIGHT_UP = re.compile(r"\b(brighter|brightness up|more brightness)\b", re.IGNORECASE)
    _RE_BRIGHT_DOWN = re.compile(r"\b(dimmer|dim|brightness down|lower brightness)\b", re.IGNORECASE)
    _RE_WARM = re.compile(r"\b(warm|cool)\s+(up\s+|down\s+)?(the\s+)?(screen|display)\b", re.IGNORECASE)
    _RE_SPOTIFY_PLAY = re.compile(r"\b(play\s+(a|some|your)?\s*(song|music|track)|start\s+(music|spotify)|open\s+and\s+play)\b", re.IGNORECASE)
    _RE_MEDIA_NOW = re.compile(r"\b(what(('s)| is)\s+(playing|this song|this track)|who\s+is\s+(singing|this)|current (song|track))\b", re.IGNORECASE)
    _RE_MEDIA_PAUSE = re.compile(r"\b(pause|stop playback|freeze music)\b", re.IGNORECASE)
    _RE_MEDIA_RESUME = re.compile(r"\b(resume|continue playing|unpause)\b", re.IGNORECASE)
    _RE_MEDIA_TOGGLE = re.compile(r"\b(play.?pause|toggle playback|toggle music)\b", re.IGNORECASE)
    _RE_MEDIA_NEXT = re.compile(r"\b(next (song|track)|skip( (this|the) (song|track))?)\b", re.IGNORECASE)
    _RE_MEDIA_PREV = re.compile(r"\b(previous (song|track)|go back a track|last track)\b", re.IGNORECASE)
    _RE_DESKTOP = re.compile(r"\b(minimize (everything|all|windows)|show desktop|hide all|restore (everything|all|windows)|bring (back|everything)|unminimize|toggle desktop)\b", re.IGNORECASE)
    _RE_SCREENSHOT = re.compile(r"\b(take (a )?screenshot|capture (the )?screen|screenshot)\b", re.IGNORECASE)
    _RE_LOCK = re.compile(r"\b(lock\s*(me|my\s*(screen|pc|laptop|computer)|the\s*(screen|pc|laptop|computer)|screen|pc|laptop|computer|session)?)\b", re.IGNORECASE)
    _RE_SYS_STATUS = re.compile(r"\b(system (status|specs|performance|info)|cpu (load|usage)|ram usage|gpu (temp|load))\b", re.IGNORECASE)
    _RE_WIFI_STATUS = re.compile(r"\b(wifi|wi.fi|network|internet connection)\b", re.IGNORECASE)
    _RE_WIFI_CHECK = re.compile(r"\b(status|connected|signal|strength)\b", re.IGNORECASE)
    _RE_TIME = re.compile(r"\b(what(('s)| is) the time|what time is it|current time|tell me the time)\b", re.IGNORECASE)
    _RE_DATE = re.compile(r"\b(what(('s)| is) (today's|the) date|what day is it)\b", re.IGNORECASE)
    _RE_BATTERY = re.compile(r"\b(battery( status| level| percentage)?|how much (battery|power|juice))\b", re.IGNORECASE)
    _RE_WEATHER_CITY = re.compile(r"\b(?:weather|forecast|temperature|temp)\s+(?:in|for|at)\s+([a-zA-Z\s]+)$", re.IGNORECASE)
    _RE_WEATHER_NOW = re.compile(r"\b(what(('s)| is) the weather|how(('s)| is) the weather|is it raining|weather today|weather forecast)\b", re.IGNORECASE)
    _RE_URL_EXPLICIT = re.compile(r"\b(?:go to|open|browse to|navigate to)\s+((?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/\S*)?)\b", re.IGNORECASE)
    _RE_URL_BARE = re.compile(r"\b((?:[a-zA-Z0-9-]+\.)+(?:com|org|net|io|edu|gov|co|app|dev|xyz|ai|me))\b", re.IGNORECASE)
    _RE_SEARCH = re.compile(r"\b(?:search\s+(?:for\s+|online\s+for\s+|in\s+browser\s+for\s+)?|google\s+)\s*(.+)$", re.IGNORECASE)
    _RE_APP = re.compile(r"^(?:open|launch)\s+(?:the\s+)?([a-zA-Z0-9_ \-]+?)(?:\s+(?:app|application|for me))?$", re.IGNORECASE)
    _RE_REMEMBER = re.compile(r"\bremember\s+(?:that\s+)?(.+)$", re.IGNORECASE)
    _RE_RECALL = re.compile(r"\b(?:what\s+do\s+you\s+remember|what\s+did\s+i\s+ask\s+you\s+to\s+remember|recall\s+(?:my\s+)?(?:memory|memories)|list\s+(?:my\s+)?memories)\b", re.IGNORECASE)
    _RE_FORGET = re.compile(r"\bforget\s+(?:about\s+)?(.+)$", re.IGNORECASE)
    _RE_NOTIFS = re.compile(r"\b(?:any|check|read|show)\s+(?:new\s+)?notifications\b|\bwhat(('s)| are)\s+(?:my\s+)?notifications\b", re.IGNORECASE)

    def _normalise(self, raw):
        t = raw.lower().strip()
        for pat in self._STRIP:
            t = pat.sub("", t).strip()
        return t

    def dispatch(self, raw_text):
        """Return a response string if handled; else None (falls through to LLM)."""
        text = self._normalise(raw_text)
        try:
            return (
                self._conversational(text)
                or self._volume(text)
                or self._brightness(text)
                or self._night_light(text)
                or self._media(text)
                or self._desktop(text)
                or self._system(text)
                or self._memory(text)
                or self._notifications(text)
                or self._weather(text)
                or self._url(text)
                or self._search(text)
                or self._app(text)
            )
        except tools.ToolError as e:
            return f"Unable to do that: {e}"
        except Exception as e:
            return f"Encountered an issue: {e}"

    def _memory(self, text):
        m = self._RE_REMEMBER.search(text)
        if m and not any(w in text for w in ("what", "how", "can", "could", "would")):
            fact = m.group(1).strip()
            if fact:
                return tools.remember(fact)
        if self._RE_RECALL.search(text):
            return tools.recall()
        m_f = self._RE_FORGET.search(text)
        if m_f:
            key = m_f.group(1).strip()
            if key:
                return tools.forget(key)
        return None

    def _notifications(self, text):
        if self._RE_NOTIFS.search(text):
            return tools.read_notifications()
        return None

    # ------------------------------------------------------------------
    # Individual handler sections
    # ------------------------------------------------------------------

    def _conversational(self, text):
        if self._RE_HEAR_ME.search(text):
            return "Loud and clear. What do you need?"
        if self._RE_GREETING.search(text) and len(text.split()) <= 4:
            return "Hello. How can I help you?"
        return None

    def _volume(self, text):
        if not any(w in text for w in ("volume", "audio", "sound", "louder", "quieter", "mute")):
            return None
        if self._RE_UNMUTE.search(text):
            tools.volume("unmute")
            return "Audio unmuted."
        if self._RE_MUTE.search(text) and not extract_level(text):
            tools.volume("mute")
            return "Muted."
        level = extract_level(text)
        if level is not None:
            tools.volume("set", level)
            return f"Volume set to {level}%."
        if self._RE_VOL_UP.search(text):
            tools.volume("up")
            return f"Volume raised to {tools._volume_text()}."
        if self._RE_VOL_DOWN.search(text):
            tools.volume("down")
            return f"Volume lowered to {tools._volume_text()}."
        return None

    def _brightness(self, text):
        if not any(w in text for w in ("brightness", "screen", "display", "dimmer", "brighter", "backlight")):
            return None
        # Avoid hijacking "screen lock", "display night light", etc.
        if self._RE_BRIGHT_LOCK.search(text):
            return None
        level = extract_level(text)
        if level is not None:
            tools.brightness("set", level)
            return f"Brightness set to {level}%."
        if self._RE_BRIGHT_UP.search(text):
            tools.brightness("up")
            lv = tools._brightness_text()
            return f"Brightness raised to {lv}." if lv else "Brightness increased."
        if self._RE_BRIGHT_DOWN.search(text):
            tools.brightness("down")
            lv = tools._brightness_text()
            return f"Brightness dimmed to {lv}." if lv else "Brightness lowered."
        return None

    def _night_light(self, text):
        keywords = ("nightlight", "night light", "night color", "nightlife", "night life", "blue light")
        if not (any(w in text for w in keywords) or self._RE_WARM.search(text)):
            return None
        return tools.night_light()

    def _media(self, text):
        # Spotify / play a song
        if "spotify" in text or self._RE_SPOTIFY_PLAY.search(text):
            return tools.spotify_play()
        # Now-playing info
        if self._RE_MEDIA_NOW.search(text):
            return tools.media_info()
        # Pause
        if self._RE_MEDIA_PAUSE.search(text):
            tools.media("pause")
            return "Playback paused."
        # Resume
        if self._RE_MEDIA_RESUME.search(text):
            tools.media("play")
            return "Resuming playback."
        # Toggle
        if self._RE_MEDIA_TOGGLE.search(text):
            tools.media("play_pause")
            return "Playback toggled."
        # Next
        if self._RE_MEDIA_NEXT.search(text):
            tools.media("next")
            return "Skipping to next track."
        # Previous
        if self._RE_MEDIA_PREV.search(text):
            tools.media("previous")
            return "Previous track."
        return None

    def _desktop(self, text):
        if self._RE_DESKTOP.search(text):
            return tools.show_desktop()
        return None

    def _system(self, text):
        if self._RE_SCREENSHOT.search(text):
            return tools.take_screenshot()
        if self._RE_LOCK.search(text):
            tools.lock_screen()
            return "Screen locked."
        if self._RE_SYS_STATUS.search(text):
            return tools.system_status()
        if self._RE_WIFI_STATUS.search(text) and self._RE_WIFI_CHECK.search(text):
            return tools.wifi_status()
        if self._RE_TIME.search(text) or self._RE_DATE.search(text):
            return tools.current_time()
        if self._RE_BATTERY.search(text):
            return tools.battery_status()
        return None

    def _weather(self, text):
        m = self._RE_WEATHER_CITY.search(text)
        if m:
            return tools.get_weather(m.group(1).strip())
        if self._RE_WEATHER_NOW.search(text):
            return tools.get_weather()
        return None

    def _url(self, text):
        # Explicit "open/go to" with a URL
        m = self._RE_URL_EXPLICIT.search(text)
        if not m:
            # Bare domain names
            m = self._RE_URL_BARE.search(text)
        if m and not any(w in text for w in ("volume", "brightness", "battery", "time", "timer")):
            url = m.group(1).strip()
            tools.open_url(url)
            return f"Opening {url}."
        return None

    def _search(self, text):
        m = self._RE_SEARCH.search(text)
        if m and not any(w in text for w in ("volume", "brightness", "battery", "time", "timer")):
            q = m.group(1).strip().strip("'\"")
            if q:
                tools.web_search(q)
                return f"Searching online for '{q}'."
        return None

    def _app(self, text):
        m = self._RE_APP.match(text)
        if not m:
            return None
        target = m.group(1).strip().lower()
        # Resolve via ALIASES first, then direct APPS lookup
        key = tools.ALIASES.get(target, target)
        if key in tools.APPS:
            tools.open_app(key)
            return f"Opening {key}."
        return None


router = CommandRouter()


# =============================================================================
# Ollama Client
# =============================================================================

class OllamaError(Exception):
    pass


# Keywords that must appear in the user query or last assistant message for a
# given tool call to be considered plausible (prevents 3B hallucinations).
TOOL_GUARD = {
    "media":           {"media", "music", "song", "track", "video", "spotify", "playback",
                        "play", "pause", "resume", "stop", "next", "previous", "skip"},
    "media_info":      {"song", "track", "music", "playing", "artist", "title", "album",
                        "who", "what"},
    "spotify_play":    {"spotify", "music", "song", "play", "tune"},
    "brightness":      {"brightness", "screen", "display", "dim", "brighter", "backlight"},
    "night_light":     {"night", "light", "blue", "color", "warm", "cool", "temperature",
                        "display", "screen"},
    "show_desktop":    {"desktop", "minimize", "restore", "windows", "hide", "bring"},
    "close_all_windows": {"close", "all", "windows", "apps", "everything"},
    "volume":          {"volume", "sound", "audio", "loud", "quiet", "mute", "unmute"},
    "lock_screen":     {"lock", "screen", "session", "me", "pc", "laptop", "computer"},
    "take_screenshot": {"screenshot", "capture", "snapshot"},
    "get_weather":     {"weather", "temperature", "forecast", "rain", "sunny", "cloudy",
                        "hot", "cold", "outside", "temp"},
    "system_status":   {"cpu", "ram", "gpu", "memory", "load", "system", "performance",
                        "usage", "specs", "hardware"},
    "wifi_status":     {"wifi", "wi-fi", "network", "internet", "connection", "signal"},
    "clipboard":       {"clipboard", "copy", "paste", "copied"},
    "note":            {"note", "remember", "jot", "write down", "save"},
    "remember":        {"remember", "store", "save", "memory", "remind"},
    "recall":          {"recall", "remember", "memory", "what", "know", "stored", "list"},
    "forget":          {"forget", "remove", "delete", "erase"},
    "read_notifications": {"notification", "notifications", "notify", "messages", "alerts"},
}


def is_tool_call_plausible(tool_name, user_prompt, last_assistant_msg=""):
    """Return False if the tool call is almost certainly a hallucination."""
    keywords = TOOL_GUARD.get(tool_name)
    if not keywords:
        return True   # no guard for this tool — always allow
    combined = f"{user_prompt} {last_assistant_msg}".lower()
    prompt_words = set(re.findall(r"\b\w+\b", combined))
    return bool(prompt_words & keywords)


def stream_chat(messages):
    """Yield ('token', text), ('tool_calls', list), ('stats', dict) from Ollama."""
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "tools": tools.schemas(),
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_ctx": NUM_CTX, "temperature": 0.3},
    }).encode()
    req = urllib.request.Request(
        f"{HOST}/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        if e.code == 404:
            raise OllamaError(f"Model '{MODEL}' not found. Run: ollama pull {MODEL}")
        raise OllamaError(f"Ollama returned HTTP {e.code}: {detail}")
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise OllamaError(
            f"Can't reach Ollama at {HOST} ({e}). "
            "Start it with: sudo systemctl start ollama   (or: ollama serve)"
        )
    with resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise OllamaError(chunk["error"])
            msg = chunk.get("message", {})
            if msg.get("content"):
                yield "token", msg["content"]
            if msg.get("tool_calls"):
                yield "tool_calls", msg["tool_calls"]
            if chunk.get("done"):
                yield "stats", chunk


def warm_up():
    """Pre-load the model onto the GPU in a background thread."""
    try:
        req = urllib.request.Request(
            f"{HOST}/api/generate",
            data=json.dumps({"model": MODEL, "keep_alive": KEEP_ALIVE}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()
    except Exception:
        pass


def parse_inline_tool_calls(text):
    """Fallback: parse JSON tool-call objects embedded in a plain-text LLM response."""
    decoder, calls, i = json.JSONDecoder(), [], 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return calls
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        i = end
        if isinstance(obj, dict) and obj.get("name") in tools.REGISTRY:
            args = obj.get("arguments", obj.get("parameters", {}))
            calls.append({"function": {"name": obj["name"], "arguments": args}})


def _generate(history, on_text, cancel_event=None):
    """Run one Ollama generation and return (text, tool_calls, stats)."""
    text, calls, stats = "", [], None
    mode, emitted = None, 0
    for kind, value in stream_chat(history):
        if cancel_event and cancel_event.is_set():
            break
        if kind == "tool_calls":
            calls.extend(value)
        elif kind == "stats":
            stats = value
        else:
            text += value
            if mode is None and text.strip():
                mode = "hold" if text.lstrip()[0] in "{[<" else "text"
            if mode == "text":
                on_text(text[emitted:])
                emitted = len(text)
    if cancel_event and cancel_event.is_set():
        return text, [], stats
    if mode == "hold":
        if not calls:
            calls = parse_inline_tool_calls(text)
        if calls:
            text = ""
        else:
            on_text(text)
    return text, calls, stats


def dynamic_system_prompt():
    """Build the system prompt with injected live context (time + battery)."""
    now = datetime.datetime.now().astimezone().strftime("%A, %d %B %Y, %I:%M %p")
    bat = "Battery status unknown"
    try:
        bat = tools.battery_status()
    except Exception:
        pass
    return f"{SYSTEM_PROMPT}\n[Live Context: {now} | {bat}]"


def run_turn(history, out, user_query, cancel_event=None):
    """Execute the LLM ↔ tool loop for one user turn."""
    stats = None
    last_result = None
    for _ in range(MAX_TOOL_STEPS):
        if cancel_event and cancel_event.is_set():
            break
        text, calls, stats = _generate(history, out.text, cancel_event=cancel_event)
        if cancel_event and cancel_event.is_set():
            break

        if not calls:
            if not text.strip():
                text = last_result or "Sorry, I didn't get a response."
                out.text(text)
            history.append({"role": "assistant", "content": text})
            return stats

        # Plausibility filter — block hallucinated hardware calls
        last_assistant = next(
            (m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"),
            ""
        )
        valid_calls = []
        for call in calls:
            fn   = call.setdefault("function", {})
            name = fn.get("name", "")
            if not is_tool_call_plausible(name, user_query, last_assistant):
                continue
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            fn["arguments"] = args
            valid_calls.append(call)

        if not valid_calls:
            fallback = text.strip() or "I am listening. How can I help you?"
            out.text(fallback)
            history.append({"role": "assistant", "content": fallback})
            return stats

        history.append({"role": "assistant", "content": text, "tool_calls": valid_calls})
        for call in valid_calls:
            if cancel_event and cancel_event.is_set():
                break
            fn     = call["function"]
            name   = fn.get("name", "")
            result = tools.call_tool(name, fn["arguments"])
            out.tool(name, fn["arguments"], result)
            history.append({"role": "tool", "tool_name": name, "content": result})
            last_result = result

    text = "Sorry, I got stuck on that one."
    out.text(text)
    history.append({"role": "assistant", "content": text})
    return stats


# =============================================================================
# Output handler
# =============================================================================

class Out:
    """Stream LLM tokens to stdout and feed sentence chunks to Mouth (TTS)."""

    def __init__(self, sentences=None, mouth=None):
        self.sentences = sentences
        self.mouth     = mouth
        self.at_start  = True
        self.interrupted = False

    def text(self, piece):
        if self.interrupted:
            return
        if self.at_start:
            print("kiwi> ", end="")
            self.at_start = False
        print(piece, end="", flush=True)
        if self.mouth and not self.interrupted:
            for sentence in self.sentences.feed(piece):
                if self.interrupted:
                    break
                self.mouth.say(sentence)

    def tool(self, name, args, result):
        if self.interrupted:
            return
        if not self.at_start:
            print()
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"  [tool] {name}({shown}) -> {result[:160]}")
        self.at_start = True

    def finish(self):
        if self.interrupted:
            self.reset()
            return
        if not self.at_start:
            print()
        self.at_start = True
        if self.mouth and not self.interrupted:
            rest = self.sentences.flush()
            if rest:
                self.mouth.say(rest)

    def reset(self):
        self.at_start = True
        self.interrupted = True
        if self.sentences:
            self.sentences.flush()


# =============================================================================
# Kiwi assistant state machine
# =============================================================================

class KiwiAssistant:
    """State machine: STANDBY (waiting for wake word) ↔ ACTIVE (conversing)."""

    def __init__(self, ear=None, mouth=None, sentences=None, push_to_talk=False,
                 idle_timeout=IDLE_TIMEOUT):
        self.ear           = ear
        self.mouth         = mouth
        self.sentences     = sentences
        self.push_to_talk  = push_to_talk
        self.idle_timeout  = idle_timeout
        self.out           = Out(sentences, mouth)
        self.history       = [{"role": "system", "content": dynamic_system_prompt()}]
        self.state         = "ACTIVE" if push_to_talk else "STANDBY"
        self.last_active   = time.time()
        self.pending_confirmation = None   # safety gate for destructive actions
        self._interrupt_event = threading.Event()

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _trim_history(self):
        """Refresh the system prompt and prune old messages."""
        self.history[0]["content"] = dynamic_system_prompt()
        rest = self.history[1:][-MAX_HISTORY:]
        while rest and rest[0]["role"] != "user":
            rest.pop(0)
        self.history = [self.history[0]] + rest

    # ------------------------------------------------------------------
    # Speech helpers & Interruption / Barge-in
    # ------------------------------------------------------------------

    def _interrupt(self):
        """Immediately cut off speech and cancel in-flight generation."""
        self._interrupt_event.set()
        if self.mouth:
            self.mouth.stop()
        self.out.reset()

    def _wait_speech(self):
        """Wait for mouth to finish speaking with multi-channel interruption.

        Interruption channels:
        1. Keyboard: any keypress in terminal (Enter, Space, Esc, s, q)
        2. Desktop shortcut: SIGUSR1 signal (e.g. pkill -USR1 -f jarvis.py) or /tmp/kiwi_stop
        3. Standard Ctrl+C
        """
        if not self.mouth:
            return

        global _SIGUSR1_TRIGGERED

        # Brief pause to let background worker pick up queued sentence
        time.sleep(0.03)
        if not self.mouth.is_busy():
            return

        try:
            with TerminalNonBlocking() as term:
                while self.mouth.is_busy():
                    # 1. Keyboard interrupt (any key in terminal)
                    if term.check_key():
                        print("\n[Stopped via keyboard]", flush=True)
                        self._interrupt()
                        break

                    # 2. Desktop shortcut trigger (SIGUSR1 or /tmp/kiwi_stop)
                    if _SIGUSR1_TRIGGERED:
                        _SIGUSR1_TRIGGERED = False
                        print("\n[Stopped via shortcut signal]", flush=True)
                        self._interrupt()
                        break
                    if os.path.exists(STOP_FILE):
                        try:
                            os.remove(STOP_FILE)
                        except OSError:
                            pass
                        print("\n[Stopped via shortcut trigger]", flush=True)
                        self._interrupt()
                        break

                    time.sleep(0.02)
        except KeyboardInterrupt:
            self._interrupt()
            print("\n(interrupted)")
        finally:
            # Let ALSA hardware buffers fully drain and settle before mic capture opens
            time.sleep(0.08)

    def _speak(self, text):
        """Say text and wait for it to finish (interruptible, with automatic media ducking)."""
        if self.mouth:
            self._interrupt_event.clear()
            self.out.interrupted = False
            if AUTO_DUCKING:
                tools.duck_media()
            try:
                self.mouth.say(text)
                self._wait_speech()
            finally:
                if AUTO_DUCKING:
                    tools.unduck_media()

    def _go_to_sleep(self, farewell=None, silent=False):
        """Transition to STANDBY. silent=True skips TTS (e.g. after Spotify starts)."""
        from voice import play_cue
        if AUTO_DUCKING:
            tools.unduck_media()
        if silent:
            play_cue("sleep")
            self.state = "STANDBY"
        else:
            msg = farewell or random.choice(SLEEP_RESPONSES)
            print(f"kiwi> {msg}")
            play_cue("sleep")
            self._speak(msg)
            self.state = "STANDBY"

    # ------------------------------------------------------------------
    # Command processing
    # ------------------------------------------------------------------

    def process_command(self, query):
        """Route query through fast-path router or Ollama LLM."""
        auto_standby = is_spotify_play_trigger(query) and not self.push_to_talk

        # Safety gate: destructive action confirmation
        if self.pending_confirmation == "close_all":
            self.pending_confirmation = None
            if re.search(r"\b(yes|confirm|proceed|do it|sure|close all|close them)\b",
                         query.lower()):
                reply = tools.close_all_windows()
            else:
                reply = "Cancelled. Open windows remain untouched."
            print(f"kiwi> {reply}")
            self._speak(reply)
            return

        if re.search(r"\b(close all (windows|apps|applications)|close everything)\b", query.lower()):
            self.pending_confirmation = "close_all"
            msg = "Are you sure you want to close all open applications? Say 'yes' to confirm."
            print(f"kiwi> {msg}")
            self._speak(msg)
            return

        # Fast-path router
        fast_reply = router.dispatch(query)
        if fast_reply:
            print(f"kiwi> {fast_reply}")
            self._speak(fast_reply)
            self.history.append({"role": "user",      "content": query})
            self.history.append({"role": "assistant",  "content": fast_reply})
            self._trim_history()
            if auto_standby or ("spotify" in fast_reply.lower()):
                self._go_to_sleep(silent=True)
                time.sleep(2.0)   # let music audio settle before mic reopens
            return

        # Ollama LLM turn
        mark = len(self.history)
        self.history.append({"role": "user", "content": query})
        if AUTO_DUCKING:
            tools.duck_media()
        try:
            self._interrupt_event.clear()
            self.out.interrupted = False
            stats = run_turn(self.history, self.out, query, cancel_event=self._interrupt_event)
            self.out.finish()
            if self.mouth:
                self._wait_speech()
            count = (stats or {}).get("eval_count")
            nanos = (stats or {}).get("eval_duration")
            if count and nanos:
                print(f"  [{count / (nanos / 1e9):.0f} tok/s]")
        except KeyboardInterrupt:
            print("\n(interrupted)")
            self._interrupt()
            del self.history[mark:]
        except OllamaError as e:
            print(f"\n[error] {e}")
            self._interrupt()
            del self.history[mark:]
        finally:
            if AUTO_DUCKING:
                tools.unduck_media()
        self._trim_history()
        if auto_standby or any(
            c.get("function", {}).get("name") == "spotify_play"
            for m in self.history[-3:]
            for c in m.get("tool_calls", [])
        ):
            self._go_to_sleep(silent=True)
            time.sleep(2.0)   # let music audio settle before mic reopens

    # ------------------------------------------------------------------
    # Voice loop steps
    # ------------------------------------------------------------------

    def run_hands_free_step(self):
        """Single iteration of the hands-free voice loop."""
        from voice import play_cue

        if self.state == "STANDBY":
            print("\r[Standby - Say 'Hey Kiwi', 'Hello Kiwi', or 'Kiwi']  ", end="", flush=True)
            audio = self.ear.record_vad(silence_duration=0.8, max_seconds=5.0)
            if audio.size == 0:
                return
            text = self.ear.transcribe(audio).strip()
            if not text:
                return
            remainder = extract_wake_remainder(text)
            if remainder is None:
                return   # wake phrase not heard

            print(f"\n[Awake: heard '{text}']")
            self.state       = "ACTIVE"
            self.last_active = time.time()
            if AUTO_DUCKING:
                tools.duck_media()
            play_cue("wake")

            if remainder:
                print(f"you> {remainder}")
                # Check for chained command + standby (e.g. "open spotify and standby")
                m = STANDBY_SUFFIX.search(remainder)
                if m:
                    sub_cmd = remainder[:m.start()].strip(" ,:?.!")
                    if sub_cmd:
                        self.process_command(sub_cmd)
                    self._go_to_sleep()
                else:
                    self.process_command(remainder)
                    # process_command already handles auto-standby + 2s delay for Spotify
            else:
                ack = random.choice(WAKE_RESPONSES)
                print(f"kiwi> {ack}")
                self._speak(ack)
            return

        # ACTIVE — listen and respond
        print("\n[Listening... speak freely]", flush=True)

        def on_speech_start():
            if self.mouth and self.mouth.is_speaking():
                self.mouth.stop()

        audio = self.ear.record_vad(
            silence_duration=0.5, max_seconds=8.0, on_speech_start=on_speech_start)
        if audio.size == 0:
            if time.time() - self.last_active > self.idle_timeout:
                self._go_to_sleep()
            return

        text = self.ear.transcribe(audio).strip()
        if not text:
            return

        print(f"you> {text}")
        self.last_active = time.time()

        if is_sleep_trigger(text):
            self._go_to_sleep()
            return

        # Check for chained command + standby (e.g. "can you open spotify and standby")
        m = STANDBY_SUFFIX.search(text)
        if m:
            sub_cmd = text[:m.start()].strip(" ,:?.!")
            if sub_cmd:
                self.process_command(sub_cmd)
            self._go_to_sleep()
            return

        self.process_command(text)
        # process_command sets state=STANDBY automatically for Spotify — just
        # add a mic-silence buffer so the music doesn't bleed into the next VAD window.
        if self.state == "STANDBY":
            time.sleep(2.0)

    def run_push_to_talk_step(self):
        """Single iteration of the push-to-talk loop."""
        prompt = "\nyou (Enter = talk)> " if self.ear else "\nyou> "
        line = input(prompt).strip()

        if not line and self.ear:
            print("  listening (press Enter when done talking)...", flush=True)
            audio = self.ear.record(input)
            if audio.size == 0:
                print("  (didn't catch anything)")
                return

            import numpy as np
            rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
            peak = float(np.abs(audio).max()) if audio.size else 0.0
            if rms < 0.004 or peak < 0.02:
                print("  (no speech detected)")
                return

            print("  [transcribing...]", flush=True)
            line = self.ear.transcribe(audio).strip()
            if not line:
                print("  (didn't catch anything)")
                return
            print(f"  you said: {line}")

        if not line:
            return

        # Slash commands
        if line in ("/quit", "/exit"):
            sys.exit(0)
        if line == "/help":
            print(HELP)
            return
        if line == "/reset":
            self.history = [{"role": "system", "content": dynamic_system_prompt()}]
            print("(conversation cleared)")
            return
        if line == "/tools":
            for name, entry in tools.REGISTRY.items():
                desc = entry["schema"]["function"]["description"].split(".")[0]
                print(f"  {name:20s}  {desc}")
            return

        # Check for chained command + standby
        m = STANDBY_SUFFIX.search(line)
        if m:
            sub_cmd = line[:m.start()].strip(" ,:?.!")
            if sub_cmd:
                self.process_command(sub_cmd)
            self._go_to_sleep()
            return

        self.process_command(line)


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    global MODEL
    parser = argparse.ArgumentParser(description="Kiwi: local AI assistant")
    parser.add_argument("--voice",  action="store_true",
                        help="hands-free mode with 'Hey Kiwi' or 'Hello Kiwi' wake word")
    parser.add_argument("--speak",  action="store_true",
                        help="type text input, hear spoken replies")
    parser.add_argument("--ptt", "--push-to-talk", dest="push_to_talk", action="store_true",
                        help="push-to-talk mode (press Enter to speak)")
    parser.add_argument("--model",  default=MODEL,
                        help=f"Ollama model (default: {MODEL})")
    parser.add_argument("--stt-model", default=DEFAULT_STT,
                        help=f"Whisper model: tiny.en, base.en, small.en, medium.en (default: {DEFAULT_STT})")
    parser.add_argument("--voice-file", default=DEFAULT_VOICE,
                        help=f"Piper voice .onnx file path (default: {DEFAULT_VOICE})")
    parser.add_argument("--tts-speed", type=float, default=DEFAULT_SPEED,
                        help=f"TTS speech speed (< 1.0 = faster, > 1.0 = slower, default: {DEFAULT_SPEED})")
    parser.add_argument("--idle-timeout", type=float, default=IDLE_TIMEOUT,
                        help=f"Seconds of silence before auto-sleep (default: {IDLE_TIMEOUT})")
    args = parser.parse_args()
    MODEL = args.model

    ear = mouth = sentences = None
    if args.voice or args.speak:
        from voice import Ear, Mouth, SentenceBuffer, VoiceError
        try:
            mouth     = Mouth(args.voice_file, speed=args.tts_speed)
            sentences = SentenceBuffer()
            if args.voice:
                ear = Ear(args.stt_model)
        except VoiceError as e:
            sys.exit(f"[voice setup failed] {e}")

    # Wire timer announcements into the voice output
    def on_timer(message):
        print(f"\n  [timer] {message}", flush=True)
        if mouth:
            mouth.say(message)
    tools.announce_hook = on_timer

    threading.Thread(target=warm_up, daemon=True).start()

    mode_label = (
        "voice (hands-free: 'Hey Kiwi' / 'Hello Kiwi' / 'Kiwi')" if (ear and not args.push_to_talk)
        else "voice (push-to-talk)"        if ear
        else "speak"                        if mouth
        else "text"
    )
    print(f"Kiwi  |  model: {MODEL}  |  {len(tools.REGISTRY)} tools  |  mode: {mode_label}")
    print(HELP)

    assistant = KiwiAssistant(
        ear=ear,
        mouth=mouth,
        sentences=sentences,
        push_to_talk=args.push_to_talk or (not ear),
        idle_timeout=args.idle_timeout,
    )

    while True:
        try:
            if ear and not args.push_to_talk:
                assistant.run_hands_free_step()
            else:
                assistant.run_push_to_talk_step()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exiting Kiwi]")
            break


if __name__ == "__main__":
    sys.exit(main())
