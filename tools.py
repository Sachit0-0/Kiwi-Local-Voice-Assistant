#!/usr/bin/env python3
"""PC control tools for Kiwi (Linux, KDE Plasma, PipeWire).

The model can only *request* a tool. Every request is validated against the
tool's schema before anything runs, and no tool accepts a free-form shell
command.

Test a tool by hand, without the LLM:
    python tools.py volume '{"action": "up"}'
    python tools.py current_time
    python tools.py system_status
"""

import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))

@lru_cache(maxsize=64)
def _which(exe):
    """Cached binary lookup in PATH to avoid repeated filesystem scans."""
    return shutil.which(exe)

# Ensure DBUS_SESSION_BUS_ADDRESS is present for qdbus6 on systemd user sessions
if "DBUS_SESSION_BUS_ADDRESS" not in os.environ:
    uid = os.getuid()
    bus_socket = f"/run/user/{uid}/bus"
    if os.path.exists(bus_socket):
        os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_socket}"

REGISTRY = {}          # tool name -> {"fn": callable, "schema": ollama tool schema}
announce_hook = None   # jarvis.py sets this; called with a message when a timer fires

INSTALL_HINTS = {
    "wpctl":        "sudo pacman -S wireplumber",
    "playerctl":    "sudo pacman -S playerctl",
    "brightnessctl": "sudo pacman -S brightnessctl",
    "wmctrl":       "sudo pacman -S wmctrl",
    "notify-send":  "sudo pacman -S libnotify",
    "wl-copy":      "sudo pacman -S wl-clipboard",
    "nmcli":        "sudo pacman -S networkmanager",
}


# =============================================================================
# Registry, validation, and dispatch
# =============================================================================

class ToolError(Exception):
    """A problem worth reporting back to the model (and the user)."""


def tool(description, properties=None, required=()):
    """Decorator: register a function as a tool with an Ollama-compatible schema."""
    def register(fn):
        REGISTRY[fn.__name__] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties or {},
                        "required": list(required),
                    },
                },
            },
        }
        return fn
    return register


def schemas():
    """Return all registered tool schemas (for Ollama API)."""
    return [entry["schema"] for entry in REGISTRY.values()]


def _coerce(name, value, spec):
    """Cast a model-supplied value to the schema type, or raise ToolError."""
    kind = spec.get("type")
    if kind == "integer":
        if isinstance(value, bool):
            raise ToolError(f"'{name}' must be a whole number")
        if isinstance(value, str):
            value = value.strip().rstrip("%").strip()
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ToolError(f"'{name}' must be a whole number, got {value!r}")
        if number != int(number):
            raise ToolError(f"'{name}' must be a whole number, got {value!r}")
        number = int(number)
        low, high = spec.get("minimum"), spec.get("maximum")
        if (low is not None and number < low) or (high is not None and number > high):
            raise ToolError(f"'{name}' must be between {low} and {high}, got {number}")
        return number
    if kind == "string":
        if not isinstance(value, str) or not value.strip():
            raise ToolError(f"'{name}' must be a non-empty string")
        value = value.strip()
        if "enum" in spec:
            normal = value.lower().replace(" ", "_").replace("-", "_")
            if normal not in spec["enum"]:
                raise ToolError(f"'{name}' must be one of: {', '.join(spec['enum'])}")
            return normal
        if len(value) > 200:
            raise ToolError(f"'{name}' is too long")
        return value
    return value


def call_tool(name, arguments):
    """Validate and run a tool. Always returns a string; never raises."""
    entry = REGISTRY.get(name)
    if entry is None:
        return f"Error: no tool called '{name}'. Available: {', '.join(REGISTRY)}"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return "Error: arguments must be a JSON object."
    if not isinstance(arguments, dict):
        return "Error: arguments must be a JSON object."

    params = entry["schema"]["function"]["parameters"]
    clean = {}
    try:
        for key in params["required"]:
            if arguments.get(key) is None:
                raise ToolError(f"missing required argument '{key}'")
        for key, spec in params["properties"].items():
            if arguments.get(key) is not None:
                clean[key] = _coerce(key, arguments[key], spec)
        return str(entry["fn"](**clean))
    except ToolError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {name} failed unexpectedly ({e})"


def run(cmd, timeout=5, allow_failure=False):
    """Run cmd (list, no shell). Returns stdout or raises ToolError.

    Set allow_failure=True to return empty string instead of raising on
    non-zero exit (useful when probing for optional features).
    """
    exe = cmd[0]
    if _which(exe) is None:
        hint = INSTALL_HINTS.get(exe, "")
        msg = f"{exe} is not installed" + (f" (install with: {hint})" if hint else "")
        if allow_failure:
            return ""
        raise ToolError(msg)
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ToolError(f"{exe} timed out")
    if done.returncode != 0 and not allow_failure:
        detail = (done.stderr or done.stdout).strip() or f"exit code {done.returncode}"
        raise ToolError(f"{exe} failed: {detail}")
    return done.stdout.strip()


def _launch(*cmd):
    """Fire-and-forget: launch a process detached from Jarvis."""
    subprocess.Popen(list(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


# =============================================================================
# Volume  (PipeWire via wpctl)
# =============================================================================

SINK = "@DEFAULT_AUDIO_SINK@"


def _volume_text():
    """Return current volume as a human-readable string, e.g. '65% (muted)'."""
    out = run(["wpctl", "get-volume", SINK])   # "Volume: 0.50" or "Volume: 0.50 [MUTED]"
    try:
        pct = f"{round(float(out.split()[1]) * 100)}%"
    except (IndexError, ValueError):
        return out
    return pct + (" (muted)" if "[MUTED]" in out else "")


@tool(
    "Change the laptop's speaker volume or mute state. "
    "Actions: set (needs amount 0-100), up, down (amount is the step, default 10), "
    "mute, unmute, toggle_mute.",
    {
        "action": {"type": "string",
                   "enum": ["set", "up", "down", "mute", "unmute", "toggle_mute"]},
        "amount": {"type": "integer", "minimum": 0, "maximum": 100,
                   "description": "Percent: target level for 'set', or step for 'up'/'down'"},
    },
    required=["action"],
)
def volume(action, amount=None):
    """Control speaker volume."""
    if action == "set":
        if amount is None:
            raise ToolError("'set' needs an amount between 0 and 100")
        run(["wpctl", "set-mute", SINK, "0"])
        run(["wpctl", "set-volume", "-l", "1.0", SINK, f"{amount}%"])
    elif action in ("up", "down"):
        step = 10 if amount is None else amount
        sign = "+" if action == "up" else "-"
        if action == "up":
            run(["wpctl", "set-mute", SINK, "0"])
        run(["wpctl", "set-volume", "-l", "1.0", SINK, f"{step}%{sign}"])
    elif action == "mute":
        run(["wpctl", "set-mute", SINK, "1"])
    elif action == "unmute":
        run(["wpctl", "set-mute", SINK, "0"])
    else:  # toggle_mute
        run(["wpctl", "set-mute", SINK, "toggle"])
    return f"Volume is now {_volume_text()}."


# =============================================================================
# Media playback  (any MPRIS player via playerctl)
# =============================================================================

_PLAYERCTL_ACTIONS = {
    "play_pause": "play-pause",
    "play":       "play",
    "pause":      "pause",
    "next":       "next",
    "previous":   "previous",
    "stop":       "stop",
}


@tool(
    "Control playback on an external media player (Spotify, YouTube, VLC). "
    "Use ONLY when the user explicitly says play/pause/stop/skip/previous. "
    "Do NOT use on greetings or conversational prompts.",
    {"action": {"type": "string", "enum": list(_PLAYERCTL_ACTIONS)}},
    required=["action"],
)
def media(action):
    """Send a playerctl command to the active MPRIS player."""
    run(["playerctl", _PLAYERCTL_ACTIONS[action]])
    return f"Sent '{action}' to the media player."


# -----------------------------------------------------------------------------
# Audio Ducking (Auto-lower media volume during Kiwi dialogue)
# -----------------------------------------------------------------------------

_DUCKED_PLAYERS = {}

def duck_media(target_volume=0.20):
    """Lower volume of active media players (Spotify, browser, etc.) during assistant speech."""
    if _which("playerctl") is None:
        return
    try:
        players = subprocess.check_output(
            ["playerctl", "-l"], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
        for p in players:
            if p and p not in _DUCKED_PLAYERS:
                try:
                    vol_str = subprocess.check_output(
                        ["playerctl", "-p", p, "volume"], text=True, stderr=subprocess.DEVNULL
                    ).strip()
                    vol = float(vol_str)
                    _DUCKED_PLAYERS[p] = vol
                    subprocess.run(
                        ["playerctl", "-p", p, "volume", str(target_volume)],
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass
    except Exception:
        pass


def unduck_media():
    """Restore media players back to their pre-ducking volume."""
    if _which("playerctl") is None or not _DUCKED_PLAYERS:
        return
    try:
        for p, original_vol in list(_DUCKED_PLAYERS.items()):
            try:
                subprocess.run(
                    ["playerctl", "-p", p, "volume", str(original_vol)],
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        _DUCKED_PLAYERS.clear()
    except Exception:
        pass


@tool("Get information about the song or audio track currently playing.")
def media_info():
    """Query playerctl for now-playing metadata."""
    if _which("playerctl") is None:
        raise ToolError("playerctl is not installed")
    player = run(["playerctl", "metadata", "--format", "{{playerName}}"], timeout=2,
                 allow_failure=True).strip().lower()
    try:
        title  = run(["playerctl", "metadata", "title"],  timeout=2)
        artist = run(["playerctl", "metadata", "artist"], timeout=2)
        # Detect actual source
        if "youtube" in title.lower():
            source = " on YouTube"
        elif player:
            source = f" on {player.capitalize()}"
        else:
            source = ""
        if title and artist:
            return f"Currently playing '{title}' by {artist}{source}."
        if title:
            return f"Currently playing '{title}'{source}."
    except ToolError:
        pass
    status = run(["playerctl", "status"], timeout=2, allow_failure=True)
    return f"Media playback is {status.lower()}." if status else "No media is currently playing."


@tool("Launch Spotify if it is not open, and start or resume playback.")
def spotify_play():
    """Open Spotify (native or Flatpak) and start playback."""
    # Gather active MPRIS players
    players = []
    if _which("playerctl"):
        try:
            players = subprocess.check_output(
                ["playerctl", "-l"], text=True, stderr=subprocess.DEVNULL
            ).splitlines()
        except Exception:
            pass

    # Pause competing non-Spotify players (browser tabs, VLC, etc.)
    for p in players:
        if "spotify" not in p.lower():
            subprocess.run(["playerctl", "-p", p, "pause"], stderr=subprocess.DEVNULL)

    # If Spotify is already on MPRIS, just play
    sp = next((p for p in players if "spotify" in p.lower()), None)
    if sp:
        run(["playerctl", "-p", sp, "play"])
        return "Playing Spotify."

    # Launch Spotify (native binary preferred, then Flatpak)
    launched = False
    for cmd in APPS.get("spotify", [["spotify"], ["flatpak", "run", "com.spotify.Client"]]):
        if _which(cmd[0]):
            _launch(*cmd)
            launched = True
            break
    if not launched:
        raise ToolError("Spotify is not installed (tried native and Flatpak)")

    # Wait in background for Spotify's MPRIS service to register, then play
    def _delayed_play():
        for _ in range(25):        # up to ~7.5 s
            time.sleep(0.3)
            try:
                pls = subprocess.check_output(
                    ["playerctl", "-l"], text=True, stderr=subprocess.DEVNULL
                ).splitlines()
                sp2 = next((p for p in pls if "spotify" in p.lower()), None)
                if sp2:
                    time.sleep(0.5)
                    subprocess.run(["playerctl", "-p", sp2, "play"], stderr=subprocess.DEVNULL)
                    return
            except Exception:
                pass
        # Last-ditch attempt
        if _which("playerctl"):
            subprocess.run(["playerctl", "play"], stderr=subprocess.DEVNULL)

    threading.Thread(target=_delayed_play, daemon=True).start()
    return "Opening Spotify and starting playback."


# =============================================================================
# Brightness  (backlight via brightnessctl)
# =============================================================================

def _brightness_text():
    """Return current brightness percent, e.g. '45%', or None on failure."""
    try:
        return run(["brightnessctl", "-m"]).split(",")[3]   # device,class,cur,45%,max
    except (ToolError, IndexError):
        return None


@tool(
    "Change the laptop screen brightness. "
    "Actions: set (needs amount 1-100), up, down (amount is the step, default 10).",
    {
        "action": {"type": "string", "enum": ["set", "up", "down"]},
        "amount": {"type": "integer", "minimum": 1, "maximum": 100,
                   "description": "Percent: target for 'set', or step for 'up'/'down'"},
    },
    required=["action"],
)
def brightness(action, amount=None):
    """Adjust screen backlight brightness."""
    step = 10 if amount is None else amount
    if action == "set":
        if amount is None:
            raise ToolError("'set' needs an amount between 1 and 100")
        value = f"{amount}%"
    else:
        value = f"{step}%{'+' if action == 'up' else '-'}"
    run(["brightnessctl", "--min-value=1", "set", value])   # never go fully black
    level = _brightness_text()
    return f"Brightness is now {level}." if level else "Brightness changed."


# =============================================================================
# Night Light / Night Color  (KDE Plasma)
# =============================================================================

@tool("Toggle KDE Night Color (blue light filter).")
def night_light(action="toggle"):
    """Toggle KDE Night Color using native KDE global shortcut."""
    run(["qdbus6", "org.kde.kglobalaccel", "/component/kwin",
         "org.kde.kglobalaccel.Component.invokeShortcut", "Toggle Night Color"])
    return "Toggled Night Color."


# =============================================================================
# App launcher
# =============================================================================

# name -> list of candidate commands; first installed is used.
APPS = {
    "browser":        [["firefox"], ["cachy-browser"], ["chromium"], ["brave"],
                       ["google-chrome-stable"], ["zen-browser"]],
    "terminal":       [["konsole"], ["kitty"], ["alacritty"]],
    "files":          [["dolphin"], ["nautilus"]],
    "settings":       [["systemsettings"]],
    "calculator":     [["kcalc"], ["gnome-calculator"]],
    "editor":         [["kate"], ["gedit"]],
    "vscode":         [["code"], ["code-oss"], ["vscodium"]],
    "system monitor": [["plasma-systemmonitor"], ["ksysguard"]],
    "spotify":        [["spotify"], ["flatpak", "run", "com.spotify.Client"]],
    "vlc":            [["vlc"], ["flatpak", "run", "org.videolan.VLC"]],
    "discord":        [["discord"], ["vesktop"], ["flatpak", "run", "com.discordapp.Discord"]],
    "steam":          [["steam"], ["flatpak", "run", "com.valvesoftware.Steam"]],
    "teams":          [["teams-for-linux"],
                       ["flatpak", "run", "com.github.IsmaelMartinez.teams_for_linux"]],
    "onlyoffice":     [["onlyoffice-desktopeditors"],
                       ["flatpak", "run", "org.onlyoffice.desktopeditors"]],
    "gimp":           [["gimp"], ["flatpak", "run", "org.gimp.GIMP"]],
    "obs":            [["obs"], ["flatpak", "run", "com.obsproject.Studio"]],
}

ALIASES = {
    # Browser
    "web browser": "browser", "internet": "browser",
    "firefox": "browser", "chrome": "browser", "chromium": "browser",
    # Terminal
    "konsole": "terminal", "console": "terminal", "shell": "terminal", "kitty": "terminal",
    # Files
    "file manager": "files", "dolphin": "files", "folders": "files",
    # Settings
    "system settings": "settings", "preferences": "settings",
    # Calculator
    "calc": "calculator", "kcalc": "calculator",
    # Editor
    "text editor": "editor", "kate": "editor", "gedit": "editor",
    # VSCode
    "code": "vscode", "visual studio code": "vscode", "vscodium": "vscode",
    # System monitor
    "task manager": "system monitor", "activity monitor": "system monitor",
    # Media
    "music": "spotify", "music player": "spotify",
    "movie player": "vlc", "media player": "vlc", "video player": "vlc",
}


def _load_user_apps():
    """Optional apps.json override next to this file: {"myapp": ["myapp-binary"]}"""
    path = os.path.join(HERE, "apps.json")
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            extra = json.load(f)
        for name, cmd in extra.items():
            APPS[name.lower()] = [cmd] if cmd and isinstance(cmd[0], str) else cmd
    except (OSError, ValueError, TypeError, AttributeError) as e:
        print(f"[warning] ignoring apps.json: {e}", file=sys.stderr)


_load_user_apps()


@tool(
    "Open an application on the laptop. Known apps: " + ", ".join(sorted(APPS)) + ".",
    {"name": {"type": "string",
              "description": "App to open, e.g. browser, terminal, files, vscode, spotify"}},
    required=["name"],
)
def open_app(name):
    """Launch a known application by name or alias."""
    key = ALIASES.get(name.lower().strip(), name.lower().strip())
    if key not in APPS:
        raise ToolError(f"I don't know an app called '{name}'. I can open: {', '.join(sorted(APPS))}")
    for cmd in APPS[key]:
        if _which(cmd[0]):
            _launch(*cmd)
            return f"Opened {key}."
    raise ToolError(f"No installed program found for '{key}'")


@tool(
    "Open a website URL in the default web browser.",
    {"url": {"type": "string",
             "description": "URL or domain, e.g. 'spotify.com' or 'https://github.com'"}},
    required=["url"],
)
def open_url(url):
    """Open a URL in the default browser via xdg-open."""
    raw = url.strip()
    if not (raw.startswith("http://") or raw.startswith("https://")):
        raw = "https://" + raw
    _launch("xdg-open", raw)
    domain = urllib.parse.urlparse(raw).netloc or raw
    return f"Opened {domain} in browser."


@tool(
    "Search the web using Google in the default browser.",
    {"query": {"type": "string", "description": "Search keywords or query"}},
    required=["query"],
)
def web_search(query):
    """Open a Google search in the default browser."""
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query.strip())
    _launch("xdg-open", url)
    return f"Searching online for '{query}'."


@tool("Lock the laptop screen immediately.")
def lock_screen():
    """Lock the session via loginctl or KDE screenlocker."""
    _launch("loginctl", "lock-session")
    _launch("qdbus6", "org.kde.kscreenlocker", "/MainApplication", "lock")
    return "Screen locked."


@tool("Take a screenshot of the computer screen.")
def take_screenshot():
    """Capture the screen with spectacle."""
    _launch("spectacle", "-b", "-n")
    return "Screenshot captured."


# =============================================================================
# Desktop & Window Management
# =============================================================================

@tool("Show or hide the desktop by toggling all windows.")
def show_desktop(action="toggle"):
    """Show or hide the KDE desktop via native KDE global shortcut."""
    run(["qdbus6", "org.kde.kglobalaccel", "/component/kwin",
         "org.kde.kglobalaccel.Component.invokeShortcut", "Show Desktop"])
    return "Toggled desktop view."


def _gui_app_names():
    """Derive a flat list of executable names from the APPS dict (for pkill fallback)."""
    names = []
    for candidates in APPS.values():
        for cmd in candidates:
            names.append(cmd[0])
    return list(dict.fromkeys(names))   # deduplicated, insertion-ordered


@tool("Close all open user application windows gracefully (requires confirmation in jarvis.py).")
def close_all_windows():
    """Close all windows via wmctrl, falling back to SIGTERM on known GUI apps."""
    # 1. Preferred: tell each window to close via wmctrl
    if _which("wmctrl"):
        lines = run(["wmctrl", "-l"], allow_failure=True).splitlines()
        if lines:
            for line in lines:
                parts = line.split()
                if parts:
                    subprocess.run(["wmctrl", "-ic", parts[0]], timeout=1,
                                   stderr=subprocess.DEVNULL)
            return "Closed all application windows."

    # 2. SIGTERM on known GUI app executables (never system processes)
    closed = []
    for app in _gui_app_names():
        res = subprocess.run(["pkill", "-15", "-x", app], capture_output=True)
        if res.returncode == 0:
            closed.append(app)
    return f"Closed: {', '.join(closed)}." if closed else "No application windows found to close."


# =============================================================================
# Clipboard  (Wayland: wl-copy / wl-paste)
# =============================================================================

@tool(
    "Read from or write to the clipboard (Wayland).",
    {
        "action": {"type": "string", "enum": ["read", "write"],
                   "description": "'read' returns clipboard contents; 'write' sets them"},
        "text":   {"type": "string", "description": "Text to put in clipboard (write only)"},
    },
    required=["action"],
)
def clipboard(action, text=None):
    """Access the Wayland clipboard via wl-paste / wl-copy."""
    if action == "read":
        return run(["wl-paste", "--no-newline"])
    # write
    if not text:
        raise ToolError("'write' needs 'text'")
    proc = subprocess.run(["wl-copy"], input=text, text=True,
                           capture_output=True, timeout=5)
    if proc.returncode != 0:
        raise ToolError("wl-copy failed")
    return f"Copied {len(text)} characters to clipboard."


# =============================================================================
# WiFi / Network Status  (via nmcli)
# =============================================================================

@tool("Show current Wi-Fi or network connection status.")
def wifi_status():
    """Return the active network connection name, type, and signal strength."""
    out = run(["nmcli", "-t", "-f", "NAME,TYPE,STATE,SIGNAL", "device"])
    lines = [l for l in out.splitlines() if "connected" in l.lower()]
    if not lines:
        return "No active network connection found."
    info = []
    for line in lines[:2]:
        parts = line.split(":")
        name   = parts[0] if len(parts) > 0 else "?"
        kind   = parts[1] if len(parts) > 1 else "?"
        signal = parts[3] if len(parts) > 3 else ""
        sig_str = f" ({signal}% signal)" if signal.isdigit() else ""
        info.append(f"{name} ({kind}){sig_str}")
    return "Connected: " + ", ".join(info) + "."


# =============================================================================
# Quick Note  (append to ~/Documents/notes.md)
# =============================================================================

NOTES_FILE = os.path.expanduser("~/Documents/notes.md")


@tool(
    "Save a quick note to ~/Documents/notes.md.",
    {"text": {"type": "string", "description": "The note text to save"}},
    required=["text"],
)
def note(text):
    """Append a timestamped note to the user's notes file."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"- [{stamp}] {text.strip()}\n"
    os.makedirs(os.path.dirname(NOTES_FILE), exist_ok=True)
    with open(NOTES_FILE, "a") as f:
        f.write(entry)
    return f"Note saved: \"{text.strip()}\""


# =============================================================================
# Persistent Memory  (memory.json in project root)
# =============================================================================

MEMORY_FILE = os.path.join(HERE, "memory.json")


def _load_memories():
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_memories(data):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise ToolError(f"Failed to write memory: {e}")


@tool(
    "Store a fact or user preference in persistent memory to remember across sessions.",
    {
        "fact": {"type": "string", "description": "The fact or preference to remember"},
        "key":  {"type": "string", "description": "Short keyword or topic, e.g. 'birthday', 'music', 'name'"},
    },
    required=["fact"],
)
def remember(fact, key=None):
    """Store a fact in memory."""
    memories = _load_memories()
    k = (key or fact).strip().lower()
    if len(k) > 40:
        k = k[:40]
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    memories[k] = {"fact": fact.strip(), "date": stamp}
    _save_memories(memories)
    return f"I will remember that: {fact.strip()}."


@tool(
    "Recall stored memories or facts that the user previously asked Kiwi to remember.",
    {"query": {"type": "string", "description": "Optional search term or keyword to look up"}},
)
def recall(query=None):
    """Retrieve memories from storage."""
    memories = _load_memories()
    if not memories:
        return "I don't have anything stored in memory yet."
    if query:
        q = query.strip().lower()
        matches = [m["fact"] for k, m in memories.items() if q in k or q in m["fact"].lower()]
        if matches:
            return "Here is what I remember: " + "; ".join(matches) + "."
        return f"I don't remember anything about '{query}'."
    all_facts = [m["fact"] for m in memories.values()]
    return "Here is what I have in memory: " + "; ".join(all_facts) + "."


@tool(
    "Remove a fact or topic from persistent memory.",
    {"key": {"type": "string", "description": "Topic or keyword to forget"}},
    required=["key"],
)
def forget(key):
    """Delete a memory from storage."""
    memories = _load_memories()
    k = key.strip().lower()
    matched = [name for name in memories if k in name or k in memories[name]["fact"].lower()]
    if not matched:
        return f"I don't have anything stored about '{key}'."
    for name in matched:
        del memories[name]
    _save_memories(memories)
    return f"Removed {len(matched)} item(s) from memory."


# =============================================================================
# Notifications  (KDE Plasma / system logs)
# =============================================================================

@tool("Check recent desktop notifications.")
def read_notifications():
    """Read recent notifications from KDE NotificationManager or journalctl."""
    try:
        out = run(["qdbus6", "org.kde.plasmashell", "/org/kde/NotificationManager",
                   "org.kde.NotificationManager.notifications"], timeout=2, allow_failure=True)
        if out:
            return f"Notifications: {out}"
    except Exception:
        pass
    try:
        out = run(["journalctl", "--user", "-u", "plasma-plasmashell", "-n", "15", "--no-pager"],
                  timeout=2, allow_failure=True)
        lines = [l for l in out.splitlines() if "notification" in l.lower()]
        if lines:
            return "Recent notifications: " + " | ".join(lines[-2:])
    except Exception:
        pass
    return "No unread notifications at the moment."


# =============================================================================
# Information: Weather, System Status, Battery, Time
# =============================================================================

@tool(
    "Get the current weather for a city or the user's current location. "
    "If no location is given, it auto-detects the local area.",
    {
        "location": {"type": "string",
                     "description": "City or location name, e.g. 'Kathmandu', 'London'"},
    },
)
def get_weather(location=None):
    """Fetch weather from wttr.in (no API key required)."""
    loc_q = urllib.parse.quote(location.strip()) if location and location.strip() else ""
    headers = {"User-Agent": "curl/7.88.0"}

    def _fetch(url):
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.read().decode()

    try:
        data = json.loads(_fetch(f"https://wttr.in/{loc_q}?format=j1"))
        cur  = data["current_condition"][0]
        city = (data.get("nearest_area", [{}])[0]
                    .get("areaName", [{}])[0]
                    .get("value", location or "your area"))
        desc     = cur["weatherDesc"][0]["value"]
        temp_c   = cur["temp_C"]
        feels_c  = cur["FeelsLikeC"]
        humidity = cur["humidity"]
        wind_km  = cur["windspeedKmph"]
        return (f"In {city}, it is currently {desc} and {temp_c}°C "
                f"(feels like {feels_c}°C), with {humidity}% humidity "
                f"and wind at {wind_km} km/h.")
    except Exception as primary_err:
        try:
            text = _fetch(f"https://wttr.in/{loc_q}?format=%l:+%C+%t+(feels+like+%f),+wind+%w")
            if text and "Unknown location" not in text:
                return f"Weather: {text.strip()}"
        except Exception:
            pass
        raise ToolError(f"Unable to fetch weather ({primary_err})")


@tool("Check the computer's CPU load, RAM usage, and GPU status.")
def system_status():
    """Read /proc stats and nvidia-smi to summarise system health."""
    parts = []

    # CPU (1-minute load average)
    try:
        with open("/proc/loadavg") as f:
            load = f.read().split()[0]
        cores = os.cpu_count() or 1
        parts.append(f"CPU load average {load} across {cores} cores")
    except Exception:
        pass

    # RAM
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                k, v = line.split(":", 1)
                mem[k.strip()] = int(v.split()[0])
        total = mem.get("MemTotal", 0) / 1_048_576
        avail = mem.get("MemAvailable", 0) / 1_048_576
        used  = total - avail
        parts.append(f"RAM {used:.1f} / {total:.1f} GB ({int(used / total * 100)}% used)")
    except Exception:
        pass

    # GPU
    if _which("nvidia-smi"):
        try:
            out = run(["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                       "--format=csv,noheader,nounits"], timeout=3)
            fields = [f.strip() for f in out.split(",")]
            if len(fields) >= 5:
                parts.append(f"GPU {fields[0]} at {fields[1]}°C ({fields[2]}% load, {fields[3]}/{fields[4]} MB VRAM)")
        except Exception:
            pass

    return (". ".join(parts) + ".") if parts else "System metrics unavailable."


POWER_SUPPLY_GLOB = "/sys/class/power_supply/BAT*"
_BATTERY_CACHE = {"time": 0.0, "value": "Battery status unknown"}


@tool("Get the laptop battery percentage and charging state.")
def battery_status(max_age=20):
    """Read battery capacity and status from /sys/class/power_supply (cached for 20s)."""
    now = time.time()
    if now - _BATTERY_CACHE["time"] < max_age and _BATTERY_CACHE["value"] != "Battery status unknown":
        return _BATTERY_CACHE["value"]
    for bat in sorted(glob.glob(POWER_SUPPLY_GLOB)):
        try:
            percent = open(os.path.join(bat, "capacity")).read().strip()
            state   = open(os.path.join(bat, "status")).read().strip().lower()
        except OSError:
            continue
        res = f"Battery is at {percent}% and {state}."
        _BATTERY_CACHE["time"] = now
        _BATTERY_CACHE["value"] = res
        return res
    raise ToolError("No battery found")


@tool("Get the current local date and time.")
def current_time():
    """Return a human-readable local date/time string."""
    return datetime.datetime.now().astimezone().strftime(
        "It is %A, %d %B %Y, %I:%M %p (%Z)."
    )


# =============================================================================
# Timers
# =============================================================================

ACTIVE_TIMERS: list = []


def format_duration(total_seconds):
    """Return a human-readable duration string, e.g. '2 minutes and 30 seconds'."""
    parts = []
    hours, rest = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    for value, unit in ((hours, "hour"), (minutes, "minute"), (seconds, "second")):
        if value:
            parts.append(f"{value} {unit}{'s' if value != 1 else ''}")
    return " and ".join(parts)


def _fire(label):
    """Called by threading.Timer when a countdown expires."""
    message = "Your timer is up." if label == "timer" else f"Your '{label}' timer is up."
    if _which("notify-send"):
        _launch("notify-send", "Kiwi", message)
    if announce_hook:
        announce_hook(message)


@tool(
    "Set a countdown timer. Provide minutes and/or seconds (e.g. 90 seconds, or 5 minutes).",
    {
        "minutes": {"type": "integer", "minimum": 0, "maximum": 1440},
        "seconds": {"type": "integer", "minimum": 0, "maximum": 86400},
        "label":   {"type": "string", "description": "Optional label, e.g. 'pasta'"},
    },
)
def set_timer(minutes=0, seconds=0, label="timer"):
    """Start a background countdown timer."""
    total = minutes * 60 + seconds
    if total <= 0:
        raise ToolError("Give a duration in minutes or seconds")
    if total > 86400:
        raise ToolError("Timers can be at most 24 hours")
    t = threading.Timer(total, _fire, args=(label.lower(),))
    t.daemon = True
    t.start()
    ACTIVE_TIMERS.append(t)
    return f"Timer set for {format_duration(total)}."


# =============================================================================
# CLI test harness
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools.py <tool> ['{json args}']")
        print("tools:", ", ".join(REGISTRY))
        sys.exit(1)
    print(call_tool(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "{}"))
