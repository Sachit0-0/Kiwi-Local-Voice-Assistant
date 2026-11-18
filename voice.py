"""Voice I/O for Jarvis: hands-free / push-to-talk speech-to-text and streaming TTS.

Ear   = faster-whisper (GPU/CPU), domain biasing, confidence filtering, speech normalization
Mouth = Kokoro-82M (state-of-the-art neural TTS) / Piper TTS with streaming playback and responsive cancellation
"""

import collections
import os
import queue
import re
import threading
import time

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono

# ---------------------------------------------------------------------------
# Whisper accuracy: domain-specific bias prompt + post-processing
# ---------------------------------------------------------------------------

# This prompt primes Whisper's decoder to favour our domain vocabulary.
# Natural sentences are used instead of word/number lists to prevent repetition loops.
WHISPER_BIAS_PROMPT = (
    "Hey Kiwi, Yo Kiwi, Hello Kiwi, Okay Kiwi, Kiwi. "
    "Please turn up the volume, turn down the volume, mute the audio, or set volume to fifty percent. "
    "Make the screen brighter, dimmer, or set brightness to seventy percent. "
    "Turn on night light, turn off night light, warm display, cool display. "
    "Minimize all windows, show desktop, restore windows, close all windows. "
    "Open Spotify and play music, pause playback, resume, next track, what song is playing. "
    "What is the weather in Kathmandu, what time is it, system status, CPU and RAM usage, battery level. "
    "Open Firefox browser, open Konsole terminal, open Dolphin file manager, open calculator, open VS Code. "
    "Take a screenshot, lock screen, set a timer for five minutes, save note, Wi-Fi status. "
    "Thank you Kiwi, stop listening, go to sleep, standby, enter standby."
)

# Acoustic confusions: Whisper systematically mishears certain words in our domain.
# Each (pattern, replacement) is applied in order after transcription.
_ACOUSTIC_FIXES = [
    # --- "turn on/off" mishearings ---
    (r"\b(?:tron|drawn|drone|torn|tone|thorn)\s+(on|off|up|down)\b",   r"turn \1"),
    (r"\b(?:call\s+is\s+)?(?:drawn|drone)\s+on\b",                    "turn on"),

    # --- Night light ---
    (r"\bnightlife\b",                       "night light"),
    (r"\bnight\s*lie\b",                     "night light"),

    # --- App / website names ---
    (r"\bcalico(?:\.com)?\b",                "calculator"),
    (r"\bspot\s*(?:of\s*(?:eye|I)|a\s*fly|if\s*I)\b", "spotify"),
    (r"\bdis\s*chord\b",                     "discord"),

    # --- "find / open" before a domain ---
    (r"\bfund\s+([a-zA-Z0-9-]+\.(?:com|org|net|io|edu|gov))\b", r"open \1"),
    (r"\bfund\b",                            "find"),

    # --- Spotify / music ---
    (r"\bvote\s+on\s+spotify\b",             "play spotify"),
    (r"\bplay\s+on\s+song\b",               "play a song"),
    (r"\bplay\s+your\s+song\b",             "play a song"),
    (r"\bplay\s+you\s+a\s+song\b",          "play a song"),
    (r"\bopen\s+and\s+play\b",              "open and play"),

    # --- Volume ---
    (r"\bfollow\s*me\b(?=.*(?:volume|up|down|loud|quiet))", "volume"),

    # --- Location ---
    (r"\bcut[- ]and[- ]do\b",               "Kathmandu"),
    (r"\bcat\s*man\s*do\b",                  "Kathmandu"),

    # --- Misc ---
    (r"\bserum\b",                           "sir"),
    (r"\boh\s*pen\b",                        "open"),
    (r"\bwhether\s+(in|for|today|forecast)",  r"weather \1"),
    (r"\bpaid softs\b",                      "best shops"),

    # --- Stop talking fixes ---
    (r"\bstop\s+(?:talk\s*in|talken|talkin)\b", "stop talking"),
    (r"\bstop\s+it\s+talking\b",               "stop talking"),

    # --- Kiwi / name mishearings ---
    (r"\b(?:qiwi|keewee|keywee|kee[- ]wee|kewee|kwee|key\s*we)\b", "Kiwi"),
    (r"\b(?:Travis|Jarves|Jarbus|Jarvus|Jervis)\b",               "Kiwi"),

    # --- "Hi Matthew" / random names Whisper hallucinates on silence ---
    (r"^(?:Hi|Hello)\s+(?:Matthew|Matteo|Mathieu)[.,!?]*$", ""),
]

# Common YouTube / subtitle outro hallucinations produced on silent/noisy audio
_HALLUCINATION_PATTERNS = [
    re.compile(r"thank\s+you\s+(very\s+much\s+)?for\s+watching", re.IGNORECASE),
    re.compile(r"thanks\s+for\s+watching", re.IGNORECASE),
    re.compile(r"that'?s\s+it\s+for\s+this\s+video", re.IGNORECASE),
    re.compile(r"see\s+you\s+(in\s+the\s+)?next\s+time", re.IGNORECASE),
    re.compile(r"see\s+you\s+next\s+time", re.IGNORECASE),
    re.compile(r"hope\s+you\s+enjoyed\s+(this\s+)?video", re.IGNORECASE),
    re.compile(r"like\s+and\s+subscribe", re.IGNORECASE),
    re.compile(r"subtitles\s+by", re.IGNORECASE),
    re.compile(r"translated\s+by", re.IGNORECASE),
    re.compile(r"^(?:bye[- ]?bye|goodbye)[.!]?$", re.IGNORECASE),
]

# Runaway token/phrase loops (e.g. "90%, 90%, 90%...")
_REPETITION_PATTERN = re.compile(r"([^\s,]+(?:\s+[^\s,]+)?)(?:[,\s]+\1){3,}", re.IGNORECASE)

# Compile once for speed
_ACOUSTIC_FIXES_COMPILED = [
    (re.compile(pat, re.IGNORECASE), repl) for pat, repl in _ACOUSTIC_FIXES
]


def normalize_speech_text(text):
    """Post-process Whisper output: drop hallucinations and fix acoustic confusions."""
    if not text:
        return ""
    t = text.strip()

    # Drop known YouTube/subtitle hallucinations on silence
    for pat in _HALLUCINATION_PATTERNS:
        if pat.search(t):
            return ""

    # Drop runaway repetition loops (e.g. "90%, 90%, 90%..." or "turn off, turn off, turn off...")
    if _REPETITION_PATTERN.search(t):
        return ""

    for pattern, replacement in _ACOUSTIC_FIXES_COMPILED:
        t = pattern.sub(replacement, t)
    return t.strip()


class VoiceError(Exception):
    pass


def _import_sounddevice():
    try:
        import sounddevice
        # Route through Pulse/PipeWire software mixer to eliminate ALSA mmap buffer collisions
        devices = sounddevice.query_devices()
        dev_names = [d["name"].lower() for d in devices]
        if "pulse" in dev_names:
            sounddevice.default.device = "pulse"
        elif "pipewire" in dev_names:
            sounddevice.default.device = "pipewire"
    except OSError as e:
        raise VoiceError("PortAudio not found. Install it with: sudo pacman -S portaudio") from e
    except ImportError as e:
        raise VoiceError("Missing Python packages. Run: pip install -r requirements.txt") from e
    return sounddevice


def play_cue(kind="wake"):
    """Play a short, subtle audio cue (synchronous 50ms so ALSA never collides with speech)."""
    try:
        sd = _import_sounddevice()
        import numpy as np
        sr = 16000
        if kind == "wake":
            t = np.linspace(0, 0.025, int(sr * 0.025), endpoint=False)
            tone1 = 0.05 * np.sin(2 * np.pi * 520 * t)
            tone2 = 0.07 * np.sin(2 * np.pi * 780 * t)
            audio = np.concatenate([tone1, tone2]).astype(np.float32)
        elif kind == "sleep":
            t = np.linspace(0, 0.035, int(sr * 0.035), endpoint=False)
            f = np.linspace(580, 380, len(t))
            audio = (0.05 * np.sin(2 * np.pi * f * t)).astype(np.float32)
        else:
            return
        sd.play(audio, samplerate=sr, blocking=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Text helpers for TTS streaming
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")
_EMOJI = re.compile("[\U00010000-\U0010ffff☀-➿️]")


def clean_for_speech(text):
    """Strip markdown/URLs/emoji and expand units/symbols to natural spoken English."""
    if not text:
        return ""
    # URLs
    text = re.sub(r"https?://\S+", "", text)
    # Emoji
    text = _EMOJI.sub("", text)
    # Expand symbols and units for natural human pronunciation
    text = re.sub(r"\b(\d+)\s*°\s*C\b", r"\g<1> degrees Celsius", text)
    text = re.sub(r"\b(\d+)\s*°\s*F\b", r"\g<1> degrees Fahrenheit", text)
    text = re.sub(r"\b(\d+)\s*%", r"\g<1> percent", text)
    text = re.sub(r"\b(\d+)\s*km/h\b", r"\g<1> kilometers per hour", text)
    text = re.sub(r"\b(\d+)\s*mph\b", r"\g<1> miles per hour", text)
    text = re.sub(r"\b(\d+)\s*K\b", r"\g<1> Kelvin", text)
    text = re.sub(r"\$(\d+)", r"\g<1> dollars", text)
    text = re.sub(r"\b&amp;\b|\b&\b", " and ", text)
    text = re.sub(r"\bw/\b", "with", text)
    text = re.sub(r"\bw/o\b", "without", text)
    # Markdown formatting
    text = text.replace("_", " ")
    text = re.sub(r"[*`#>~]+", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceBuffer:
    """Collects streamed LLM tokens and hands back complete sentences / clauses for TTS.

    min_len=3 ensures sentence boundaries (. ! ?) are respected.
    clause_min_len=28 allows long clauses (, ;) to start speaking early,
    eliminating the awkward latency gap before Kiwi begins talking.
    """

    def __init__(self, min_len=3, clause_min_len=28):
        self.buf = ""
        self.min_len = min_len
        self.clause_min_len = clause_min_len

    def feed(self, text):
        self.buf += text
        sentences = []
        while True:
            # 1. Period / exclamation / question / newline boundary
            m_sent = _SENTENCE_END.search(self.buf)
            if m_sent and m_sent.start() >= self.min_len:
                chunk = self.buf[:m_sent.start()].strip()
                self.buf = self.buf[m_sent.end():]
                if chunk:
                    sentences.append(chunk)
                continue

            # 2. Clause boundary (, ; :) if enough characters have accumulated
            m_clause = _CLAUSE_END.search(self.buf)
            if m_clause and m_clause.start() >= self.clause_min_len:
                chunk = self.buf[:m_clause.start()].strip()
                self.buf = self.buf[m_clause.end():]
                if chunk:
                    sentences.append(chunk)
                continue

            break
        return sentences

    def flush(self):
        rest, self.buf = self.buf.strip(), ""
        return rest or None


# ---------------------------------------------------------------------------
# Speech-to-text  (faster-whisper)
# ---------------------------------------------------------------------------

class Ear:
    def __init__(self, model_size="medium.en", language="en"):
        self.sd = _import_sounddevice()
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise VoiceError("Missing Python packages. Run: pip install -r requirements.txt") from e

        # --- Try GPU (CUDA) first for dramatically faster & more accurate STT ---
        device = "cpu"
        compute_type = "int8"
        threads = min(os.cpu_count() or 4, 6)
        try:
            import ctranslate2
            if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
                device = "cuda"
                compute_type = "float16"
                threads = 1
                print(f"Loading speech model '{model_size}' on GPU (CUDA, float16)...", flush=True)
            else:
                print(f"Loading speech model '{model_size}' on CPU (int8, {threads} threads)...",
                      flush=True)
        except Exception:
            print(f"Loading speech model '{model_size}' on CPU (int8, {threads} threads)...",
                  flush=True)

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=threads,
            num_workers=1,
        )
        self._device = device
        self.language = language

        # Ultra-fast keyword spotter model (tiny.en, ~39MB) for real-time "stop talking" detection
        try:
            self.spotter = WhisperModel(
                "tiny.en",
                device="cpu",
                compute_type="int8",
                cpu_threads=min(os.cpu_count() or 4, 4),
                num_workers=1,
            )
        except Exception:
            self.spotter = self.model

    def check_stop_keyword(self, audio):
        """Spot 'stop talking' phrase (<0.25s) to interrupt speech on explicit user request."""
        if audio.size < SAMPLE_RATE * 0.20:
            return None
        try:
            segments, _ = self.spotter.transcribe(
                audio,
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
                language="en",
                initial_prompt="stop talking.",
            )
            raw_text = " ".join(s.text for s in segments).strip()
            norm = normalize_speech_text(raw_text)
            if re.search(r"\bstop\s+talking\b", norm, re.IGNORECASE) or re.search(r"\bstop\s+talking\b", raw_text, re.IGNORECASE):
                return norm or raw_text
        except Exception:
            pass
        return None

    def record_vad(self, silence_duration=0.7, max_seconds=8.0, on_speech_start=None):
        """Record with ambient noise calibration and automatic end-of-speech detection."""
        import numpy as np

        pre_roll = collections.deque(maxlen=7)   # ~220 ms pre-roll buffer
        captured_frames = []
        speech_started = False
        silence_start = None
        start_time = time.time()
        block_size = 512

        ambient_samples = []

        def callback(indata, frame_count, time_info, status):
            block = indata.copy()
            if not speech_started:
                pre_roll.append(block)
            else:
                captured_frames.append(block)

        try:
            with self.sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                     dtype="float32", blocksize=block_size,
                                     callback=callback):
                # Calibrate ambient noise level (~150 ms)
                calib_start = time.time()
                while time.time() - calib_start < 0.15:
                    time.sleep(0.02)
                    if pre_roll:
                        ambient_samples.append(
                            float(np.sqrt(np.mean(pre_roll[-1] ** 2))))

                ambient_level = (float(np.mean(ambient_samples))
                                 if ambient_samples else 0.005)
                # Threshold: 2.5× ambient (was 2.2; slightly higher rejects more noise)
                threshold = max(ambient_level * 2.5, 0.010)

                while time.time() - start_time < max_seconds:
                    time.sleep(0.025)
                    if not speech_started:
                        if pre_roll:
                            rms = float(np.sqrt(np.mean(pre_roll[-1] ** 2)))
                            if rms > threshold:
                                speech_started = True
                                captured_frames.extend(list(pre_roll))
                                if on_speech_start:
                                    on_speech_start()
                    else:
                        if captured_frames:
                            rms = float(np.sqrt(np.mean(captured_frames[-1] ** 2)))
                            if rms > threshold * 0.75:
                                silence_start = None
                            else:
                                if silence_start is None:
                                    silence_start = time.time()
                                elif time.time() - silence_start >= silence_duration:
                                    break
        except self.sd.PortAudioError as e:
            raise VoiceError(f"Microphone problem: {e}") from e

        if not captured_frames or not speech_started:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(captured_frames)[:, 0]

    def record(self, wait_for_stop):
        """Record until wait_for_stop() returns (push-to-talk)."""
        import numpy as np
        frames = []

        def callback(indata, frame_count, time_info, status):
            frames.append(indata.copy())

        try:
            with self.sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                     dtype="float32", callback=callback):
                wait_for_stop()
        except self.sd.PortAudioError as e:
            raise VoiceError(f"Microphone problem: {e}") from e
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames)[:, 0]

    def transcribe(self, audio):
        """float32 mono 16 kHz audio → text.

        Pipeline:
        1. DC offset removal (common laptop mic issue)
        2. Peak normalization to 0.95
        3. Whisper greedy decode (beam_size=1) for speed
        4. Confidence filtering: drop hallucinated / garbage segments
        5. Acoustic normalization: fix domain-specific mis-hearings
        """
        import numpy as np

        # Minimum 300 ms of audio (was 250 ms — too short catches clicks)
        if audio.size < SAMPLE_RATE * 0.30:
            return ""

        # --- Audio preprocessing ---
        # Remove DC offset (laptop mics often have a DC bias)
        audio = audio - np.mean(audio)

        # Check RMS and peak energy to reject ambient noise / silence
        # Real speech into a laptop mic has peak >= 0.02 and rms >= 0.004
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.abs(audio).max())
        if peak < 0.02 or rms < 0.004:
            return ""   # pure silence, fan hum, or key clicks

        # Safe gain: scale quiet audio towards 0.95 peak, but CAP gain to 4.0x
        # so ambient hiss/fan noise is never boosted into false loud speech
        gain = min(0.95 / peak, 4.0)
        audio = (audio * gain).astype(np.float32)

        # --- Transcribe with Silero VAD to reject non-speech audio ---
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=1,                            # greedy decode — fastest
            temperature=0.0,                        # deterministic decoding
            initial_prompt=WHISPER_BIAS_PROMPT,
            vad_filter=True,                        # Silero VAD filters out noise chunks
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=400,
                speech_pad_ms=300,
            ),
            condition_on_previous_text=False,        # each utterance is independent
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,         # reject hallucinated repetitions
        )

        # --- Confidence filtering ---
        # Drop segments that Whisper is uncertain about — these are usually
        # hallucinated text from background noise, fan hum, or Jarvis's own
        # speech leaking back through the mic.
        texts = []
        for seg in segments:
            # Very low confidence → almost certainly garbage
            if seg.avg_logprob < -1.0:
                continue
            # High probability that this segment is not speech at all
            if seg.no_speech_prob > 0.6:
                continue
            t = seg.text.strip()
            if t:
                texts.append(t)

        raw_text = " ".join(texts).strip()
        return normalize_speech_text(raw_text)


# ---------------------------------------------------------------------------
# Text-to-speech  (Kokoro-82M / Piper)
# ---------------------------------------------------------------------------

def _split_clauses(text, target_len=55):
    """Split text into conversational phrase chunks for low-latency TTS synthesis."""
    if len(text) <= target_len:
        return [text]
    parts = re.split(r"(?<=[,;:\n])\s+", text)
    if len(parts) <= 1:
        parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) <= 1:
        return [text]

    chunks = []
    curr = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not curr:
            curr = part
        elif len(curr) + len(part) + 1 <= target_len:
            curr += " " + part
        else:
            chunks.append(curr)
            curr = part
    if curr:
        chunks.append(curr)
    return chunks


class Mouth:
    """Speaks queued sentences on a background thread with clean PortAudio stream lifecycle.
    
    Supports:
    - Kokoro-82M (engine='kokoro'): State-of-the-art neural TTS, 24kHz, human cadence.
    - Piper (engine='piper'): Lightweight fast ONNX TTS.
    """

    def __init__(self, voice_path=None, engine="kokoro", kokoro_voice="af_heart",
                 kokoro_model="voices/kokoro-v1.0.onnx", kokoro_voices_file="voices/voices-v1.0.bin",
                 speed=1.0):
        self.engine = (engine or "kokoro").lower()
        self.speed = speed
        self.kokoro_voice = kokoro_voice or "af_heart"
        self.kokoro_model = kokoro_model
        self.kokoro_voices_file = kokoro_voices_file
        self.voice_path = voice_path
        self.sd = _import_sounddevice()
        self._queue = queue.Queue()
        self._cancel = threading.Event()
        self._is_speaking = threading.Event()
        self._current_stream = None
        self._stream_lock = threading.Lock()

        # Check if Kokoro is requested
        if self.engine == "kokoro":
            if not os.path.exists(kokoro_model) or not os.path.exists(kokoro_voices_file):
                # Fallback to piper if piper voice file exists
                if voice_path and os.path.exists(voice_path):
                    print(f"  [Kokoro model missing, falling back to Piper: {voice_path}]", flush=True)
                    self.engine = "piper"
                else:
                    raise VoiceError(
                        f"Kokoro model or voices file not found:\n"
                        f"  Model: {kokoro_model}\n"
                        f"  Voices: {kokoro_voices_file}"
                    )

        if self.engine == "kokoro":
            try:
                from kokoro_onnx import Kokoro
            except ImportError as e:
                raise VoiceError("Missing kokoro-onnx package. Run: pip install kokoro-onnx soundfile") from e
            self.kokoro = Kokoro(kokoro_model, kokoro_voices_file)
            self.rate = 24000
            self.dtype = "float32"
            # Pre-warm ONNX session so first spoken response starts instantly
            try:
                self.kokoro.create("warm", voice=self.kokoro_voice, speed=1.0, lang="en-us")
            except Exception:
                pass
        else:
            if not voice_path or not os.path.exists(voice_path):
                raise VoiceError(
                    f"Piper voice file not found: {voice_path}\n"
                    "Download one with:\n"
                    "  python -m piper.download_voices en_GB-alan-medium --download-dir voices"
                )
            try:
                from piper import PiperVoice
                from piper.config import SynthesisConfig
            except ImportError as e:
                raise VoiceError("Missing piper-tts package. Run: pip install piper-tts") from e
            self.voice = PiperVoice.load(voice_path)
            self.rate = self.voice.config.sample_rate
            self.dtype = "int16"
            self._syn_config = SynthesisConfig(
                length_scale=speed,
                noise_scale=0.667,
                noise_w_scale=0.8,
            )

        threading.Thread(target=self._worker, daemon=True).start()

    def is_speaking(self):
        """Returns True if speech playback is currently active."""
        return self._is_speaking.is_set()

    def is_busy(self):
        """Returns True if there is speech queued OR actively playing."""
        return self._is_speaking.is_set() or not self._queue.empty()

    def say(self, text):
        text = clean_for_speech(text)
        if text:
            self._cancel.clear()
            self._queue.put(text)

    def wait(self):
        """Block until everything queued has been spoken."""
        self._queue.join()

    def stop(self):
        """Cut off speech immediately, abort playback, and discard queued speech."""
        self._cancel.set()
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self._is_speaking.clear()

    def _worker(self):
        while True:
            text = self._queue.get()
            if not text or self._cancel.is_set():
                self._queue.task_done()
                if self._queue.empty():
                    self._cancel.clear()
                    self._is_speaking.clear()
                continue
            self._is_speaking.set()
            try:
                self._speak(text)
            except Exception as e:
                if not self._cancel.is_set():
                    print(f"\n[speech error] {e}", flush=True)
            finally:
                self._is_speaking.clear()
                self._queue.task_done()
                if self._queue.empty():
                    self._cancel.clear()

    def _play_array_callback(self, samples):
        """Play a numpy audio array via thread-safe non-blocking callback (no ALSA crashes on cancel)."""
        if self._cancel.is_set() or len(samples) == 0:
            return

        import threading
        pos = 0
        total_samples = len(samples)
        done = threading.Event()

        def callback(outdata, frames, time_info, status):
            nonlocal pos
            if self._cancel.is_set() or pos >= total_samples:
                outdata.fill(0)
                raise self.sd.CallbackStop()

            chunk = samples[pos:pos + frames]
            pos += len(chunk)
            if len(chunk) < frames:
                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):, 0] = 0
                raise self.sd.CallbackStop()
            else:
                outdata[:, 0] = chunk

        try:
            with self.sd.OutputStream(samplerate=self.rate, channels=1, dtype=self.dtype,
                                      callback=callback, finished_callback=done.set) as stream:
                with self._stream_lock:
                    self._current_stream = stream
                while not done.is_set():
                    if self._cancel.is_set():
                        break
                    done.wait(timeout=0.02)
        except self.sd.PortAudioError:
            pass
        finally:
            with self._stream_lock:
                self._current_stream = None

    def _speak(self, text):
        if self.engine == "kokoro":
            self._speak_kokoro(text)
        else:
            self._speak_piper(text)

    def _speak_kokoro(self, text):
        if self._cancel.is_set():
            return
        chunks = _split_clauses(text)
        for chunk in chunks:
            if self._cancel.is_set():
                return
            try:
                samples, _sr = self.kokoro.create(
                    chunk,
                    voice=self.kokoro_voice,
                    speed=self.speed,
                    lang="en-us",
                )
            except Exception as e:
                if not self._cancel.is_set():
                    print(f"\n[Kokoro synthesis error] {e}", flush=True)
                return

            if self._cancel.is_set() or len(samples) == 0:
                continue

            if samples.ndim > 1:
                samples = samples.flatten()

            self._play_array_callback(samples)

    def _speak_piper(self, text):
        if self._cancel.is_set():
            return
        try:
            import numpy as np
            chunks = []
            for chunk in self.voice.synthesize(text, syn_config=self._syn_config):
                if self._cancel.is_set():
                    return
                chunks.append(chunk.audio_int16_array)
            if chunks:
                audio = np.concatenate(chunks)
                self._play_array_callback(audio)
        except Exception as e:
            if not self._cancel.is_set():
                print(f"\n[Piper error] {e}", flush=True)
