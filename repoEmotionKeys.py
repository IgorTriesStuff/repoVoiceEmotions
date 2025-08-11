"""
REPO Emotion Keys — mic → mood word → number key

- Offline speech recognition with Vosk (no network).
- System tray icon (Pause/Enable, Reload config, Test keypress, Open folder, Quit).
- External config file: mood_config.json (synonyms, key mapping, cooldown, device, debug).
- Robust logging to repoEmotionKeys.log for remote debugging.
- PyInstaller-friendly (onefile/onedir), including libvosk.dll search.

Build tips (Windows):
  py -m venv .venv
  ".venv\\Scripts\\activate"
  pip install -U pip
  pip install pyinstaller vosk pynput sounddevice pystray pillow
  pyinstaller --onedir --console --clean -y --noupx ^
    --collect-binaries vosk --collect-data vosk ^
    --add-data "models\\vosk-model-small-en-us-0.15;models\\vosk-model-small-en-us-0.15" ^
    repoEmotionKeys.py
"""

import json
import os
import queue
import re
import sys
import time
import threading
from pathlib import Path
from contextlib import suppress
from datetime import datetime

# ---------------------------
# Make libvosk.dll discoverable (PyInstaller onefile)
# ---------------------------
def prime_dll_search():
    """
    When bundled as onefile, PyInstaller extracts to sys._MEIPASS.
    Add common subfolders to the Windows DLL search path so libvosk.dll is found.
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    for p in (base, base / "vosk", base / "lib", base / "bin"):
        if p.exists():
            with suppress(Exception):
                os.add_dll_directory(str(p))

prime_dll_search()

import sounddevice as sd
from vosk import Model, KaldiRecognizer
from pynput.keyboard import Controller
from PIL import Image, ImageDraw
import pystray

# Optional beep (Windows only); fine if it fails elsewhere
with suppress(Exception):
    import winsound  # type: ignore

# ======================
# Runtime paths & files
# ======================
CONFIG_FILENAME = "mood_config.json"
LOG_FILENAME = "repoEmotionKeys.log"

def runtime_dir() -> Path:
    # Folder where the EXE/script lives (next to config and log)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

def config_path() -> Path:
    return runtime_dir() / CONFIG_FILENAME

def log_path() -> Path:
    return runtime_dir() / LOG_FILENAME

# --------------- logging ---------------
LOG_LOCK = threading.Lock()
def log(msg: str):
    try:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        with LOG_LOCK:
            with log_path().open("a", encoding="utf-8") as f:
                f.write(line)
        # Also print to console if available (use console build while debugging)
        print(line, end="")
    except Exception:
        pass

# ======================
# Audio settings
# ======================
INPUT_DEVICE = None  # can be overridden by config "input_device_index"
SAMPLE_RATE = 16000
BLOCKSIZE = 8000
CHANNELS = 1

# ======================
# Defaults (used if config missing/invalid)
# ======================
DEFAULT_COOLDOWN = 1.2
DEFAULT_REQUIRE_CONTEXT = False
DEFAULT_MOODS = {
    "angry":      {"key": "5", "synonyms": ["angry", "mad", "furious", "irritated", "pissed", "upset"]},
    "sad":        {"key": "6", "synonyms": ["sad", "unhappy", "down", "blue"]},
    "suspicious": {"key": "7", "synonyms": ["suspicious", "skeptical", "doubtful", "fishy", "sus"]},
    "sleepy":     {"key": "8", "synonyms": ["sleepy", "tired", "drowsy", "sleepyhead", "yawning"]},
    "surprised":  {"key": "9", "synonyms": ["surprised", "shocked"]},
    "happy":      {"key": "0", "synonyms": ["happy", "glad", "joyful", "cheerful"]},
}

# ======================
# Mutable runtime state
# ======================
CONFIG_LOCK = threading.Lock()
STATE_LOCK  = threading.Lock()

COOLDOWN_SEC_PER_MOOD = DEFAULT_COOLDOWN
REQUIRE_CONTEXT_PHRASE = DEFAULT_REQUIRE_CONTEXT
CONTEXT_REGEX = re.compile(r"\b(i\s*(am|m)\b|i\s*feel(ing)?\b)", re.IGNORECASE)

MOOD_TO_KEY = {m: d["key"] for m, d in DEFAULT_MOODS.items()}
WORD_TO_MOOD = {w.lower(): mood for mood, d in DEFAULT_MOODS.items() for w in d["synonyms"]}
WORD_REGEX = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, WORD_TO_MOOD.keys()), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

INPUT_DEVICE_INDEX = None  # from config
DEBUG_LOG_RECOG = False    # from config.debug.log_recognition
DEBUG_BEEP = False         # from config.debug.beep_on_trigger

keyboard = Controller()
last_mood_press_time = {mood: 0.0 for mood in MOOD_TO_KEY.keys()}

# Tray globals
LISTEN_EVENT = threading.Event(); LISTEN_EVENT.set()
QUIT_EVENT = threading.Event()
ICON = None  # keep a strong reference so tray icon isn't GC'ed

# ======================
# Config handling
# ======================
def apply_config(cfg: dict) -> None:
    """
    Validate and apply config atomically.
    Accepts keys: cooldown_sec_per_mood (float), require_context_phrase (bool),
                  input_device_index (int or null),
                  debug: { log_recognition: bool, beep_on_trigger: bool },
                  moods: { moodName: { key: "0-9", synonyms: [..] }, ... }
    """
    global COOLDOWN_SEC_PER_MOOD, REQUIRE_CONTEXT_PHRASE, MOOD_TO_KEY, WORD_TO_MOOD, WORD_REGEX
    global last_mood_press_time, INPUT_DEVICE_INDEX, DEBUG_LOG_RECOG, DEBUG_BEEP

    cooldown = cfg.get("cooldown_sec_per_mood", DEFAULT_COOLDOWN)
    require_context = cfg.get("require_context_phrase", DEFAULT_REQUIRE_CONTEXT)
    moods_cfg = cfg.get("moods", {})
    input_idx = cfg.get("input_device_index", None)
    debug_cfg = cfg.get("debug", {})

    # Build new maps from config (or fall back to defaults if invalid)
    new_mood_to_key = {}
    new_word_to_mood = {}
    if isinstance(moods_cfg, dict) and moods_cfg:
        for mood, entry in moods_cfg.items():
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("key", DEFAULT_MOODS.get(mood, {}).get("key", "")))
            synonyms = entry.get("synonyms", DEFAULT_MOODS.get(mood, {}).get("synonyms", []))
            if not key or not synonyms:
                continue
            new_mood_to_key[mood] = key
            for w in synonyms:
                if isinstance(w, str) and w.strip():
                    new_word_to_mood[w.lower().strip()] = mood

    # Fallback to defaults if config produced nothing valid
    if not new_word_to_mood:
        new_mood_to_key = {m: d["key"] for m, d in DEFAULT_MOODS.items()}
        new_word_to_mood = {w.lower(): m for m, d in DEFAULT_MOODS.items() for w in d["synonyms"]}

    new_regex = re.compile(
        r"\b(" + "|".join(sorted(map(re.escape, new_word_to_mood.keys()), key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )

    with CONFIG_LOCK:
        COOLDOWN_SEC_PER_MOOD = float(cooldown)
        REQUIRE_CONTEXT_PHRASE = bool(require_context)
        MOOD_TO_KEY = new_mood_to_key
        WORD_TO_MOOD = new_word_to_mood
        WORD_REGEX = new_regex

        # Reset cooldown map to include new moods
        last_mood_press_time = {m: last_mood_press_time.get(m, 0.0) for m in MOOD_TO_KEY.keys()}

        INPUT_DEVICE_INDEX = input_idx if isinstance(input_idx, int) else None
        DEBUG_LOG_RECOG = bool(debug_cfg.get("log_recognition", False))
        DEBUG_BEEP = bool(debug_cfg.get("beep_on_trigger", False))

    log(f"Config applied: cooldown={COOLDOWN_SEC_PER_MOOD}, require_context={REQUIRE_CONTEXT_PHRASE}, "
        f"device_index={INPUT_DEVICE_INDEX}, moods={list(MOOD_TO_KEY.keys())}")

def load_config_from_disk() -> None:
    """Load JSON config from mood_config.json if present, else apply defaults."""
    path = config_path()
    if not path.exists():
        log("Config not found; using defaults.")
        apply_config({"moods": DEFAULT_MOODS,
                      "cooldown_sec_per_mood": DEFAULT_COOLDOWN,
                      "require_context_phrase": DEFAULT_REQUIRE_CONTEXT})
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        apply_config(cfg)
        log(f"Loaded config: {path}")
    except Exception as e:
        log(f"Failed to load config ({path}): {e}; falling back to defaults.")
        apply_config({"moods": DEFAULT_MOODS,
                      "cooldown_sec_per_mood": DEFAULT_COOLDOWN,
                      "require_context_phrase": DEFAULT_REQUIRE_CONTEXT})

def reload_config(_icon=None, _item=None):
    load_config_from_disk()
    if ICON:
        with suppress(Exception):
            ICON.notify("Config reloaded")

# ======================
# Vosk model discovery
# ======================
def looks_like_vosk_model_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    names = {x.name.lower() for x in p.iterdir()}
    return ("conf" in names) or ("am" in names) or any(n.endswith(".conf") for n in names)

def find_model_dir() -> str:
    """
    Look for a Vosk model in:
      - VOSK_MODEL_DIR env var, or
      - 'models/vosk-model-small-en-us-0.15' next to the EXE/py, or
      - any single model-looking subfolder under 'models/'
    """
    env = os.environ.get("VOSK_MODEL_DIR")
    if env and Path(env).exists():
        return str(Path(env).resolve())

    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    candidates = [base / "models" / "vosk-model-small-en-us-0.15", base / "models"]
    for c in candidates:
        if c.is_dir():
            if looks_like_vosk_model_dir(c):
                return str(c.resolve())
            subs = [p for p in c.iterdir() if p.is_dir() and looks_like_vosk_model_dir(p)]
            if len(subs) == 1:
                return str(subs[0].resolve())
            if len(subs) > 1:
                preferred = sorted(
                    subs, key=lambda p: (("en" not in p.name.lower()),
                                         ("small" not in p.name.lower()),
                                         p.name.lower())
                )[0]
                return str(preferred.resolve())

    raise FileNotFoundError(
        "Vosk model not found. Put the unzipped model in 'models' next to the app "
        "(e.g., models/vosk-model-small-en-us-0.15), or set VOSK_MODEL_DIR."
    )

# ======================
# Key pressing & matching
# ======================
def press_key_for_mood(mood: str):
    now = time.time()
    with STATE_LOCK, CONFIG_LOCK:
        last = last_mood_press_time.get(mood, 0.0)
        cooldown = COOLDOWN_SEC_PER_MOOD
        key_str = MOOD_TO_KEY.get(mood)
        if not key_str:
            return
        if now - last < cooldown:
            return
        last_mood_press_time[mood] = now

    try:
        keyboard.press(key_str); keyboard.release(key_str)
        log(f"Trigger: {mood} -> key '{key_str}'")
        if DEBUG_BEEP and 'winsound' in globals():
            with suppress(Exception):
                winsound.Beep(880, 120)  # short beep on trigger
    except Exception as e:
        log(f"Error pressing key for {mood}: {e}")

def maybe_trigger(text: str):
    """Check recognized text and press the mapped key if mood words are found."""
    if not text or not LISTEN_EVENT.is_set():
        return
    txt = text.lower()
    with CONFIG_LOCK:
        regex = WORD_REGEX
        word_to_mood = WORD_TO_MOOD.copy()
        require_ctx = REQUIRE_CONTEXT_PHRASE
        log_rec = DEBUG_LOG_RECOG

    if log_rec:
        log(f"Heard: {txt}")
    if require_ctx and not CONTEXT_REGEX.search(txt):
        return

    for m in regex.finditer(txt):
        mood = word_to_mood.get(m.group(1).lower())
        if mood:
            press_key_for_mood(mood)

# ======================
# Recognition worker
# ======================
def dump_devices_to_log():
    try:
        devs = sd.query_devices()
        log("[Audio] Input devices (idx | name | max_in):")
        for idx, d in enumerate(devs):
            mi = d.get("max_input_channels", 0)
            if mi > 0:
                log(f"  {idx:2d} | {d.get('name')} | {mi}")
    except Exception as e:
        log(f"[Audio] Could not list devices: {e}")

def recognizer_thread():
    try:
        model_dir = find_model_dir()
        log(f"Using Vosk model: {model_dir}")
    except FileNotFoundError as e:
        log(str(e))
        if ICON:
            with suppress(Exception):
                ICON.notify("Vosk model not found. See log.")
        QUIT_EVENT.set()
        return

    try:
        model = Model(model_dir)
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        rec.SetWords(False)

        audio_q: "queue.Queue[bytes]" = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                log(f"[Audio status] {status}")
            audio_q.put(bytes(indata))

        with CONFIG_LOCK:
            device_index = INPUT_DEVICE_INDEX

        if device_index is not None:
            log(f"Opening mic device index: {device_index}")
        else:
            log("Opening default mic device")

        dump_devices_to_log()

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            device=device_index if device_index is not None else None,
            dtype="int16",
            channels=CHANNELS,
            callback=audio_callback,
        ):
            log("Mic stream started. Listening…")
            while not QUIT_EVENT.is_set():
                try:
                    data = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if rec.AcceptWaveform(data):
                    try:
                        text = json.loads(rec.Result()).get("text", "")
                    except json.JSONDecodeError:
                        text = ""
                    if text:
                        maybe_trigger(text)
                else:
                    try:
                        ptext = json.loads(rec.PartialResult()).get("partial", "")
                    except json.JSONDecodeError:
                        ptext = ""
                    if ptext:
                        maybe_trigger(ptext)
    except Exception as e:
        log(f"[Audio error] {e}")
        if ICON:
            with suppress(Exception):
                ICON.notify(f"Audio error: {e}")
        QUIT_EVENT.set()

# ======================
# Tray UI
# ======================
def make_icon_image(on=True):
    """Create a simple tray icon (green = listening, red = paused)."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([4, 4, 60, 60], radius=12, fill=(35, 35, 40, 255))
    fill = (0, 200, 0, 255) if on else (200, 60, 30, 255)
    d.ellipse([40, 40, 58, 58], fill=fill)
    d.text((10, 12), "R", fill=(230, 230, 235, 255))
    return img

ICON_ON = make_icon_image(True)
ICON_OFF = make_icon_image(False)

def tray_toggle(icon, _item):
    if LISTEN_EVENT.is_set():
        LISTEN_EVENT.clear()
        icon.icon = ICON_OFF
        icon.title = "REPO Emotion Keys — Paused"
        log("Paused listening via tray.")
    else:
        LISTEN_EVENT.set()
        icon.icon = ICON_ON
        icon.title = "REPO Emotion Keys — Listening"
        log("Enabled listening via tray.")

def tray_quit(icon, _item):
    log("Quit via tray.")
    QUIT_EVENT.set()
    icon.stop()

def tray_test_key(icon, _item):
    """Send a test '1' keystroke to the focused window (for quick sanity check)."""
    try:
        keyboard.press('1'); keyboard.release('1')
        log("Test key '1' sent.")
        if ICON:
            with suppress(Exception):
                ICON.notify("Test key '1' sent")
    except Exception as e:
        log(f"Test key error: {e}")

def tray_open_folder(icon, _item):
    """Open the runtime folder (where config & log live)."""
    try:
        os.startfile(str(runtime_dir()))
        log("Opened runtime folder.")
    except Exception as e:
        log(f"Open folder error: {e}")

def dynamic_label(_icon):
    # pystray passes the icon as the first arg
    return "Pause listening" if LISTEN_EVENT.is_set() else "Enable listening"

def build_menu():
    return pystray.Menu(
        pystray.MenuItem(dynamic_label, tray_toggle),
        pystray.MenuItem("Reload config", reload_config),
        pystray.MenuItem("Test keypress (1)", tray_test_key),
        pystray.MenuItem("Open folder", tray_open_folder),
        pystray.MenuItem("Quit", tray_quit),
    )

# ======================
# Main
# ======================
def main():
    global ICON
    # fresh log header
    with suppress(Exception):
        log_path().unlink()
    log("=== REPO Emotion Keys starting ===")
    load_config_from_disk()

    # Start recognition in background thread
    t_rec = threading.Thread(target=recognizer_thread, daemon=True)
    t_rec.start()

    # Tray MUST run on the main thread on Windows
    ICON = pystray.Icon("repo_emotion_keys", ICON_ON,
                        "REPO Emotion Keys — Listening", build_menu())
    ICON.run()

    # After tray exits, shut down background worker
    QUIT_EVENT.set()
    try:
        t_rec.join(timeout=1.0)
    except Exception:
        pass
    log("=== Exiting ===")

if __name__ == "__main__":
    main()
