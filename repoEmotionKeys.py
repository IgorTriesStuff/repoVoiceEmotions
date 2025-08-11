import json
import os
import queue
import re
import sys
import time
import threading
from pathlib import Path
from contextlib import suppress

# ---- Ensure libvosk.dll is discoverable in PyInstaller onefile mode ----
def prime_dll_search():
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    for p in (base, base / "vosk", base / "lib", base / "bin"):
        if p.exists():
            with suppress(Exception):
                os.add_dll_directory(str(p))
prime_dll_search()

import sounddevice as sd
from vosk import Model, KaldiRecognizer
from pynput.keyboard import Controller
from PIL import Image, ImageDraw          # pip install pillow
import pystray                             # pip install pystray

# ======================
# Runtime paths & config
# ======================
CONFIG_FILENAME = "mood_config.json"

def runtime_dir() -> Path:
    # Folder to look for mood_config.json at runtime (next to .exe for onefile)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

def config_path() -> Path:
    return runtime_dir() / CONFIG_FILENAME

# ======================
# Audio settings
# ======================
INPUT_DEVICE = None
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
# Mutable runtime state (protected by locks)
# ======================
CONFIG_LOCK = threading.Lock()   # protects config-derived structures
STATE_LOCK  = threading.Lock()   # protects per-mood cooldowns

# Config-derived globals (initialized from defaults, then possibly replaced by config)
COOLDOWN_SEC_PER_MOOD = DEFAULT_COOLDOWN
REQUIRE_CONTEXT_PHRASE = DEFAULT_REQUIRE_CONTEXT
MOOD_TO_KEY = {m: d["key"] for m, d in DEFAULT_MOODS.items()}
WORD_TO_MOOD = {w.lower(): mood for mood, d in DEFAULT_MOODS.items() for w in d["synonyms"]}
WORD_REGEX = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, WORD_TO_MOOD.keys()), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

keyboard = Controller()
last_mood_press_time = {mood: 0.0 for mood in MOOD_TO_KEY.keys()}

# Tray globals
LISTEN_EVENT = threading.Event(); LISTEN_EVENT.set()
QUIT_EVENT = threading.Event()
ICON = None

# ======================
# Config loading & applying
# ======================
def apply_config(cfg: dict) -> None:
    """Validate and apply config atomically."""
    global COOLDOWN_SEC_PER_MOOD, REQUIRE_CONTEXT_PHRASE, MOOD_TO_KEY, WORD_TO_MOOD, WORD_REGEX, last_mood_press_time

    cooldown = cfg.get("cooldown_sec_per_mood", DEFAULT_COOLDOWN)
    require_context = cfg.get("require_context_phrase", DEFAULT_REQUIRE_CONTEXT)
    moods_cfg = cfg.get("moods", {})

    if not isinstance(moods_cfg, dict) or not moods_cfg:
        moods_cfg = DEFAULT_MOODS

    # Build new maps
    new_mood_to_key = {}
    new_word_to_mood = {}

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

    if not new_mood_to_key or not new_word_to_mood:
        # Fallback to defaults if config produced nothing valid
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
        keep = last_mood_press_time if 'last_mood_press_time' in globals() else {}
        last_mood_press_time = {m: keep.get(m, 0.0) for m in MOOD_TO_KEY.keys()}

def load_config_from_disk() -> None:
    """Load JSON config from mood_config.json if present, else apply defaults."""
    path = config_path()
    if not path.exists():
        apply_config({"moods": DEFAULT_MOODS,
                      "cooldown_sec_per_mood": DEFAULT_COOLDOWN,
                      "require_context_phrase": DEFAULT_REQUIRE_CONTEXT})
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        apply_config(cfg)
    except Exception as e:
        # On error, fall back to defaults but notify
        apply_config({"moods": DEFAULT_MOODS,
                      "cooldown_sec_per_mood": DEFAULT_COOLDOWN,
                      "require_context_phrase": DEFAULT_REQUIRE_CONTEXT})
        if ICON:
            with suppress(Exception):
                ICON.notify(f"Failed to load {CONFIG_FILENAME}: {e}")

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
    keyboard.press(key_str); keyboard.release(key_str)

def maybe_trigger(text: str):
    if not text or not LISTEN_EVENT.is_set():
        return
    txt = text.lower()
    with CONFIG_LOCK:
        regex = WORD_REGEX
        word_to_mood = WORD_TO_MOOD.copy()
        require_ctx = REQUIRE_CONTEXT_PHRASE
    if require_ctx and not re.search(r"\b(i\s*(am|m)\b|i\s*feel(ing)?\b)", txt, flags=re.IGNORECASE):
        return
    for m in regex.finditer(txt):
        mood = word_to_mood.get(m.group(1).lower())
        if mood:
            press_key_for_mood(mood)

# ======================
# Recognition worker
# ======================
def recognizer_thread():
    try:
        model_dir = find_model_dir()
    except FileNotFoundError:
        if ICON:
            with suppress(Exception):
                ICON.notify("Vosk model not found.\nPut it into 'models' next to the EXE.")
        QUIT_EVENT.set()
        return

    try:
        model = Model(model_dir)
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        rec.SetWords(False)
        audio_q: "queue.Queue[bytes]" = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                pass
            audio_q.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            device=INPUT_DEVICE,
            dtype="int16",
            channels=CHANNELS,
            callback=audio_callback,
        ):
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
        if ICON:
            with suppress(Exception):
                ICON.notify(f"Audio error: {e}\nCheck mic device/permissions.")
        QUIT_EVENT.set()

# ======================
# Tray UI
# ======================
def make_icon_image(on=True):
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
    else:
        LISTEN_EVENT.set()
        icon.icon = ICON_ON
        icon.title = "REPO Emotion Keys — Listening"

def tray_quit(icon, _item):
    QUIT_EVENT.set()
    icon.stop()

def dynamic_label(_icon):
    return "Pause listening" if LISTEN_EVENT.is_set() else "Enable listening"

def build_menu():
    return pystray.Menu(
        pystray.MenuItem(dynamic_label, tray_toggle),
        pystray.MenuItem("Reload config", reload_config),
        pystray.MenuItem("Quit", tray_quit)
    )

# ======================
# Main
# ======================
def main():
    global ICON
    load_config_from_disk()

    t_rec = threading.Thread(target=recognizer_thread, daemon=True)
    t_rec.start()

    ICON = pystray.Icon("repo_emotion_keys", ICON_ON,
                        "REPO Emotion Keys — Listening", build_menu())
    ICON.run()

    QUIT_EVENT.set()
    try:
        t_rec.join(timeout=1.0)
    except Exception:
        pass

if __name__ == "__main__":
    main()
