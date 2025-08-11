
# REPO Emotion Keys

Mic → mood word → simulated number key.  
A tiny tray app that listens offline for mood words (e.g., “angry”, “happy”) and presses the mapped number key in the **currently focused** window (your game, OBS, etc.). Runs fully offline with Vosk.

# Disclaimer
Written with some help from AI. Sorry for possible errors or misleads.

# What it does

-   Listens to your microphone.
    
-   Detects target words **inside normal speech** (“I’m kinda angry about that”).
    
-   Types a single number key per detected mood (with a short cooldown).
    
-   Lives in the **system tray** with: **Pause/Resume**, **Reload config**, **Quit**.
    

# Default word → key mapping

-   **angry** → `5`
    
-   **sad** → `6`
    
-   **suspicious** → `7`
    
-   **sleepy** → `8`
    
-   **surprised** → `9`
    
-   **happy** → `0`
    

Default synonyms (case-insensitive, all editable via config):

-   angry: `angry, mad, furious, irritated, pissed, upset`
    
-   happy: `happy, glad, joyful, cheerful`
    
-   sleepy: `sleepy, tired, drowsy, sleepyhead, yawning`
    
-   sad: `sad, unhappy, down, blue`
    
-   suspicious: `suspicious, skeptical, doubtful, fishy, sus`
    
-   surprised: `surprised, shocked`
    

# Quick start (prebuilt EXE)

1.  Put `repoEmotionKeys.exe` and (optional) `mood_config.json` **in the same folder**.
    
2.  Double-click the EXE. A tray icon appears:
    
    -   **Green** = listening, **Red** = paused.
        
3.  Right-click the tray icon for the menu:
    
    -   **Pause/Enable listening**
        
    -   **Reload config**
        
    -   **Quit**
        

# Editing the config (no rebuild needed)

Create or edit a file named **`mood_config.json`** next to the EXE (or next to the `.py` if running from source). Then choose **Tray → Reload config**.

Minimal example:

json

    {
      "cooldown_sec_per_mood": 1.2,
      "require_context_phrase": false,
      "moods": {
        "angry":      { "key": "5", "synonyms": ["angry", "mad", "furious", "irritated", "pissed", "upset"] },
        "sad":        { "key": "6", "synonyms": ["sad", "unhappy", "down", "blue"] },
        "suspicious": { "key": "7", "synonyms": ["suspicious", "skeptical", "doubtful", "fishy", "sus"] },
        "sleepy":     { "key": "8", "synonyms": ["sleepy", "tired", "drowsy", "sleepyhead", "yawning"] },
        "surprised":  { "key": "9", "synonyms": ["surprised", "shocked"] },
        "happy":      { "key": "0", "synonyms": ["happy", "glad", "joyful", "cheerful"] }
      }
    }

Notes:

-   Add **new moods** by adding a new object under `"moods"` with `"key"` (single character) and `"synonyms"`.
    
-   Set `"require_context_phrase": true` to only trigger after phrases like “I’m / I am / I feel …”.
    

# Run from source (devs)

## Requirements

-   Python 3.10+
    
-   `pip install vosk pynput sounddevice pystray pillow`
    
-   **Vosk model** (offline, ~50–70 MB): download `vosk-model-small-en-us-0.15` and unzip to:
    
## Project Layout

 - project/
	 - repoEmotionKeys.py
	 - models/
		 - vosk-model-small-en-us-0.15/



## To Build

bash

py -m venv .venv
".venv\Scripts\activate"
pip install -U pip
pip install pyinstaller vosk pynput sounddevice pystray pillow

pyinstaller --onedir --console --clean -y --noupx ^
  --collect-binaries vosk --collect-data vosk ^
  --add-data "models\vosk-model-small-en-us-0.15;models\vosk-model-small-en-us-0.15" ^
  repoEmotionKeys.py


# Tray controls

-   **Enable/Disable listening** — toggles hot mic detection (useful on stream).
    
-   **Reload config** — reads `mood_config.json` again without restarting.
    
-   **Quit** — cleanly shuts down the mic and exits.
    

# Tips & troubleshooting

-   **Antivirus warnings:** portable EXEs that listen to mic & simulate keys may trigger heuristics. Consider shutting it down for a while.
    
-   **No key presses?** Make sure the game/window is **focused**. The app types into the active window.

    
# Privacy

-   All speech recognition is **offline** (Vosk).
    
-   No internet access; nothing is uploaded.
    
-   The app only simulates number keys you configure.

# Third-Party Notices

- Vosk — Apache License 2.0 (license included in licenses/VOSK_LICENSE.txt).
- Vosk English model (vosk-model-small-en-us-0.15) — see licenses/VOSK_MODEL_LICENSE.txt.
- pynput, pystray, Pillow, python-sounddevice — see included licenses in licenses/. 
This tool includes software developed by the Apache Software Foundation (http://www.apache.org/).
