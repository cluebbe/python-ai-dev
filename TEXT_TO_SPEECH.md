# Workshop: Text to Speech with Python and pyttsx3

---

## Introduction

### Background

Text-to-speech (TTS) is the process of converting written text into spoken
audio. It powers screen readers, voice assistants, accessibility tools, and
any application that needs to communicate through audio instead of a screen.

This workshop uses **pyttsx3**, a Python library that drives the platform's
built-in speech engine directly — no internet connection or API key is
required. On macOS it uses the built-in **NSSpeechSynthesizer**, on Windows
it uses **SAPI5**, and on Linux it uses **eSpeak**.

Because everything runs locally, responses are instant and work offline.

### Setting Up the Development Environment

<details>
<summary><b>Hint:</b> Installing Python 3.12 on Windows and macOS</summary>

All tutorials in this repository use **Python 3.12** — `requirements.txt` pins
`torch`, which has no wheels for Python 3.13+. Install 3.12 alongside any other
Python you already have; the virtual environment decides which one is used.

**Windows**

1. Download the latest **Python 3.12.x Windows installer (64-bit)** from
   [python.org/downloads/windows](https://www.python.org/downloads/windows/)
   — or run `winget install Python.Python.3.12` in a terminal.
2. In the installer, tick **Add python.exe to PATH**, then click **Install Now**.
   The other checkbox, **Use admin privileges when installing py.exe**, is
   optional: Python itself is installed for your user either way. Untick it if
   you have no admin rights on the machine.
3. On the last screen, click **Disable path length limit** if it is offered
   (needs admin rights once). `torch` and `transformers` install deeply nested
   files, and Windows' default 260-character path limit can make
   `pip install` fail. Without admin rights, keep the project in a short
   folder such as `C:\src\python-ai-dev` instead.
4. Open a *new* PowerShell window and check the version:
   ```powershell
   py -3.12 --version
   ```

Windows has no `python3.12` command — wherever the steps below use it, run
`py -3.12` instead. If activating the virtual environment in PowerShell reports
*"running scripts is disabled on this system"*, allow local scripts once for
your user and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS**

1. Install [Homebrew](https://brew.sh) if you don't have it yet. On Apple
   Silicon Macs (M1 and later), `which brew` should print
   `/opt/homebrew/bin/brew`.
2. Install Python 3.12 and check the version:
   ```bash
   brew install python@3.12
   python3.12 --version
   ```
   (The macOS installer from [python.org](https://www.python.org/downloads/macos/)
   works too.)

</details>

**1. Create and activate a virtual environment**

```bash
mkdir text_to_speech
cd text_to_speech

python3.12 -m venv venv     # Windows: py -3.12 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**2. Install system dependencies (if needed)**

| Platform | Command |
|---|---|
| macOS | None — uses built-in NSSpeechSynthesizer |
| Windows | None — uses built-in SAPI5 |
| Linux | `sudo apt-get install espeak` |

**3. Install Python dependencies**

All packages are pinned in `requirements.txt` to avoid version conflicts:

```bash
pip install -r requirements.txt
```

**4. Verify the installation**

```python
import pyttsx3
engine = pyttsx3.init()
print(engine.getProperty("rate"))    # default speaking rate
print(engine.getProperty("volume"))  # default volume
```

If this prints two numbers without errors, your environment is ready.

---

## Task 1 — Listing Voices

Write a function `list_voices(engine)` that prints every available voice on
the system together with its index number, using a `pyttsx3` engine instance.

Example output:
```
Available voices:
  [0] Alex — com.apple.speech.synthesis.voice.Alex
  [1] Samantha — com.apple.speech.synthesis.voice.samantha
```

<details>
<summary>Solution</summary>

```python
import pyttsx3

def list_voices(engine):
    """Print all available voices and their index numbers."""
    print("Available voices:")
    for index, voice in enumerate(engine.getProperty("voices")):
        print(f"  [{index}] {voice.name} — {voice.id}")
```

`engine.getProperty("voices")` returns a list of `Voice` objects. Each object
has a `.name` (human-readable label) and an `.id` (the system string used to
actually select the voice). `enumerate()` pairs each voice with a numeric
index that you can pass to `speak_once()` or `speak_loop()`.

</details>

---

## Task 2 — Single-Shot Speech

Write a function `speak_once(text, voice_index=None, rate=150, volume=1.0)` that:

1. Creates a `pyttsx3` engine instance.
2. Sets the speaking rate and volume from the arguments.
3. Switches to the specified voice if `voice_index` is not `None`.
4. Speaks the given text and blocks until playback is finished.
5. Releases the audio driver before returning.

<details>
<summary>Solution</summary>

```python
import pyttsx3

def speak_once(text, voice_index=None, rate=150, volume=1.0):
    engine = pyttsx3.init()

    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)

    if voice_index is not None:
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[voice_index].id)

    engine.say(text)
    engine.runAndWait()
    engine.stop()
```

**Key points:**
- `setProperty("rate", ...)` controls words per minute. The default is ~200;
  150 is a comfortable, slightly slower pace for clarity.
- `setProperty("volume", ...)` accepts a float from `0.0` (mute) to `1.0`
  (maximum).
- `engine.say()` only **queues** the text — no audio plays until
  `runAndWait()` is called.
- `engine.stop()` releases the audio driver so other processes (or a future
  `pyttsx3.init()` call) can use it.

</details>

---

## Task 3 — Controlling the Voice

Explain the difference between `engine.say()` and `engine.runAndWait()`.
Why must both be called to produce audio?

<details>
<summary>Solution</summary>

| Method | What it does |
|---|---|
| `engine.say(text)` | Adds the text to an internal queue — no audio is produced yet |
| `engine.runAndWait()` | Starts the event loop, plays all queued items, and blocks until done |

`pyttsx3` uses an event-driven model. `say()` schedules work; `runAndWait()`
executes it. This design lets you queue multiple phrases before playback
begins:

```python
engine.say("Hello.")
engine.say("How are you?")
engine.runAndWait()  # speaks both sentences back-to-back
```

Calling `say()` without `runAndWait()` produces no sound. Calling
`runAndWait()` without any prior `say()` returns immediately with no effect.

</details>

---

## Task 4 — Continuous Speech Loop

Write a function `speak_loop(voice_index=None, rate=150, volume=1.0)` that:

1. Repeatedly prompts the user to type text and speaks it aloud.
2. Skips empty input with a friendly message instead of trying to speak nothing.
3. Stops cleanly when the user types `"quit"`, `"stop"`, or `"exit"`.
4. Speaks each line with a **fresh** engine instance (via `speak_once()`)
   rather than reusing one instance across iterations.

<details>
<summary>Solution</summary>

```python
import pyttsx3

def speak_loop(voice_index=None, rate=150, volume=1.0):
    print("\nText-to-Speech loop. Type 'quit' or 'stop' to exit.\n")

    while True:
        text = input("Enter text to speak: ")

        if text.strip().lower() in ("quit", "stop", "exit"):
            print("Stopping.")
            break

        if not text.strip():
            print("(nothing typed, try again)")
            continue

        speak_once(text, voice_index=voice_index, rate=rate, volume=volume)
```

**Key points:**
- **Do not** reuse a single engine instance across iterations. On Windows
  (SAPI5) and macOS (NSSS), the driver's event loop only runs correctly once
  per engine instance — a second `runAndWait()` call on the same instance
  silently produces no audio, even though `say()` queues the text without
  error. Creating a new engine per utterance (by delegating to `speak_once()`)
  avoids this and works consistently across macOS, Windows, and Linux.
- **Watch out for the engine cache.** `pyttsx3.init()` returns the *existing*
  engine as long as any variable still references it, and only builds a new
  one once the old one has been garbage-collected. A single long-lived
  variable such as `_engine = pyttsx3.init()` in `__main__` is enough to make
  every `speak_once()` call reuse that engine, so only the first sentence is
  heard.
- `text.strip().lower()` removes surrounding whitespace and normalises case so
  `"Quit"`, `"QUIT"`, and `" quit "` all trigger the exit condition.
- The empty-input guard prevents `speak_once("")` from being called, which can
  behave unexpectedly depending on the platform driver.
- `speak_once()` already handles setting rate/volume/voice and releasing the
  audio driver (`engine.stop()`) after each utterance.

</details>

---

## Task 5 — Putting It All Together

Write a `__main__` block that:

1. Lists all available voices.
2. Runs a single-shot speech with a fixed example sentence.
3. Then starts the continuous speech loop.

<details>
<summary>Solution</summary>

```python
if __name__ == "__main__":
    print("=== Text-to-Speech Tutorial ===\n")

    # List all available voices
    _engine = pyttsx3.init()
    list_voices(_engine)
    del _engine  # release it, or every later pyttsx3.init() reuses this engine

    # --- Single-shot mode ---
    print("--- Single-shot mode ---")
    speak_once("Hello! This is a text to speech demonstration.")

    # --- Continuous mode ---
    print("\n--- Continuous mode ---")
    speak_loop()
```

The `if __name__ == "__main__"` guard ensures this block only runs when the
script is executed directly — not when it is imported as a module by another
script.

</details>
