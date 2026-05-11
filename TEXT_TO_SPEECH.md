# Exam: Text to Speech with Python and pyttsx3

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

**1. Create and activate a virtual environment**

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**2. Install Python dependencies**

```bash
pip install pyttsx3
```

**Platform notes:**

| Platform | Extra steps if needed |
|---|---|
| macOS | None — uses built-in NSSpeechSynthesizer |
| Windows | None — uses built-in SAPI5 |
| Linux | `sudo apt-get install espeak` |

**3. Verify the installation**

```python
import pyttsx3
engine = pyttsx3.init()
print(engine.getProperty("rate"))    # default speaking rate
print(engine.getProperty("volume"))  # default volume
```

If this prints two numbers without errors, your environment is ready.

---

## Question 1 — Listing Voices

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

## Question 2 — Single-Shot Speech

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

## Question 3 — Controlling the Voice

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

## Question 4 — Continuous Speech Loop

Write a function `speak_loop(voice_index=None, rate=150, volume=1.0)` that:

1. Creates a **single** engine instance to reuse across all iterations.
2. Applies the rate, volume, and optional voice settings once before the loop.
3. Repeatedly prompts the user to type text and speaks it aloud.
4. Skips empty input with a friendly message instead of trying to speak nothing.
5. Stops cleanly when the user types `"quit"`, `"stop"`, or `"exit"`.
6. Releases the audio driver after the loop ends.

<details>
<summary>Solution</summary>

```python
import pyttsx3

def speak_loop(voice_index=None, rate=150, volume=1.0):
    engine = pyttsx3.init()

    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)

    if voice_index is not None:
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[voice_index].id)

    print("\nText-to-Speech loop. Type 'quit' or 'stop' to exit.\n")

    while True:
        text = input("Enter text to speak: ")

        if text.strip().lower() in ("quit", "stop", "exit"):
            print("Stopping.")
            break

        if not text.strip():
            print("(nothing typed, try again)")
            continue

        engine.say(text)
        engine.runAndWait()

    engine.stop()
```

**Key points:**
- The engine is created **once** outside the loop — reinitialising it on every
  iteration is slow and can cause driver conflicts on some platforms.
- `text.strip().lower()` removes surrounding whitespace and normalises case so
  `"Quit"`, `"QUIT"`, and `" quit "` all trigger the exit condition.
- The empty-input guard prevents `engine.say("")` from being called, which can
  behave unexpectedly depending on the platform driver.
- `engine.stop()` is called after `break` to cleanly release the audio driver.

</details>

---

## Question 5 — Putting It All Together

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
