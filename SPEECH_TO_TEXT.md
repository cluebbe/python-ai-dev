# Exam: Speech to Text with Python and Google Web Speech API

---

## Introduction

### Background

Speech-to-text (STT) is the process of converting spoken audio into written
text. It powers voice assistants, transcription tools, accessibility features,
and hands-free interfaces.

This workshop uses **Google's Web Speech API**, accessed through the
[`SpeechRecognition`](https://pypi.org/project/SpeechRecognition/) library.
The Web Speech API is Google's free, cloud-based recognition service — no
account or API key is required for low-volume use. Audio is recorded locally,
sent over HTTPS to Google's servers, and the transcription is returned as plain
text.

Under the hood, `SpeechRecognition` uses **PyAudio** to talk to the system's
microphone. PyAudio is a Python binding for **PortAudio**, a cross-platform C
library for real-time audio I/O.

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
pip install SpeechRecognition pyaudio
```

**Platform notes for PyAudio:**

| Platform | If `pip install pyaudio` fails |
|---|---|
| macOS | `brew install portaudio` then retry `pip install pyaudio` |
| Linux | `sudo apt-get install python3-pyaudio portaudio19-dev` |
| Windows | `pip install pipwin && pipwin install pyaudio` |

**3. Verify the installation**

```python
import speech_recognition as sr
print(sr.__version__)
print(sr.Microphone.list_microphone_names())
```

If this prints a version number and a list of microphone names, your environment
is ready.

---

## Question 1 — Listing Devices

Write a function `list_microphones()` that prints every available microphone on
the system together with its device index, using the `speech_recognition` library.

Example output:
```
Available microphones:
  [0] Built-in Microphone
  [1] USB Audio Device
```

<details>
<summary>Solution</summary>

```python
import speech_recognition as sr

def list_microphones():
    """Print all available microphone devices and their index numbers."""
    print("Available microphones:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{index}] {name}")
```

`sr.Microphone.list_microphone_names()` returns a list of strings. `enumerate()`
pairs each name with its position, which is also the `device_index` you pass to
`sr.Microphone()` to select a specific mic.

</details>

---

## Question 2 — Single-Shot Transcription

Write a function `transcribe_once(mic_index=None, language="en-US")` that:

1. Creates a `Recognizer` and opens the specified microphone (or the system
   default if `mic_index` is `None`).
2. Calibrates for ambient noise for 1 second before listening.
3. Captures one spoken phrase and sends it to Google for transcription.
4. Returns the transcribed text, or `None` if recognition failed.
5. Handles both the case where Google cannot understand the audio **and** the
   case where the network or API call fails.

<details>
<summary>Solution</summary>

```python
import speech_recognition as sr

def transcribe_once(mic_index=None, language="en-US"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=mic_index)

    with mic as source:
        print("Calibrating for ambient noise... (hold still)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... (speak now)")
        audio = recognizer.listen(source)

    print("Transcribing...")
    try:
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Google API request failed: {e}")
        return None
```

**Key points:**
- `adjust_for_ambient_noise()` sets the silence threshold so the recognizer
  knows what counts as background noise versus speech.
- `recognize_google()` accepts a BCP-47 `language` tag (e.g. `"es-ES"`,
  `"de-DE"`) to transcribe languages other than English.
- `UnknownValueError` → audio was received but could not be decoded.
- `RequestError` → network failure or API-level error.

</details>

---

## Question 3 — Error Handling

Explain the difference between `sr.UnknownValueError` and `sr.RequestError`.
When is each exception raised, and why must both be caught separately?

<details>
<summary>Solution</summary>

| Exception | When raised | Cause |
|---|---|---|
| `sr.UnknownValueError` | Recognition succeeds technically, but the audio is unintelligible | Too much noise, silence, or unclear speech |
| `sr.RequestError` | The API call itself fails | No internet, API quota exceeded, invalid key |

They must be caught separately because the appropriate response differs:

- `UnknownValueError` is recoverable — the program can ask the user to speak
  again.
- `RequestError` usually means the service is unavailable — continuing to retry
  will not help until the underlying issue (network, quota) is resolved.

Example:

```python
try:
    text = recognizer.recognize_google(audio)
except sr.UnknownValueError:
    print("Could not understand audio.")   # ask user to repeat
    return None
except sr.RequestError as e:
    print(f"Google API request failed: {e}")  # abort / notify user
    return None
```

</details>

---

## Question 4 — Continuous Transcription

Write a function `transcribe_loop(mic_index=None, language="en-US")` that:

1. Calibrates for ambient noise **once** before entering the loop.
2. Continuously listens and prints each transcribed phrase.
3. Stops cleanly when the user says `"quit"`, `"stop"`, or `"exit"`.
4. If no speech is detected within 5 seconds, prints a waiting message and
   continues listening rather than crashing.
5. Caps each recording at 10 seconds of speech per phrase.
6. Breaks out of the loop on an API error.

<details>
<summary>Solution</summary>

```python
import speech_recognition as sr

def transcribe_loop(mic_index=None, language="en-US"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=mic_index)

    with mic as source:
        print("Calibrating for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

    print("\nListening continuously. Say 'quit' or 'stop' to exit.\n")

    while True:
        with mic as source:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                print("(no speech detected, still listening...)")
                continue

        try:
            text = recognizer.recognize_google(audio, language=language)
            print(f"You said: {text}")

            if text.lower() in ("quit", "stop", "exit"):
                print("Stopping.")
                break

        except sr.UnknownValueError:
            print("(could not understand)")
        except sr.RequestError as e:
            print(f"API error: {e}")
            break
```

**Key points:**
- Calibration happens **outside** the loop — recalibrating every iteration
  would add a 1-second pause between every phrase.
- `timeout=5` raises `sr.WaitTimeoutError` if silence lasts more than 5 s;
  catching it with `continue` keeps the loop alive.
- `phrase_time_limit=10` prevents a single phrase from blocking the loop
  indefinitely if the user keeps talking.
- `text.lower()` normalises the stop word check so "Quit", "QUIT", etc. all
  work.

</details>

---

## Question 5 — Putting It All Together

Write a `__main__` block that:

1. Runs a single-shot transcription and prints the result if one was returned.
2. Then starts continuous transcription.

<details>
<summary>Solution</summary>

```python
if __name__ == "__main__":
    print("=== Google Speech-to-Text Tutorial ===\n")

    # Uncomment to see all available microphones:
    # list_microphones()

    print("--- Single-shot mode ---")
    result = transcribe_once()
    if result:
        print(f"Transcription: {result}\n")

    print("--- Continuous mode ---")
    transcribe_loop()
```

The `if __name__ == "__main__"` guard ensures this block only runs when the
script is executed directly — not when it is imported as a module by another
script.

</details>
