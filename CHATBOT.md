# Workshop: Conversational Chatbot with Speech I/O and DialoGPT

---

## Introduction

### Background

This workshop combines the two previous workshops — Speech to Text and Text to
Speech — and adds a natural-language model to create a fully conversational
chatbot. The user can speak or type, the model generates a reply, and the reply
is both printed and spoken aloud.

The language model is **DialoGPT-medium** by Microsoft, a GPT-2-based model
fine-tuned on conversational data. It runs locally via the Hugging Face
`transformers` library. No cloud API or account is required.

**How the three components connect:**

```
User speaks / types
       ↓
  Speech-to-Text  (SpeechRecognition + Google Web Speech)
       ↓
  Language Model  (DialoGPT via transformers pipeline)
       ↓
  Text-to-Speech  (pyttsx3)  +  print to terminal
```

### Setting Up the Development Environment

**1. Create and activate a virtual environment with Python 3.12**

> DialoGPT requires `torch`, which does not yet support Python 3.13+.
> Use Python 3.12 to avoid compatibility issues.

```bash
python3.12 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**2. Install system dependencies (if needed)**

| Platform | Command |
|---|---|
| macOS | `brew install portaudio` |
| Linux | `sudo apt-get install python3-pyaudio portaudio19-dev espeak` |
| Windows | `pip install pipwin && pipwin install pyaudio` |

**3. Install Python dependencies**

All packages are pinned in `requirements.txt` to avoid version conflicts:

```bash
pip install -r requirements.txt
```

> The first run will download the DialoGPT-medium model (~863 MB). This only
> happens once — it is cached locally afterwards.

**4. Verify the installation**

```python
import torch, transformers, speech_recognition, pyttsx3
print(torch.__version__)          # should print 2.2.2
print(transformers.__version__)   # should print 4.38.0
```

If both version numbers print without errors, your environment is ready.

---

## Task 1 — Loading the Model

Write a function `build_chatbot()` that loads the `microsoft/DialoGPT-medium`
model into a Hugging Face `text-generation` pipeline and returns it.

Print a message before loading so the user knows to wait, and another once it
is ready.

<details>
<summary>Solution</summary>

```python
from transformers import pipeline

def build_chatbot():
    print("Loading DialoGPT-medium (downloads ~863 MB on first run)...")
    bot = pipeline("text-generation", model="microsoft/DialoGPT-medium")
    print("Model ready.\n")
    return bot
```

**Key points:**
- `pipeline("text-generation", ...)` wraps the model in a callable that
  accepts a text prompt and returns generated text.
- The model is downloaded once and cached in `~/.cache/huggingface/` on
  subsequent runs.
- Storing the pipeline in a variable and reusing it avoids reloading the
  model on every call, which would be very slow.

</details>

---

## Task 2 — Generating a Reply

Write a function `get_reply(user_input, history, chatbot)` that:

1. Appends `user_input` to `history`.
2. Formats the conversation as a single string where each turn is separated
   by the `<|endoftext|>` token (DialoGPT's turn separator).
3. Passes the formatted string to the pipeline and extracts only the newly
   generated reply (not the original prompt).
4. Appends the reply to `history` and returns `(reply, history)`.
5. Falls back to a safe default string if the model returns an empty reply.

<details>
<summary>Solution</summary>

```python
EOS = "<|endoftext|>"

def get_reply(user_input, history, chatbot):
    history.append(user_input)
    prompt = EOS.join(history) + EOS

    output = chatbot(
        prompt,
        max_new_tokens=100,
        pad_token_id=50256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    generated = output[0]["generated_text"]
    reply = generated[len(prompt):].split(EOS)[0].strip()

    if not reply:
        reply = "I'm not sure how to respond to that."

    history.append(reply)
    return reply, history
```

**Key points:**
- DialoGPT was trained on conversations formatted with `<|endoftext|>` between
  turns. Using this separator lets the model understand the full conversation
  context.
- `pad_token_id=50256` suppresses a warning — 50256 is the EOS token ID for
  the GPT-2 family.
- `do_sample=True` with `temperature=0.7` and `top_p=0.9` produces varied,
  natural-sounding replies instead of deterministic, repetitive ones.
- The pipeline returns the **full string** (prompt + reply). Slicing
  `generated[len(prompt):]` isolates only the new content.
- `.split(EOS)[0]` discards any extra turns the model may have hallucinated
  beyond the first reply.

</details>

---

## Task 3 — Voice Input

Write a function `listen(recognizer, mic)` that:

1. Opens the microphone and listens for speech (up to 6 s timeout, 12 s
   phrase limit).
2. Transcribes the audio using Google Web Speech and prints what was heard.
3. Returns the transcribed string, or `None` on timeout, unintelligible
   audio, or API failure.

<details>
<summary>Solution</summary>

```python
import speech_recognition as sr

def listen(recognizer, mic):
    with mic as source:
        print("Listening... (speak now)")
        try:
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=12)
        except sr.WaitTimeoutError:
            print("(no speech detected)")
            return None

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("(could not understand, try again)")
        return None
    except sr.RequestError as e:
        print(f"Speech API error: {e}")
        return None
```

**Key points:**
- `timeout=6` raises `sr.WaitTimeoutError` if no speech starts within 6 s —
  catching it with `return None` keeps the loop alive instead of crashing.
- Returning `None` on any failure lets the chat loop skip the current turn
  and re-prompt the user without special-casing each error type.

</details>

---

## Task 4 — Voice + Text Output

Write a function `respond(text, engine)` that prints the bot's reply prefixed
with `"Bot: "` and speaks it aloud using a `pyttsx3` engine.

<details>
<summary>Solution</summary>

```python
def respond(text, engine):
    print(f"\nBot: {text}\n")
    engine.say(text)
    engine.runAndWait()
```

**Key points:**
- Printing first means the user can read the reply while audio is playing,
  which is useful if the TTS engine speaks faster or slower than expected.
- `engine.say()` queues the text; `engine.runAndWait()` plays it and blocks
  until done — the next loop iteration only starts after the bot finishes
  speaking.

</details>

---

## Task 5 — Main Chat Loop

Write a function `chat_loop(chatbot, tts_engine, recognizer=None, mic=None, use_voice=True)`
that:

1. Starts with an empty conversation `history`.
2. Each iteration collects input via voice (if `use_voice=True`) or keyboard.
3. Skips the iteration if no input was captured.
4. Stops cleanly when the user says or types `"quit"`, `"stop"`, or `"exit"`,
   saying goodbye before exiting.
5. Otherwise generates a reply with `get_reply()` and delivers it with
   `respond()`.

<details>
<summary>Solution</summary>

```python
def chat_loop(chatbot, tts_engine, recognizer=None, mic=None, use_voice=True):
    history = []
    print("Chat started. Say or type 'quit' to exit.\n")

    while True:
        if use_voice:
            user_input = listen(recognizer, mic)
        else:
            user_input = input("You: ").strip() or None

        if user_input is None:
            continue

        if user_input.strip().lower() in ("quit", "stop", "exit"):
            respond("Goodbye!", tts_engine)
            break

        reply, history = get_reply(user_input, history, chatbot)
        respond(reply, tts_engine)
```

**Key points:**
- `history` accumulates every turn so DialoGPT has full context throughout
  the session. Resetting it would make the bot forget earlier turns.
- Checking `user_input is None` before the stop-word check avoids a
  `NoneType` error when `listen()` returns `None`.
- `respond()` is called for the farewell too, so the bot always speaks its
  last message before the program ends.

</details>

---

## Task 6 — Putting It All Together

Write a `__main__` block that:

1. Loads the chatbot model and TTS engine.
2. Asks the user whether they want voice or text input.
3. Sets up the microphone and calibrates for ambient noise if voice was chosen.
4. Starts the chat loop in the appropriate mode.

<details>
<summary>Solution</summary>

```python
import speech_recognition as sr
import pyttsx3
from transformers import pipeline

EOS = "<|endoftext|>"

if __name__ == "__main__":
    print("=== Chatbot Tutorial — Speech + DialoGPT ===\n")

    bot = build_chatbot()

    tts = pyttsx3.init()
    tts.setProperty("rate", 160)
    tts.setProperty("volume", 1.0)

    mode = input("Input mode — type 'v' for voice or 't' for text: ").strip().lower()
    use_voice = (mode == "v")

    if use_voice:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            print("Calibrating microphone for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Microphone ready.\n")
        chat_loop(bot, tts, recognizer, mic, use_voice=True)
    else:
        chat_loop(bot, tts, use_voice=False)
```

**Key points:**
- The TTS engine and recognizer are created once in `__main__` and passed
  into the functions — this avoids reinitialising hardware drivers on every
  turn.
- Microphone calibration only happens in voice mode, keeping startup fast
  when using keyboard input.
- `use_voice = (mode == "v")` is a concise boolean assignment — it evaluates
  the comparison and stores `True` or `False` directly.

</details>
