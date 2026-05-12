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
  Language Model  (DialoGPT via AutoModelForCausalLM)
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
model and its tokenizer directly using `AutoModelForCausalLM` and
`AutoTokenizer`, and returns both.

Also suppress transformers warnings at the top of the file so only errors are
shown.

<details>
<summary>Solution</summary>

```python
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers").setLevel(logging.ERROR)

def build_chatbot():
    print("Loading DialoGPT-medium (downloads ~863 MB on first run)...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.eval()
    print("Model ready.\n")
    return model, tokenizer
```

**Key points:**
- `AutoModelForCausalLM` loads the model weights directly, giving full control
  over tokenisation and generation — unlike `pipeline`, which overrides
  tokenizer settings internally.
- `tokenizer.pad_token = tokenizer.eos_token` is required because DialoGPT has
  no pad token defined; without it padding falls back to an incorrect default.
- `model.eval()` disables dropout layers, which are only needed during
  training — inference is faster and deterministic without them.
- `logging.getLogger("transformers").setLevel(logging.ERROR)` silences
  transformers' own logger (separate from Python's `warnings` module).
- The model is downloaded once and cached in `~/.cache/huggingface/`.

</details>

---

## Task 2 — Generating a Reply

Write a function `get_reply(user_input, history_ids, model, tokenizer)` that:

1. Encodes `user_input` (plus the EOS token) into a tensor of token IDs.
2. Concatenates it with `history_ids` from previous turns (`None` on the first turn).
3. Trims the combined input to the most recent 512 tokens to stay within
   DialoGPT's 1024-token context limit.
4. Generates a reply using the model directly, passing an explicit all-ones
   attention mask.
5. Decodes only the newly generated tokens into a string and returns
   `(reply, output_ids)`.
6. Falls back to a safe default string if the model returns an empty reply.

<details>
<summary>Solution</summary>

```python
import torch

MAX_HISTORY_TOKENS = 512

def get_reply(user_input, history_ids, model, tokenizer):
    new_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt",
    )

    input_ids = (
        torch.cat([history_ids, new_ids], dim=-1)
        if history_ids is not None
        else new_ids
    )

    if input_ids.shape[-1] > MAX_HISTORY_TOKENS:
        input_ids = input_ids[:, -MAX_HISTORY_TOKENS:]

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    reply = tokenizer.decode(
        output_ids[:, input_ids.shape[-1]:][0],
        skip_special_tokens=True,
    ).strip()

    if not reply:
        reply = "I'm not sure how to respond to that."

    return reply, output_ids
```

**Key points:**
- History is stored as a **token ID tensor**, not a list of strings — this is
  the native format DialoGPT was designed for and produces coherent results.
- `MAX_HISTORY_TOKENS = 512` prevents the context from overflowing DialoGPT's
  1024-token limit; without this the model deteriorates after a few exchanges.
- `torch.ones_like(input_ids)` creates an all-ones attention mask, explicitly
  telling the model there is no padding in the sequence.
- `torch.no_grad()` disables gradient computation during inference, saving
  memory and speeding up generation.
- `output_ids[:, input_ids.shape[-1]:]` slices off the input prefix, leaving
  only the newly generated reply tokens.

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

Write a function `chat_loop(model, tokenizer, tts_engine, recognizer=None, mic=None, use_voice=True)`
that:

1. Starts with `history_ids = None` (no prior context).
2. Each iteration collects input via voice (if `use_voice=True`) or keyboard.
3. Skips the iteration if no input was captured.
4. Stops cleanly when the user says or types `"quit"`, `"stop"`, or `"exit"`,
   saying goodbye before exiting.
5. Otherwise generates a reply with `get_reply()` and delivers it with
   `respond()`.

<details>
<summary>Solution</summary>

```python
def chat_loop(model, tokenizer, tts_engine, recognizer=None, mic=None, use_voice=True):
    history_ids = None
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

        reply, history_ids = get_reply(user_input, history_ids, model, tokenizer)
        respond(reply, tts_engine)
```

**Key points:**
- `history_ids = None` on the first turn signals `get_reply()` to skip
  concatenation and use only the new message as input.
- `history_ids` is updated each turn with the full output tensor, so
  DialoGPT always has the recent conversation as context.
- Checking `user_input is None` before the stop-word check avoids a
  `NoneType` error when `listen()` returns `None`.

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
import logging
import torch
import speech_recognition as sr
import pyttsx3
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers").setLevel(logging.ERROR)
MAX_HISTORY_TOKENS = 512

if __name__ == "__main__":
    print("=== Chatbot Tutorial — Speech + DialoGPT ===\n")

    model, tokenizer = build_chatbot()

    tts = build_tts()

    mode = input("Input mode — type 'v' for voice or 't' for text: ").strip().lower()
    use_voice = (mode == "v")

    if use_voice:
        recognizer, mic = build_recognizer()
        chat_loop(model, tokenizer, tts, recognizer, mic, use_voice=True)
    else:
        chat_loop(model, tokenizer, tts, use_voice=False)
```

**Key points:**
- `model, tokenizer = build_chatbot()` unpacks the tuple returned by the
  function — both are needed separately by `get_reply()`.
- The TTS engine and recognizer are created once and passed into the loop —
  this avoids reinitialising hardware drivers on every turn.
- Microphone calibration only happens in voice mode, keeping startup fast
  when using keyboard input.
- `use_voice = (mode == "v")` is a concise boolean assignment — it evaluates
  the comparison and stores `True` or `False` directly.

</details>
