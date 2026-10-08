# Workshop: Conversational Chatbot with Speech I/O and Qwen2.5

---

## Introduction

### Background

This workshop combines the two previous workshops — Speech to Text and Text to
Speech — and adds a natural-language model to create a fully conversational
chatbot. The user can speak or type, the model generates a reply, and the reply
is both printed and spoken aloud.

The language model is **Qwen2.5-0.5B-Instruct**, a small instruction-tuned
model that runs locally via the Hugging Face `transformers` library. It is
Apache-2.0 licensed and ungated, so no cloud API, account, or access token is
required — unlike Llama, which needs a Hugging Face login and a signed licence
before it will download.

> **Why an instruction-tuned model?** An older conversational model such as
> DialoGPT is smaller and about 2.5x faster, but it was trained only to produce
> a *plausible next reply* — never to answer questions or stay consistent. Asked
> "are you a girl or a guy" it will say "Girl", then "I am a guy" two turns
> later, and asked "what is a president" it answers "I am a woman". Instruction
> tuning is what fixes that, and it is worth the extra seconds.

**How the three components connect:**

```
User speaks / types
       ↓
  Speech-to-Text  (SpeechRecognition + Google Web Speech)
       ↓
  Language Model  (Qwen2.5 via AutoModelForCausalLM)
       ↓
  Text-to-Speech  (pyttsx3)  +  print to terminal
```

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
3. Open a *new* PowerShell window and check the version:
   ```powershell
   py -3.12 --version
   ```
4. Create and activate the virtual environment. Windows has no `python3.12`
   command — use the `py` launcher instead:
   ```powershell
   py -3.12 -m venv venv
   venv\Scripts\activate
   ```
   If PowerShell reports *"running scripts is disabled on this system"*, allow
   local scripts once for your user and activate again:
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
3. Create and activate the virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

**Both platforms:** once the environment is active your prompt starts with
`(venv)`, and `python --version` prints `3.12.x`. From here on `python` and
`pip` refer to the venv, so the remaining commands are the same on Windows and
macOS. Run `deactivate` to leave the environment, and activate it again in
every new terminal window.

</details>

**1. Create and activate a virtual environment with Python 3.12**

> This project requires `torch`, which does not yet support Python 3.13+.
> Use Python 3.12 to avoid compatibility issues.

```bash
python3.12 -m venv venv     # Windows: py -3.12 -m venv venv

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
| Windows | None — PyAudio ships prebuilt wheels for Python 3.12 |

**3. Install Python dependencies**

All packages are pinned in `requirements.txt` to avoid version conflicts:

```bash
pip install -r requirements.txt
```

> The first run will download Qwen2.5-0.5B-Instruct (~1 GB). This only
> happens once — it is cached locally afterwards.
>
> Expect roughly **10–15 seconds per reply** on a CPU.

**4. Verify the installation**

```python
import torch, transformers, speech_recognition, pyttsx3
print(torch.__version__)          # should print 2.2.2
print(transformers.__version__)   # should print 4.38.0
```

If both version numbers print without errors, your environment is ready.

---

## Task 1 — Loading the Model

Write a function `build_chatbot()` that loads `Qwen/Qwen2.5-0.5B-Instruct` and
its tokenizer directly using `AutoModelForCausalLM` and `AutoTokenizer`, and
returns both.

Also suppress transformers warnings at the top of the file, and define a system
prompt that pins the assistant's identity and keeps replies short.

<details>
<summary>Solution</summary>

```python
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant running locally on the Qwen2.5 model. "
    "If you are asked your name, say you are a Qwen2.5 assistant. "
    "Keep replies to one or two short sentences, since they will be read aloud."
)

def build_chatbot():
    print(f"Loading {MODEL_NAME} (downloads ~1 GB on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    model.eval()
    print("Model ready.\n")
    return model, tokenizer
```

**Key points:**
- `AutoModelForCausalLM` loads the model weights directly, giving full control
  over tokenisation and generation — unlike `pipeline`, which overrides
  tokenizer settings internally.
- **No `pad_token` assignment is needed.** Qwen ships a real pad token. Older
  GPT-2-based models do not, which is why tutorials for those models must borrow
  the EOS token first.
- **`torch_dtype=torch.float32` is deliberate.** The tempting choice on a small
  machine is `bfloat16`, which halves the weights to ~1 GB. But CPUs without
  native bf16 support emulate it: measured on a 1.1 GHz Intel i5, float32 ran
  **~11% faster** (3.34 vs 3.01 tokens/sec). At 0.5B parameters float32 needs
  only ~2 GB, so the memory saving buys nothing here. Larger models flip this.
- **The system prompt earns its place twice.** It pins the identity — asked its
  name, the bare model confidently answers *"I am Claude, created by
  Anthropic"*, a hallucination absorbed from assistant transcripts in its
  training data. And it demands short replies, which matters far more here than
  in a text-only chatbot: every reply is read aloud, and a rambling paragraph is
  painful to sit through.
- `model.eval()` disables dropout layers, which are only needed during
  training — inference is faster and deterministic without them.
- `logging.getLogger("transformers").setLevel(logging.ERROR)` silences
  transformers' own logger (separate from Python's `warnings` module).
- The model is downloaded once and cached in `~/.cache/huggingface/`.

</details>

---

## Task 2 — Building the Prompt and Generating a Reply

Instruction-tuned models do not take raw concatenated text. They expect a
**list of messages with roles**, rendered into the exact control tokens the
model was trained on. Getting that formatting right is what separates coherent
replies from garbage.

Write two functions:

- `build_prompt(history, tokenizer)` — prepend the system prompt, render the
  messages with `apply_chat_template`, and drop the oldest turns until the
  result fits `MAX_CONTEXT_TOKENS`. Return `(input_ids, history)`.
- `get_reply(user_input, history, model, tokenizer)` — append the user's
  message, generate, decode only the new tokens, fall back to a safe default if
  the reply is empty, and return `(reply, updated_history)`.

<details>
<summary>Solution</summary>

```python
import torch

MAX_CONTEXT_TOKENS = 1024
MAX_NEW_TOKENS = 128

def build_prompt(history, tokenizer):
    while True:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if input_ids.shape[-1] <= MAX_CONTEXT_TOKENS:
            return input_ids, history
        if len(history) <= 1:
            return input_ids, history

        history = history[2:]   # drop the oldest user+assistant pair


def get_reply(user_input, history, model, tokenizer):
    history = history + [{"role": "user", "content": user_input}]

    input_ids, history = build_prompt(history, tokenizer)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
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

    return reply, history + [{"role": "assistant", "content": reply}]
```

**Key points:**
- **History is a list of `{"role", "content"}` dicts**, not a token tensor. The
  roles are what let the model tell your words from its own — which is exactly
  what a non-instruct model lacks, and why such models contradict themselves
  about who they are.
- **`apply_chat_template` writes the control tokens for you.** Qwen expects
  turns wrapped as `<|im_start|>user … <|im_end|>`. Every model family uses a
  different scheme, and the template ships with the tokenizer, so this one call
  stays correct if you swap models. Hand-formatting that string is the single
  most common cause of garbled output.
- **`add_generation_prompt=True` is essential.** It appends the opening of the
  assistant's turn, so the model continues *as the assistant*. Omit it and the
  model often invents your next line instead of replying to you.
- **Trimming drops whole turns, not tokens.** Cutting a token tensor at a fixed
  length — the natural approach without roles — can slice through the middle of
  a message and leave a dangling half-turn. Removing `history[:2]` (one user +
  assistant pair) always leaves valid structure.
- The system prompt is prepended **inside the loop**, so it is never part of
  what gets trimmed. Store it in the history instead and a long conversation
  would eventually delete the assistant's own instructions.
- `len(history) <= 1` stops the loop when only the current user message remains.
  Without that guard, one oversized message would loop forever.
- `history + [...]` builds a **new list** rather than calling `.append()`. If
  generation raises, the caller's history is left untouched instead of holding a
  user message that never got a reply.
- `torch.ones_like(input_ids)` creates an all-ones attention mask, explicitly
  telling the model there is no padding in the sequence.
- `torch.no_grad()` disables gradient computation during inference, saving
  memory and speeding up generation.
- `output_ids[:, input_ids.shape[-1]:]` slices off the input prefix, leaving
  only the newly generated reply tokens.
- `MAX_NEW_TOKENS` is a **ceiling, not a target**. Generation stops at the
  end-of-turn token, usually well short of it.

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

Write a function `respond(text, rate=160, volume=1.0)` that prints the bot's
reply prefixed with `"Bot: "` and speaks it aloud, creating a **fresh**
`pyttsx3` engine instance for each call rather than reusing one across turns.

<details>
<summary>Solution</summary>

```python
import pyttsx3

def respond(text, rate=160, volume=1.0):
    print(f"\nBot: {text}\n")
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
```

**Key points:**
- Printing first means the user can read the reply while audio is playing,
  which is useful if the TTS engine speaks faster or slower than expected.
- `engine.say()` queues the text; `engine.runAndWait()` plays it and blocks
  until done — the next loop iteration only starts after the bot finishes
  speaking.
- A new engine is created **every call** instead of being passed in and
  reused. On Windows, the SAPI5 driver's event loop only runs correctly once
  per engine instance — reusing one instance across turns causes every reply
  after the first to be silently skipped, even though `say()` still queues
  the text without error. Creating a short-lived engine per turn avoids this
  and works consistently across macOS, Windows, and Linux.

</details>

---

## Task 5 — Main Chat Loop

Write a function `chat_loop(model, tokenizer, recognizer=None, mic=None, use_voice=True, tts_rate=160, tts_volume=1.0)`
that:

1. Starts with `history = []` (no prior context).
2. Each iteration collects input via voice (if `use_voice=True`) or keyboard.
3. Skips the iteration if no input was captured.
4. Stops cleanly when the user says or types `"quit"`, `"stop"`, or `"exit"`,
   saying goodbye before exiting.
5. Otherwise generates a reply with `get_reply()` and delivers it with
   `respond()`.

<details>
<summary>Solution</summary>

```python
def chat_loop(model, tokenizer, recognizer=None, mic=None, use_voice=True, tts_rate=160, tts_volume=1.0):
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
            respond("Goodbye!", tts_rate, tts_volume)
            break

        print("(thinking...)")
        reply, history = get_reply(user_input, history, model, tokenizer)
        respond(reply, tts_rate, tts_volume)
```

**Key points:**
- `history = []` on the first turn means `get_reply()` sends just the system
  prompt plus the new message.
- `history` is reassigned each turn from `get_reply()`'s return value, so the
  model always has the recent conversation as context — and so any trimming
  done inside `build_prompt()` is preserved rather than recomputed.
- **The `(thinking...)` line is not decoration.** Generation takes 10–15 s on a
  CPU. Without it the terminal sits silent and looks frozen, and in voice mode
  the user cannot tell whether the bot heard them at all.
- Checking `user_input is None` before the stop-word check avoids a
  `NoneType` error when `listen()` returns `None`.
- `tts_rate` and `tts_volume` are passed through to `respond()` rather than
  a shared engine instance, since `respond()` now builds its own engine per
  call (see Task 4).

</details>

---

## Task 6 — Putting It All Together

Write a `__main__` block that:

1. Loads the chatbot model.
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
MAX_CONTEXT_TOKENS = 1024
MAX_NEW_TOKENS = 128

if __name__ == "__main__":
    print("=== Chatbot Tutorial — Speech + Qwen2.5 ===\n")

    model, tokenizer = build_chatbot()

    mode = input("Input mode — type 'v' for voice or 't' for text: ").strip().lower()
    use_voice = (mode == "v")

    if use_voice:
        recognizer, mic = build_recognizer()
        chat_loop(model, tokenizer, recognizer, mic, use_voice=True)
    else:
        chat_loop(model, tokenizer, use_voice=False)
```

**Key points:**
- `model, tokenizer = build_chatbot()` unpacks the tuple returned by the
  function — both are needed separately by `get_reply()`.
- There is no shared TTS engine to build up front — `respond()` creates a
  short-lived engine for each reply (see Task 4), so no driver handle needs
  to be threaded through the main block.
- The recognizer and microphone are still created once and passed into the
  loop, since re-calibrating the mic every turn would be slow and unnecessary
  — this only applies to speech *input*, not speech *output*.
- Microphone calibration only happens in voice mode, keeping startup fast
  when using keyboard input.
- `use_voice = (mode == "v")` is a concise boolean assignment — it evaluates
  the comparison and stores `True` or `False` directly.

</details>
