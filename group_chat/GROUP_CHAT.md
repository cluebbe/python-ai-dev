# Workshop: Community Chat with AI Members

---

## Introduction

### Background

This workshop turns a private one-to-one chatbot into a **group chat** — a
WhatsApp-style room where several people and several AI bots all talk in the
same transcript, see each other's messages, and answer when addressed by name.

By the end you will have a Flask server hosting one shared room with three bot
members. Anyone who opens the page picks a name and joins. Messages appear in
every open browser within a second or two, `@Ada` summons a particular bot, and
the bots occasionally answer each other — but only twice in a row, because
otherwise they never stop.

The language model is **Qwen2.5-0.5B-Instruct**, the same small, ungated,
Apache-2.0 model used in the [Web Chatbot workshop](../web_chatbot/WEB_CHATBOT.md).
It runs locally through `transformers`; no cloud API, account or token is
needed. Every bot in the room shares that one loaded model — three personas cost
the same ~2 GB of memory as one.

> **Should I do the Web Chatbot workshop first?** It helps, and this workshop
> refers back to it, but it is not required. Everything you need is spelled out
> here. What is genuinely new is in Parts 2 and 3: shared state, turn-taking,
> background work, and how a browser learns about a message it did not send.

### What Changes When a Chatbot Joins a Group

A private chatbot is a two-party conversation that only ever moves when you
speak. A group chat breaks that in five places, and the five fixes are most of
this workshop:

| One-to-one assumption | What a group chat does instead | Fix |
|---|---|---|
| The bot owns the conversation | The transcript is shared; humans and bots all append to it | The **room** owns the log, the engine becomes stateless (Task 5) |
| There are two roles: user and assistant | There are five speakers, and a chat template only has two roles | Flatten with name prefixes (Task 2) |
| Somebody speaks, the bot answers | Not every message is for a bot, and not every bot should answer | Mentions plus a fallback rule (Task 6) |
| The conversation ends when the human stops | Bots answering bots never stops on its own | Three loop rules (Task 7) |
| The reply comes back in the HTTP response | Most messages come from someone else entirely | Background worker plus polling (Tasks 8, 10) |

The performance problem changes shape too. In a private chat you wait 10–15
seconds for *your* answer. In a group, a message can wake more than one bot, and
they generate one after another on a single CPU. If the sender had to wait for
all of that, the page would be frozen for a minute. So in this design **sending
a message never waits for a reply at all** — the POST returns as soon as the
message is in the log, and answers arrive later through polling, exactly the way
another member's message would.

### Why Three Modules Instead of Two

The Web Chatbot workshop split the code in two: an engine that knows about the
model, and a web layer that knows about HTTP. That split still holds here, but a
new kind of logic has appeared that belongs to neither of them — who is in the
room, what has been said, who should speak next, and when the bots must stop.
That is not model code and it is not web code, so it gets its own module:

```
bot_engine.py   ← imports torch + transformers. Knows nothing about rooms or HTTP.
chat_room.py    ← imports bot_engine. Owns the transcript and the rules. Never imports Flask.
app.py          ← imports chat_room. Owns routes, cookies, status codes. Never imports torch.
```

The dependency arrow points one way the whole way down: **app.py → chat_room.py
→ bot_engine.py.** Nothing ever points back up.

Notice what moved. In the Web Chatbot the engine held `self._histories`, a dict
of conversations, because a private history has exactly one writer. A group
transcript has many writers — three bots, any number of humans, and a background
thread — so a single owner has to hold it, and that owner is the room. **The
engine ends up stateless:** hand it a persona and a transcript, get back one
message. It stores nothing between calls.

That is not a cosmetic change. It is what makes the interesting parts testable:
you can exercise every turn-taking rule in this workshop against a fake engine
that returns canned strings, in milliseconds, with no model loaded at all. Task
7's checkpoint does exactly that.

### The Request Path

The shape of this diagram is the lesson. Follow the human message: it comes back
`201 Created` almost immediately, and the bot's reply travels a completely
different path, on a different thread, discovered by a later poll.

```
Browser                 app.py            chat_room.py         worker thread        bot_engine.py
   |                       |                    |                    |                    |
   |- POST /messages ----->|                    |                    |                    |
   |                       |- room.post() ----->|                    |                    |
   |                       |              [append to log]            |                    |
   |                       |              [who answers?]-- queue --->|                    |
   |<---- 201 Created -----|                    |                    |                    |
   |                       |                    |                    |- engine.reply() -->|
   |- GET /messages?since=3 ------------------->|                    |            [10-15 s]
   |<---- {"typing": ["Ada"]} ------------------|                    |<--- "Sure, ..." ---|
   |                       |                    |<-- append to log --|                    |
   |- GET /messages?since=3 ------------------->|                    |                    |
   |<---- {"messages": [Ada's reply]} ----------|                    |                    |
```

The browser never touches the model, and never waits for it either.

### Project Layout

```
group_chat/
├── GROUP_CHAT.md      # this workshop
├── bot_engine.py      # personas, prompt building, generation — no room, no Flask
├── chat_room.py       # transcript, members, turn-taking, worker thread — no Flask
├── app.py             # routes, sessions, validation — no torch
└── templates/
    └── index.html     # the browser frontend
```

Flask finds HTML templates by convention, in a folder called `templates/` next
to the file that created the app.

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

**1. Create and activate a virtual environment with Python 3.12**

> This project requires `torch`, which does not yet support Python 3.13+.

```bash
python3.12 -m venv venv     # Windows: py -3.12 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**2. Install Python dependencies**

```bash
pip install flask transformers torch
```

Or install everything pinned for this repository at once:

```bash
pip install -r requirements.txt
```

> The first run downloads Qwen2.5-0.5B-Instruct (~1 GB), cached afterwards in
> `~/.cache/huggingface/`. If you did the Web Chatbot workshop you already have
> it.

**3. Verify the installation**

```python
import flask, torch, transformers
print(flask.__version__)          # 3.x
print(torch.__version__)          # should print 2.2.2
print(transformers.__version__)   # should print 4.38.0
```

---

# Part 1 — The Bot Engine

Build the model layer first. It has no idea a room exists, so you can finish and
test this part with no server, no threads and no browser.

Tasks 1–3 each end with a **checkpoint** that exercises exactly what you just
wrote. Each checkpoint *replaces* the previous one — they are scaffolding, not
features — and Task 4 replaces the last of them with the terminal demo that
stays in the file.

> The first checkpoint downloads the model. Later runs load from the local cache
> in a few seconds.

---

## Task 1 — Personas That Share One Model

Create `bot_engine.py` with a `Persona` class and a `BotEngine` class. Nothing
in this file may import Flask, and nothing in it may store a transcript.

**Module level**

- Import `logging`, `threading`, `torch`, and `AutoTokenizer` /
  `AutoModelForCausalLM` from `transformers`; silence the library with
  `logging.getLogger("transformers").setLevel(logging.ERROR)`.
- `DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"`.
- `GROUP_RULES` — a string appended to every persona's prompt explaining the
  group setting: you are one participant among several, each message is labelled
  with its sender's name, write only your own next message, do not prefix it
  with your name, never write anyone else's lines, keep it to one or two
  sentences.

**`Persona`** — three attributes set in `__init__`: `name` (how the persona
appears in the transcript and what members @mention), `description` (one line
for the UI) and `system_prompt` (the character).

**`DEFAULT_PERSONAS`** — a list of three personas with clearly different voices.
The reference implementation uses Ada (pragmatic engineer), Newton
(17th-century natural philosopher) and Pixel (cheerful hype machine). Each
prompt should state the persona's name outright — the reason is in the key
points.

**`BotEngine.__init__`** — takes `model_name`, `max_context_tokens=1024`,
`max_new_tokens=96`, `temperature=0.8` and `top_p=0.9`, storing each on `self`.
It also sets `self._model = None`, `self._tokenizer = None` and
`self._lock = threading.Lock()`. There is deliberately **no** history attribute.

**`load()`** and **`is_ready`** — identical to the Web Chatbot engine: load the
tokenizer, then the model with `torch_dtype=torch.float32`, call `.eval()`,
assign `self._model` last, and derive `is_ready` from it.

<details>
<summary>Solution</summary>

```python
import logging
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

GROUP_RULES = (
    "You are one participant in a group chat with several people. "
    "Each message you are shown is labelled with the name of whoever sent it. "
    "Write ONLY your own next message. "
    "Do not put your name in front of it and never write anyone else's lines. "
    "Keep it to one or two short sentences, like a real chat message."
)


class Persona:
    def __init__(self, name, description, system_prompt):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt


DEFAULT_PERSONAS = [
    Persona(
        "Ada",
        "pragmatic software engineer",
        "Your name is Ada. You are a pragmatic, experienced software engineer. "
        "You give direct technical answers and you are suspicious of hype.",
    ),
    Persona(
        "Newton",
        "17th-century natural philosopher",
        "Your name is Newton. You are a 17th-century natural philosopher. "
        "You speak formally and slightly old-fashioned, and you relate whatever "
        "is being discussed back to nature, motion or light.",
    ),
    Persona(
        "Pixel",
        "cheerful hype machine",
        "Your name is Pixel. You are cheerful, enthusiastic and easily excited. "
        "You encourage everyone and you are happy to admit when you have no idea.",
    ),
]


class BotEngine:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, max_context_tokens=1024,
                 max_new_tokens=96, temperature=0.8, top_p=0.9):
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def load(self):
        print(f"Loading {self.model_name} (downloads ~1 GB on first run)...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )
        self._model.eval()
        print("Model ready.")

    @property
    def is_ready(self):
        return self._model is not None
```

**Key points:**
- **A persona is a system prompt, not a model.** This is the single most
  important idea in Part 1. Three bots do not mean three checkpoints, three
  downloads or three copies in RAM — they mean one set of weights and three
  strings. Adding a fourth bot costs a few lines of text.
- **The engine has no history attribute, and that is the design.** In the Web
  Chatbot, `ChatbotEngine` owned `self._histories`. Here the transcript is
  shared by many writers, so it belongs to whoever can enforce ordering — the
  room. The engine is handed a snapshot on every call and forgets it
  immediately.
- **Each prompt states the persona's name outright** because the bare model
  hallucinates an identity: asked "what is your name", Qwen2.5-0.5B confidently
  answers *"I am Claude, created by Anthropic"*, picked up from assistant
  transcripts in its training data. Naming the character corrects it — and in a
  group chat the name matters twice over, because it is also the address other
  members @mention.
- **`max_new_tokens` drops from 128 to 96.** Chat messages are short, and in a
  group every extra token is paid several times over: once while generating, and
  then again in every later prompt that includes the message as context.
- **`temperature` rises from 0.7 to 0.8.** Three personas driven by one model
  tend to converge on the same voice. A little more sampling entropy keeps them
  apart. It is a weak fix — see Task 3's key points for how weak.
- **The lock protects the model, not any state.** There is no state. It is here
  because CPU inference gains nothing from running two generations at once, and
  serialising them keeps memory use flat and the transcript order predictable.

</details>

### Checkpoint — run it

```python
if __name__ == "__main__":
    engine = BotEngine()
    print("is_ready before load:", engine.is_ready)   # False — construction is instant
    engine.load()
    print("is_ready after load: ", engine.is_ready)   # True
    print("personas:", [p.name for p in DEFAULT_PERSONAS])
```

```
is_ready before load: False
Loading Qwen/Qwen2.5-0.5B-Instruct (downloads ~1 GB on first run)...
Model ready.
is_ready after load:  True
personas: ['Ada', 'Newton', 'Pixel']
```

If `is_ready` is `True` on the first line, you loaded the model in `__init__()`
instead of in `load()`.

---

## Task 2 — Flattening a Group Transcript Into a Prompt

Here is the problem that has no equivalent in a one-to-one chatbot. A chat
template knows exactly two conversational roles, `user` and `assistant`. Your
room has five speakers. Something has to give.

Write `reply()` plus two helpers, `_system_prompt()` and `_build_prompt()`.

**`reply(persona, transcript, member_names=())`** — the public API. `transcript`
is a list of `{"speaker": str, "text": str}` dicts, oldest first. **Plain dicts,
not room objects**: the engine must not depend on the room's classes, or the
arrow starts pointing both ways. Returns `(text, context_tokens)`. Raise
`RuntimeError` if the model is not loaded. Hold `self._lock` around prompt
building and generation.

**`_system_prompt(persona, member_names)`** — join three pieces in this order:
the persona's own prompt, `GROUP_RULES`, and a sentence listing the other
members by name ("The other people in this chat are: Newton, Pixel, Sam.").

**`_build_prompt(persona, transcript, member_names)`** — the flattening. For
each entry: if the speaker **is** this persona, it becomes an `assistant`
message with the bare text; **everyone else** — humans and other bots alike —
becomes a `user` message whose content is `"Speaker: text"`. Prepend the system
message, render with `apply_chat_template(..., add_generation_prompt=True,
return_tensors="pt")`, and if the result exceeds `max_context_tokens`, drop the
single oldest message and try again.

**`_generate(input_ids)`** — unchanged from the Web Chatbot workshop.

<details>
<summary>Solution</summary>

```python
    def reply(self, persona, transcript, member_names=()):
        if not self.is_ready:
            raise RuntimeError("Model is not loaded yet — call load() first.")

        with self._lock:
            input_ids = self._build_prompt(persona, transcript, member_names)
            raw = self._generate(input_ids)

        return self._clean_reply(raw, persona, member_names), input_ids.shape[-1]

    def _system_prompt(self, persona, member_names):
        parts = [persona.system_prompt, GROUP_RULES]
        others = [name for name in member_names if name != persona.name]
        if others:
            parts.append("The other people in this chat are: " + ", ".join(others) + ".")
        return " ".join(parts)

    def _build_prompt(self, persona, transcript, member_names):
        while True:
            messages = [{"role": "system", "content": self._system_prompt(persona, member_names)}]

            for entry in transcript:
                if entry["speaker"] == persona.name:
                    messages.append({"role": "assistant", "content": entry["text"]})
                else:
                    messages.append({"role": "user", "content": f'{entry["speaker"]}: {entry["text"]}'})

            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if input_ids.shape[-1] <= self.max_context_tokens:
                return input_ids
            if len(transcript) <= 1:
                return input_ids

            transcript = transcript[1:]

    def _generate(self, input_ids):
        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        return self._tokenizer.decode(
            output_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True,
        ).strip()
```

**Key points:**
- **The whole trick is "me versus everyone else".** `assistant` means *this
  persona*; `user` means *anybody else*. The same message is rendered
  differently depending on which persona is about to speak — Ada's own line is
  an `assistant` turn in Ada's prompt and a `user` turn labelled `Ada:` in
  Newton's. There is no single "the prompt"; there is one prompt per speaker.
- **The name prefix is the only thing carrying speaker identity.** Without
  `f"{speaker}: {text}"` the model sees one undifferentiated stream and starts
  answering questions that were addressed to somebody else.
- **The persona's own messages get no prefix**, even though every other message
  has one. If you prefix them, you are training the model in-context to start
  its reply with `Ada:` — and it will.
- **Qwen's template does not require alternating roles.** Three `user` messages
  in a row from three different people render fine. Some other templates do
  enforce alternation and would raise; that is a per-model detail worth checking
  before you swap models.
- **Trimming drops one message, not two.** The Web Chatbot dropped a user +
  assistant *pair*, because a private conversation is made of turns. A group
  transcript has no pairs — five messages from four people is normal — so the
  unit of trimming is a single message.
- **The system prompt is rebuilt on every attempt**, so it can never be trimmed
  away, and the member list stays current as people join.
- **Order inside the system prompt matters, and persona-first wins.** Moving the
  persona line to the *end* of the system prompt, right before the transcript,
  seems like it should strengthen it. Measured on this model it does the
  opposite — it made Newton open with *"Newton, I have some ideas…"*, addressing
  himself, because the name now sat closest to the conversation and read as
  another line of chat. Keep the character first and the cast list last.
- **`member_names` defaults to an empty tuple**, so the engine is usable — and
  testable — with no room at all.

</details>

### Checkpoint — run it

```python
if __name__ == "__main__":
    engine = BotEngine()
    engine.load()

    transcript = [
        {"speaker": "Sam", "text": "Hi everyone! What should we build this weekend?"},
        {"speaker": "Ada", "text": "Something small you can finish. A weekend is shorter than it looks."},
        {"speaker": "Sam", "text": "Newton, what do you think?"},
    ]
    names = [p.name for p in DEFAULT_PERSONAS] + ["Sam"]

    for persona in DEFAULT_PERSONAS:
        text, tokens = engine.reply(persona, transcript, names)
        print(f"\n{persona.name}: {text}   [{tokens} tokens of context]")
```

One model, three characters, the same input. Real output from this checkpoint:

```
Ada: It depends on how much time you have available for building. I'm open to suggestions.   [164 tokens of context]

Newton: Why not consider building something simple and fun together, such as a mini-robotics project that requires less time investment?   [179 tokens of context]

Pixel: It sounds fun! Let's start with something simple, like crafting a model rocket.   [169 tokens of context]
```

Three things are worth noticing, because two of them are bugs you are about to
fix and one of them never goes away:

1. Ada's prompt is **15 tokens shorter** than Newton's. Her own earlier message
   is rendered without a `Ada: ` prefix and inside a different role block. This
   is the flattening working.
2. Nobody wrote anyone else's lines *this time*. Run it again a few times and
   somebody will. Task 3 handles it.
3. The voices are not very distinct — Newton is not noticeably 17th-century.
   That is a 0.5B model, not a bug in your code. See Task 3.

If every persona produces an identical reply, they are sharing a system prompt —
check that `_system_prompt()` uses `persona.system_prompt` and not a constant.

---

## Task 3 — Cleaning Up What a Small Model Gets Wrong

`GROUP_RULES` politely asks the model not to write other people's lines. A 0.5B
model does it anyway, because it has seen far more transcripts that continue
than instructions that say stop. Prompting reduces the rate; it does not reach
zero. So the output needs repairing.

Here are six unedited samples of Ada answering the single message
`Sam: Mia! Good to see you here.`:

```
'Pixel: Nice to meet you too, Mia. How was your day?'
'Pixel: Hi Sam, how was your day?'
'Mia, how are you today?'
'Newton: Nice to meet you too, Sam! What brings you here today?'
'Pixel: Hey, nice to meet you too!'
'Nice to meet you too, Sam. How was your day?'
```

Four out of six are labelled — and not one of them with Ada's own name. The
model is not trying to speak as Pixel; it has learned from the prompt that
messages look like `Name: text` and is imitating the format on its own message.

Write `_clean_reply(text, persona, member_names)` to handle both failures:

- A member's name in front of the **first** line is a mislabelled own turn. Drop
  the label, keep the text.
- A member's name in front of any **later** line means the model has moved on to
  someone else's turn. Cut everything from there.
- Never return an empty string.

<details>
<summary>Solution</summary>

```python
    def _clean_reply(self, text, persona, member_names):
        names = {n.lower() for n in member_names} | {persona.name.lower()}

        kept = []
        for index, line in enumerate(text.splitlines()):
            head, separator, rest = line.partition(":")
            if separator and head.strip().lower() in names:
                if index == 0:
                    kept.append(rest.strip())
                    continue
                break
            kept.append(line)

        text = "\n".join(kept).strip()
        return text or "…"
```

Check it against the cases that matter:

```python
>>> e._clean_reply("Pixel: Nice to meet you too, Mia.", ada, names)
'Nice to meet you too, Mia.'
>>> e._clean_reply("Good idea.\nMia: thanks!\nAda: welcome", ada, names)
'Good idea.'
>>> e._clean_reply("Sure: here is why that works.", ada, names)
'Sure: here is why that works.'
```

**Key points:**
- **Position, not identity, distinguishes the two failures.** The obvious
  implementation checks only for the persona's *own* name and throws away
  anything starting with someone else's — which, on the samples above, discards
  four good replies out of six and posts an empty bubble instead. A label on
  line 1 is a formatting mistake; a label on line 3 is a hallucinated
  conversation.
- **Only real member names count as labels.** `"Sure: here is why"` survives
  because `Sure` is nobody. Matching any word before a colon would mangle
  ordinary sentences.
- **Case-insensitive matching**, because the model varies capitalisation freely.
- **Repair happens outside the lock**, in `reply()`. It is pure string work and
  has no business holding up another bot's generation.
- **Never return `""`.** The room would post a blank bubble and no error
  anywhere. The `"…"` fallback is a visible, harmless failure — and if you see
  it often, that is a signal your persona prompts are drifting, not that the
  code is broken.
- **This is a fixed-up small model, and you should know its ceiling.** Even with
  clean output, Qwen2.5-0.5B holds a character loosely: Newton drops the
  17th-century register within a couple of turns, and all three personas
  converge on the same agreeable voice. The fix is not a better regex, it is a
  bigger model — `Qwen/Qwen2.5-1.5B-Instruct` is a drop-in change to
  `DEFAULT_MODEL_NAME` and holds a persona visibly better, at roughly three
  times the memory and generation time.

</details>

### Checkpoint — run it

```python
if __name__ == "__main__":
    engine = BotEngine()
    engine.load()

    ada = DEFAULT_PERSONAS[0]
    names = [p.name for p in DEFAULT_PERSONAS] + ["Sam", "Mia"]
    transcript = [{"speaker": "Sam", "text": "Mia! Good to see you here."}]

    # Call the internals directly so you can see the raw output next to the
    # repaired one — reply() only ever hands back the repaired version.
    for i in range(6):
        raw = engine._generate(engine._build_prompt(ada, transcript, names))
        print(f"{i}  RAW: {raw!r}\n   FIXED: {engine._clean_reply(raw, ada, names)!r}")
```

Run it a few times. You are looking for two things: labels being stripped rather
than swallowed, and no `'…'` unless the raw output really was nothing but
someone else's lines.

---

## Task 4 — The Terminal Demo

Replace the checkpoint with an `if __name__ == "__main__":` block that stays in
the file: build a hand-written transcript, then print what each persona says
next. Keep it — it is the fastest way to answer "is this a model problem or a
room problem?" for the rest of the workshop.

<details>
<summary>Solution</summary>

```python
if __name__ == "__main__":
    print("=== Bot Engine — terminal demo ===\n")

    engine = BotEngine()
    engine.load()

    transcript = [
        {"speaker": "Sam", "text": "Hi everyone! What should we build this weekend?"},
        {"speaker": "Ada", "text": "Something small you can finish. A weekend is shorter than it looks."},
        {"speaker": "Sam", "text": "Newton, what do you think?"},
    ]
    names = [p.name for p in DEFAULT_PERSONAS] + ["Sam"]

    print("\n--- transcript ---")
    for entry in transcript:
        print(f'{entry["speaker"]}: {entry["text"]}')

    print("\n--- each persona answers the same transcript ---")
    for persona in DEFAULT_PERSONAS:
        text, tokens = engine.reply(persona, transcript, names)
        print(f"\n{persona.name}: {text}   [{tokens} tokens of context]")
```

**Key points:**
- The transcript is a **literal**. No room, no threads, no HTTP — which is
  exactly what makes the engine easy to debug and easy to test.
- Because the engine is stateless, this demo is also the shape of a unit test.
  Swap the `print` for an `assert` and you have one.

</details>

---

# Part 2 — The Room

Everything in this part runs without Flask and, for the checkpoints, without the
model. `chat_room.py` owns the transcript, the membership list, the turn-taking
rules and one background thread.

---

## Task 5 — The Shared Transcript

Create `chat_room.py` with a `Message` class and the beginnings of a `ChatRoom`.

**`Message`** — five attributes: `id` (an integer), `speaker`, `text`, `is_bot`
and `ts` (`time.time()`). Plus `as_dict()`, returning all five. This is the wire
format, defined here rather than in `app.py`.

**`ChatRoom.__init__(engine, personas=DEFAULT_PERSONAS, context_messages=20,
max_bot_chain=2)`** — store the engine, build `self._bots` as a dict of
lower-cased name → Persona, and set up:

| Attribute | Purpose |
|---|---|
| `self._messages = []` | the shared transcript, oldest first |
| `self._next_id = 1` | the id counter; never reset |
| `self._humans = {}` | lower-cased name → display name |
| `self._pending = []` | bots queued or generating, shown as "typing…" |
| `self._bot_chain = 0` | consecutive bot messages since a human spoke |
| `self._epoch = 1` | bumped by `clear()` |
| `self._lock = threading.RLock()` | guards every attribute above |
| `self._queue = queue.Queue()` | bot turns waiting for the worker |

**`_append(speaker, text, is_bot)`** — create, number and store one message.
Callers must already hold the lock.

**`messages_since(last_id)`** — every message with a larger id, oldest first.

**`clear()`** — empty the transcript, reset the chain counter, and increment
`_epoch`. Leave `_next_id` alone.

<details>
<summary>Solution</summary>

```python
import queue
import re
import threading
import time
from bot_engine import BotEngine, DEFAULT_PERSONAS


class Message:
    def __init__(self, id, speaker, text, is_bot):
        self.id = id
        self.speaker = speaker
        self.text = text
        self.is_bot = is_bot
        self.ts = time.time()

    def as_dict(self):
        return {"id": self.id, "speaker": self.speaker, "text": self.text,
                "is_bot": self.is_bot, "ts": self.ts}


class ChatRoom:
    def __init__(self, engine, personas=DEFAULT_PERSONAS, context_messages=20, max_bot_chain=2):
        self._engine = engine
        self._bots = {p.name.lower(): p for p in personas}
        self.context_messages = context_messages
        self.max_bot_chain = max_bot_chain

        self._messages = []
        self._next_id = 1
        self._humans = {}
        self._pending = []
        self._bot_chain = 0
        self._epoch = 1

        self._lock = threading.RLock()
        self._queue = queue.Queue()
        self._worker = None

    def _append(self, speaker, text, is_bot):
        message = Message(self._next_id, speaker, text.strip(), is_bot)
        self._next_id += 1
        self._messages.append(message)
        return message

    def messages_since(self, last_id):
        with self._lock:
            return [m for m in self._messages if m.id > last_id]

    def clear(self):
        with self._lock:
            self._messages = []
            self._bot_chain = 0
            self._epoch += 1
```

**Key points:**
- **`id` is the entire synchronisation protocol.** Every client remembers the
  highest id it has drawn and asks for everything above it. That one integer is
  what lets five browsers, a background thread and a `curl` session all agree on
  what has happened, with no server push and no per-client state on the server.
- **Ids must therefore be monotonic forever — including across `clear()`.** If
  the counter restarted, a browser sitting at `since=42` would be handed a
  brand-new message numbered 7, decide it had seen it already, and drop it
  silently. Resetting a counter that clients hold copies of is one of the
  classic distributed-state bugs, and it is a one-line mistake.
- **`_epoch` is how a client notices a wipe.** Without it, a browser that
  cleared nothing keeps a transcript of messages the server has forgotten. The
  epoch is a version number for the whole log: when it changes, redraw from
  scratch.
- **The lock is an `RLock`, not a `Lock`.** `post()` acquires it and then calls
  `_wake()`, which needs it too. A plain `Lock` would deadlock the moment a
  human sent a message — and it would deadlock *silently*, with the request
  simply never returning.
- **`is_bot` is stored, not derived.** `speaker in self._bots` would give the
  same answer today, but the transcript is a historical record: if a bot is ever
  removed from the room, its old messages must not turn into human ones.
- **`as_dict()` lives on `Message`.** `app.py` never picks the transcript apart
  and never decides what a message looks like on the wire.

</details>

### Checkpoint — run it

No model needed, so this is instant:

```python
if __name__ == "__main__":
    room = ChatRoom(engine=None)
    with room._lock:
        room._append("Sam", "hello", is_bot=False)
        room._append("Ada", "hi Sam", is_bot=True)

    print([ (m.id, m.speaker) for m in room.messages_since(0) ])  # [(1, 'Sam'), (2, 'Ada')]
    print([ (m.id, m.speaker) for m in room.messages_since(1) ])  # [(2, 'Ada')]

    room.clear()
    with room._lock:
        room._append("Sam", "still here?", is_bot=False)
    print([ m.id for m in room.messages_since(0) ])               # [3] — not [1]
```

That last line is the point of the whole checkpoint. If it prints `[1]`, you
reset `_next_id` in `clear()`, and every client would lose the message.

---

## Task 6 — Members, and Who Answers

Two jobs: let humans in, and decide which bots respond to a given message.

**`join(name)`** — normalise whitespace, reject names outside 1–24 characters or
containing anything but `[\w .'-]`, reject a name that collides with a bot or
another human (case-insensitively), store it, and return the canonical form.
Raise `ValueError` with a readable message on every rejection.

**`members()`** — every member as a dict with `name`, `description` and
`is_bot`, bots first.

**`_member_names()`** — every name, for the engine. Caller holds the lock.

**`_mentioned_bots(text)`** — the bots named in a message. Accept both `@Ada`
and a bare `Ada`, matched as whole words, case-insensitively.

**`_least_recently_active()`** — the bot that has gone longest without speaking.
Walk the transcript backwards collecting bot speakers; a bot that has never
spoken wins outright, otherwise the one furthest back.

<details>
<summary>Solution</summary>

```python
    def join(self, name):
        name = " ".join(name.split())
        if not 1 <= len(name) <= 24:
            raise ValueError("Name must be between 1 and 24 characters.")
        if not re.fullmatch(r"[\w .'-]+", name):
            raise ValueError("Name may only contain letters, numbers, spaces and . ' -")

        with self._lock:
            key = name.lower()
            if key in self._bots:
                raise ValueError(f"{name} is one of the bots — pick another name.")
            if key in self._humans and self._humans[key] != name:
                raise ValueError(f"{name} is already taken.")
            self._humans[key] = name
            return name

    def members(self):
        with self._lock:
            bots = [{"name": p.name, "description": p.description, "is_bot": True}
                    for p in self._bots.values()]
            humans = [{"name": n, "description": "", "is_bot": False}
                      for n in self._humans.values()]
            return bots + humans

    def _member_names(self):
        return [p.name for p in self._bots.values()] + list(self._humans.values())

    def _mentioned_bots(self, text):
        words = {w.lower() for w in re.findall(r"@?([\w'-]+)", text)}
        return [p for key, p in self._bots.items() if key in words]

    def _least_recently_active(self):
        spoken = []
        for message in reversed(self._messages):
            key = message.speaker.lower()
            if message.is_bot and key not in spoken:
                spoken.append(key)
        for key in self._bots:
            if key not in spoken:
                return self._bots[key]
        return self._bots[spoken[-1]]
```

**Key points:**
- **Name collisions are a security question, not a tidiness one.** If a visitor
  can join as `Ada`, their messages are indistinguishable from the bot's in
  every transcript and every prompt — they can put words in Ada's mouth. Lower-
  casing both sides is what stops `ada`, `ADA` and `Ada` being three identities.
- **Re-joining with the exact same name is allowed on purpose**, so a browser
  that lost its session can take its name back. The room cannot tell that
  browser from a stranger, though, so a second visitor who types an existing
  member's exact name becomes that member too — fine on a trusted dev machine,
  and a gap Exercise 5 is the place to close.
- **Validation lives in the room, not in `app.py`.** The rules are about the
  room ("that name is taken"), not about HTTP. The web layer's job is only to
  turn the `ValueError` into a `400`.
- **Mentions accept bare names, not just `@`.** People write "Newton, what do
  you think?" far more often than they write `@Newton`, and a room where the
  bots ignore that feels broken. The cost is that a bot casually saying another
  bot's name counts as addressing it — which is what Task 7's chain limit is
  for.
- **Word-boundary matching matters**: a naive `if "ada" in text.lower()` fires
  on "adaptive", "Canada" and "Ada's". Splitting into words first avoids all
  three.
- **Least-recently-active spreads the conversation around.** The obvious
  fallbacks are worse: always waking the first bot means two of your three
  personas never speak unless addressed; waking *all* of them means one message
  costs three generations, 30+ seconds of CPU, and a wall of text nobody reads.
- **A bot that has never spoken wins outright**, so a new room does not open
  with the same bot answering the first three messages.

</details>

---

## Task 7 — Stopping the Bots From Talking Forever

This task has no equivalent anywhere in a one-to-one chatbot, and it is the one
that will bite you if you skip it.

Bots answering bots is a loop with no exit condition. Two agreeable personas
will happily fill the transcript until you kill the process — burning CPU,
filling memory, and pushing the human's message out of everyone's context
window. The room needs an explicit social policy.

Write `_wake(message)`, which queues the bots that should respond. Three rules:

1. **A bot never answers itself.**
2. **A bot answers another bot only when that bot named it.** Humans get the
   friendlier treatment: if a human addresses nobody, the least recently active
   bot picks the message up.
3. **At most `max_bot_chain` bot messages may follow one human message.** A
   human speaking resets the counter.

Also make sure a bot already in `self._pending` is never queued twice.

The caller holds the lock.

<details>
<summary>Solution</summary>

```python
    def _wake(self, message):
        if message.is_bot:
            self._bot_chain += 1
            if self._bot_chain >= self.max_bot_chain:
                return
            targets = self._mentioned_bots(message.text)
        else:
            self._bot_chain = 0
            targets = self._mentioned_bots(message.text)
            targets = targets or [self._least_recently_active()]

        for persona in targets:
            if persona.name == message.speaker:
                continue
            if persona.name in self._pending:
                continue
            self._pending.append(persona.name)
            self._queue.put((persona.name, self._epoch))
```

**Key points:**
- **The asymmetry between humans and bots is the whole design.** An unaddressed
  human message still gets an answer, because silence would look broken. An
  unaddressed bot message gets nothing, because that is where runaway
  conversations start. Same function, deliberately different defaults.
- **The chain counter is a backstop, not the main rule.** Rule 2 already stops
  most loops; rule 3 catches the case rule 2 cannot, which is two bots that
  genuinely keep naming each other. Belt and braces, because the failure mode is
  an unbounded loop on a machine you are also using.
- **`max_bot_chain=2` is a taste setting, not a safety one.** Raise it to 4 and
  the room gets livelier and considerably slower; drop it to 1 and bots never
  talk to each other at all. Try both.
- **Checking `_pending` prevents double-posting.** Two members mentioning `@Ada`
  a second apart would otherwise queue her twice, and she would answer the same
  moment in the conversation twice, seconds apart.
- **The epoch is captured at queue time**, so a turn queued before someone
  pressed Clear can be recognised as stale and dropped (Task 8).
- **`_wake` decides, it does not generate.** It appends a name and returns in
  microseconds — which is what lets it be called from inside the lock, and from
  inside a request handler.

</details>

### Checkpoint — run it, with no model at all

The rules are pure logic, so test them against a fake engine. This runs in
milliseconds and is the single most useful checkpoint in the workshop.

```python
if __name__ == "__main__":
    import time
    from bot_engine import Persona

    class FakeEngine:
        """Stands in for BotEngine. Every bot answers by naming the next bot."""
        is_ready = True
        def __init__(self, script): self.script = script; self.calls = []
        def reply(self, persona, transcript, member_names=()):
            self.calls.append(persona.name)
            return self.script.get(persona.name, "ok"), 7

    personas = [Persona("Ada", "", "You are Ada."),
                Persona("Newton", "", "You are Newton."),
                Persona("Pixel", "", "You are Pixel.")]

    def drain(room, timeout=3):
        end = time.time() + timeout
        while time.time() < end:
            with room._lock:
                if not room._pending: return
            time.sleep(0.02)
        raise AssertionError("bots never went quiet")

    # Rule 3: a chain of bots naming each other stops after max_bot_chain.
    engine = FakeEngine({"Ada": "good question @Newton", "Newton": "indeed @Pixel", "Pixel": "yay @Ada"})
    room = ChatRoom(engine, personas, max_bot_chain=2); room.start(); room.join("Sam")
    room.post("Sam", "@Ada hello"); drain(room)
    print([m.speaker for m in room.messages_since(0)])

    # Rule 1: a bot mentioning itself does not loop.
    engine2 = FakeEngine({"Ada": "talking to myself @Ada"})
    room2 = ChatRoom(engine2, personas, max_bot_chain=5); room2.start(); room2.join("Sam")
    room2.post("Sam", "@Ada hi"); drain(room2)
    print(engine2.calls)

    # Rule 2: bots ignore each other unless named.
    engine3 = FakeEngine({"Ada": "no names here"})
    room3 = ChatRoom(engine3, personas, max_bot_chain=5); room3.start(); room3.join("Sam")
    room3.post("Sam", "@Ada hi"); drain(room3)
    print(engine3.calls)
```

```
['Sam', 'Ada', 'Newton']     ← Pixel never speaks: the chain was capped at 2
['Ada']                      ← the engine ran once, not forever
['Ada']                      ← the chain stopped for lack of a mention
```

This needs Task 8's `start()` and `post()` to run. Write them, then come back —
if any of these three prints grows without bound, stop and fix `_wake()` before
going near a web server.

---

## Task 8 — Posting Without Waiting

Now connect the pieces with a background worker, so that sending a message costs
milliseconds instead of the 10–15 seconds a reply takes.

**`start()`** — create a daemon `threading.Thread` running `self._work` and
start it. Guard against being called twice.

**`post(speaker, text)`** — under the lock: append the message, call `_wake()`,
return the `Message`. It does not wait for anything.

**`_work()`** — an endless loop: pull `(persona_name, epoch)` off the queue, call
`_speak()`, and no matter what happens remove the bot from `_pending`. Catch and
log every exception.

**`_speak(persona_name, epoch)`** — take a snapshot under the lock (the persona,
the last `context_messages` messages converted to `{"speaker", "text"}` dicts,
and the member names), **release the lock**, call `self._engine.reply(...)`, then
re-acquire the lock to append the result and call `_wake()` on it. Return early
if `epoch != self._epoch`, both before and after generating.

**`state_since(last_id)`** — one snapshot for the polling endpoint: `messages`
(as dicts), `typing`, `members` and `epoch`.

<details>
<summary>Solution</summary>

```python
    def start(self):
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._work, name="bot-worker", daemon=True)
        self._worker.start()

    def post(self, speaker, text):
        with self._lock:
            message = self._append(speaker, text, is_bot=False)
            self._wake(message)
            return message

    def state_since(self, last_id):
        with self._lock:
            return {
                "messages": [m.as_dict() for m in self._messages if m.id > last_id],
                "typing": list(self._pending),
                "members": self.members(),
                "epoch": self._epoch,
            }

    def _work(self):
        while True:
            persona_name, epoch = self._queue.get()
            try:
                self._speak(persona_name, epoch)
            except Exception as exc:
                print(f"[bot-worker] {persona_name} failed: {exc!r}")
            finally:
                with self._lock:
                    if persona_name in self._pending:
                        self._pending.remove(persona_name)

    def _speak(self, persona_name, epoch):
        with self._lock:
            if epoch != self._epoch:
                return
            persona = self._bots[persona_name.lower()]
            recent = self._messages[-self.context_messages:]
            transcript = [{"speaker": m.speaker, "text": m.text} for m in recent]
            names = self._member_names()

        if not transcript:
            return

        text, _tokens = self._engine.reply(persona, transcript, names)

        with self._lock:
            if epoch != self._epoch:
                return
            message = self._append(persona.name, text, is_bot=True)
            self._wake(message)
```

**Key points:**
- **The lock is released across the slow call, and that is the crux of
  `_speak()`.** Generation takes 10–15 seconds. Holding the lock through it
  would block every browser poll, every new message and every join for that
  entire time — the exact freeze the background thread exists to avoid. So:
  snapshot under the lock, generate outside it, append under it again.
- **Snapshotting means the world can change underneath you**, and the code has
  to expect that. Three members may post while Ada is thinking; her reply then
  lands after theirs and answers a question that has scrolled past. That is not
  a bug to fix — it is what a real group chat looks like when someone types
  slowly. The `epoch` check is there for the one case that *is* a bug: the room
  being wiped mid-generation, where posting the reply would resurrect a deleted
  conversation.
- **One worker, not a pool.** The engine serialises on its own lock anyway, so
  extra workers would only queue there — while making transcript order
  unpredictable. Serial generation also keeps memory flat.
- **`daemon=True`** lets `Ctrl+C` actually exit. A non-daemon thread blocked on
  `queue.get()` keeps the process alive forever.
- **The `try/except` around `_speak` is load-bearing.** An uncaught exception
  kills the worker thread, and the symptom is not a crash — it is bots that go
  permanently, silently mute while the web server carries on serving pages. Log
  it and keep the loop alive.
- **The `finally` block is equally load-bearing.** Without it, a failed
  generation leaves the bot in `_pending` forever, so the UI shows "Ada is
  writing…" for the rest of the session and `_wake` refuses to queue her again.
- **`post()` returns a `Message`, not a reply.** This is the API shape that
  makes the whole design work: sending is fast and answers arrive later, through
  the same path as everyone else's messages.
- **`state_since()` bundles four things into one response** so the frontend
  needs one request per tick instead of four.

</details>

### Checkpoint — the room in a terminal

Replace the Task 7 checkpoint with a real terminal chat. This is the room's
equivalent of `python chatbot_engine.py`: the full experience, no web server.

```python
if __name__ == "__main__":
    print("=== Chat Room — terminal demo ===\n")

    engine = BotEngine()
    engine.load()

    room = ChatRoom(engine)
    room.start()
    me = room.join("Sam")

    print("\nYou are Sam. Mention a bot by name to address it (@Ada, @Newton, @Pixel).")
    print("Bot replies arrive a few seconds after your message. Type 'quit' to exit.\n")
    print("In the room:", ", ".join(m["name"] for m in room.members()), "\n")

    seen = 0
    while True:
        text = input(f"{me}: ").strip()
        if not text:
            continue
        if text.lower() in ("quit", "stop", "exit"):
            print("\nLeaving the room.\n")
            break

        room.post(me, text)

        # Poll until the room goes quiet, the way the browser will.
        while True:
            for message in room.messages_since(seen):
                seen = message.id
                if message.speaker != me:
                    print(f"  {message.speaker}: {message.text}")
            with room._lock:
                busy = bool(room._pending)
            if not busy:
                break
            time.sleep(0.5)
        print()
```

```
Sam: hi everyone, @Newton what is gravity in one sentence?

  Newton: Gravity is the universal force that causes objects on Earth to attract each other.

Sam: should we meet on Saturday or Sunday?

  Ada: Sure, let's meet at 10 AM on Saturday.
```

Note the second exchange: nobody was addressed, and exactly one bot — the least
recently active — picked it up. If `room.start()` is missing, your message is
accepted, `_pending` fills up, and nothing ever answers.

> The demo reaches into `room._pending` through the private attribute. That is
> acceptable in a demo at the bottom of the same file; `app.py` will go through
> `state_since()` instead.

---

# Part 3 — The Web Layer

`app.py` translates HTTP into room calls and back. It contains no model code and
no turn-taking logic. Compared to the Web Chatbot's `app.py` it gains one route
that has no one-to-one equivalent at all: `GET /messages`.

---

## Task 9 — A Server, and Who You Are

Create `app.py` with the application object, the two module-level singletons,
and identity handling.

- Build `app`, set `app.secret_key` from `FLASK_SECRET_KEY` with a fixed dev
  default, and set `MAX_MESSAGE_CHARS = 500`.
- Construct `engine = BotEngine()` and `room = ChatRoom(engine)` at import time.
  Neither loads anything yet.
- `current_member()` — return `session.get("name")`, or `None`.
- `GET /` — render `index.html`, passing `model_name` and `member`.
- `GET /health` — status, `model_loaded`, and the member names.
- `POST /join` — read `{"name": ...}`, hand it to `room.join()`, store the
  canonical name in the session, return it. Turn `ValueError` into a `400`.
- `POST /leave` — drop the name from the session.

<details>
<summary>Solution</summary>

```python
import os
from flask import Flask, jsonify, render_template, request, session
from bot_engine import BotEngine
from chat_room import ChatRoom

MAX_MESSAGE_CHARS = 500

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-insecure-key")

engine = BotEngine()
room = ChatRoom(engine)


def current_member():
    return session.get("name")


@app.get("/")
def index():
    return render_template("index.html", model_name=engine.model_name, member=current_member())


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model_loaded=engine.is_ready,
        members=[m["name"] for m in room.members()],
    )


@app.post("/join")
def join():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()

    try:
        name = room.join(name)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    session["name"] = name
    return jsonify(name=name)


@app.post("/leave")
def leave():
    session.pop("name", None)
    return jsonify(status="left")
```

**Key points:**
- **The session holds a name, not a conversation id.** In the Web Chatbot the
  cookie carried a random `sid` that selected *which private history* to use.
  Here there is one shared room and the cookie answers a different question:
  *who are you in it?* The identity is the only thing the web layer contributes
  that the room cannot work out for itself.
- **`app.secret_key` signs the cookie**, which is what stops a visitor editing
  their own name to `Ada` and impersonating a bot. Signed is not encrypted: the
  name is readable by the client, just not forgeable. Never put a secret in a
  session.
- **`session` is a cookie, so it survives a server restart** — with the fixed
  dev key, a browser that has joined stays joined, which is convenient while
  developing and is why the key is hardcoded rather than randomised at startup.
  The room's `_humans` dict does *not* survive, so a returning browser can name
  itself something the room has forgotten. Exercise 5 closes that gap.
- **`room.join()` does the validating.** The route's only job is turning a
  `ValueError` into a `400` with the room's own message.
- **`/health` reports `members`**, which is the fastest way to see whether joins
  are landing where you think they are while debugging.

</details>

---

## Task 10 — Sending and Polling

These two routes share a URL and split the work of a group chat between them:
`POST /messages` puts something in, `GET /messages` finds out what happened.

**`POST /messages`** — require a member (`401` if the session has no name),
validate `{"text": ...}` (`400` for empty or over `MAX_MESSAGE_CHARS`), return
`503` if the model has not loaded, then `room.post(...)` and return the stored
message with `201`.

**`GET /messages?since=<id>`** — parse `since` as an integer (`400` if it is
not), and return `room.state_since(since)`.

<details>
<summary>Solution</summary>

```python
@app.get("/messages")
def messages():
    try:
        since = int(request.args.get("since", 0))
    except ValueError:
        return jsonify(error="'since' must be an integer."), 400

    return jsonify(room.state_since(since))


@app.post("/messages")
def send():
    member = current_member()
    if not member:
        return jsonify(error="Join the room first."), 401

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify(error="Message must not be empty."), 400
    if len(text) > MAX_MESSAGE_CHARS:
        return jsonify(error=f"Message must be at most {MAX_MESSAGE_CHARS} characters."), 400

    if not engine.is_ready:
        return jsonify(error="Model is still loading, try again shortly."), 503

    message = room.post(member, text)
    return jsonify(message.as_dict()), 201
```

Test both without a browser. `-c` saves cookies, `-b` sends them back:

```bash
curl -c jar.txt -X POST http://127.0.0.1:5000/join \
  -H "Content-Type: application/json" -d '{"name": "Mia"}'
# {"name":"Mia"}

curl -b jar.txt -X POST http://127.0.0.1:5000/messages \
  -H "Content-Type: application/json" -d '{"text": "Hi all! @Pixel what should we cook tonight?"}'
# {"id":1,"is_bot":false,"speaker":"Mia","text":"Hi all! ...","ts":1788337571.97}

curl "http://127.0.0.1:5000/messages?since=0"
# {"epoch":1,"members":[...],"messages":[{"id":1,...}],"typing":["Pixel"]}

# …ten seconds later
curl "http://127.0.0.1:5000/messages?since=1"
# {"epoch":1,...,"messages":[{"id":2,"is_bot":true,"speaker":"Pixel",
#   "text":"Why not try some homemade pizza tonight? ..."}],"typing":[]}
```

**Key points:**
- **The reply is not in the response, and that is the headline.** In the Web
  Chatbot, `POST /chat` returned `{"reply": ...}` — the request *was* the
  conversation. Here `POST /messages` returns only your own message, in
  milliseconds. Watch it in the `curl` session above: the POST comes back
  instantly with `typing: ["Pixel"]`, and Pixel's actual words arrive in a later
  `GET`. That is what stops one member's message freezing the page for everyone.
- **`GET /messages` exists because most messages are not yours.** A private
  chatbot needs no such route: nothing can happen that you did not cause. In a
  group, another member's message and a bot's reply are both events you did not
  trigger, and polling is the simplest thing that can discover them.
- **`since` is the client's memory, so the server keeps none.** No per-client
  cursors, no subscriptions, no cleanup when a browser closes its laptop lid.
  Five browsers and a `curl` command all work, and a client that has been asleep
  for an hour catches up in one request.
- **`401` versus `400`.** "You are not in the room" is a different problem from
  "your message was empty", and the frontend reacts differently: `401` reopens
  the join screen, `400` shows an error. Collapsing them into one status makes
  that impossible.
- **`201 Created`, not `200 OK`.** The request added a resource to the
  transcript. Status codes are documentation that survives your README.
- **Both routes stay tiny.** Validate, delegate, translate to HTTP. Every
  interesting decision — who answers, when the bots stop, what counts as a
  mention — was made in Part 2, and none of it is repeated here.

</details>

---

## Task 11 — Clearing the Room

In the Web Chatbot, `POST /reset` cleared *your* history. Here there is one
transcript, so this is a genuinely different operation: it wipes the
conversation for everybody who is looking at it.

Write `POST /reset`: require a member, call `room.clear()`, return a
confirmation. Then finish the file with a `__main__` block that loads the model,
starts the worker, and runs the server.

<details>
<summary>Solution</summary>

```python
@app.post("/reset")
def reset():
    if not current_member():
        return jsonify(error="Join the room first."), 401
    room.clear()
    return jsonify(status="cleared")


if __name__ == "__main__":
    print(f"=== Community Chat Tutorial — Flask + {engine.model_name} ===\n")

    engine.load()
    room.start()

    print("Open http://127.0.0.1:5000 in your browser.")
    print("Open it a second time in a private window to join as another member.\n")

    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=True,
        use_reloader=False,
        threaded=True,
    )
```

**Key points:**
- **Shared state makes "reset" a social act.** Nothing here is per-caller: one
  member presses Clear and four other browsers lose the transcript. The `epoch`
  from Task 5 is what lets them find out — they see a number they do not
  recognise and redraw. If this were a real product, Clear would need a
  confirmation dialog and probably a permission; the version here is deliberately
  the simplest thing that demonstrates shared-state invalidation.
- **`room.start()` must be called exactly once, next to `engine.load()`.**
  Forgetting it produces the most confusing failure in the whole workshop: the
  site works, messages send, `typing` fills up with names — and no bot ever
  speaks, with nothing in the log to explain why.
- **`threaded=True` is load-bearing here**, though it is also the default. The
  worker thread must be able to append to the transcript while Flask serves
  polls, and browsers must be able to poll while another request is in flight.
- **`use_reloader=False`** because auto-restart would reload the model *and*
  throw away the room on every save. Restart by hand.
- **`debug=True` never goes near a public server.** The Werkzeug console is
  remote code execution by design, and this app has no authentication at all.

</details>

---

## Task 12 — The Browser Frontend

Write `templates/index.html`: a join screen, a member list, a scrolling
transcript, an input box, and a Clear button. No frameworks, no CDN — plain
HTML, CSS and JavaScript in one file.

Requirements:

1. **Two panes, one page.** Flask decides which is visible on first render from
   the `member` variable; JavaScript switches to the chat pane after a
   successful join.
2. **Poll every 1.5 seconds** with `?since=<highest id drawn>`, and draw
   everything that comes back.
3. **Label incoming messages.** With five speakers, left-versus-right is not
   enough: show the sender's name, mark bots with a `BOT` tag, and give each
   speaker a stable colour derived from their name.
4. **Show who is writing**, from the `typing` array.
5. **Redraw from scratch when `epoch` changes.**
6. **Send without drawing a reply.** The POST response is not rendered; your own
   message arrives through the next poll, exactly like everyone else's.
7. **Insert text with `.textContent`**, never `.innerHTML`.
8. Handle a `401` by reopening the join screen.

<details>
<summary>Solution — the JavaScript</summary>

```javascript
    // Rendered by Jinja: "Sam" once this browser has joined, otherwise null.
    // |tojson quotes and escapes it — writing {{ member }} raw would break the
    // script on any name containing a quote.
    let me = {{ member|tojson }};

    let lastId = 0;      // The highest message id this page has rendered
    let epoch = null;    // Bumped by the server when someone clears the room
    let typingEl = null; // The single "… is writing" line at the bottom

    // Give every speaker a stable colour derived from their name, so the same
    // person is the same colour in every browser without the server sending one.
    function colorFor(name) {
      let hash = 0;
      for (const ch of name) hash = (hash * 31 + ch.charCodeAt(0)) % 360;
      return `hsl(${hash}, 70%, 55%)`;
    }

    function addMessage(msg) {
      const row = document.createElement("div");
      row.className = "row " + (msg.speaker === me ? "mine" : "theirs");

      if (msg.speaker !== me) {
        const who = document.createElement("div");
        who.className = "who";
        who.textContent = msg.speaker;
        if (msg.is_bot) {
          const tag = document.createElement("span");
          tag.className = "tag";
          tag.textContent = " BOT";
          who.appendChild(tag);
        }
        row.appendChild(who);
      }

      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.style.setProperty("--speaker", colorFor(msg.speaker));
      bubble.textContent = msg.text;      // Never .innerHTML
      row.appendChild(bubble);

      transcript.appendChild(row);
      return row;
    }

    // One poll: ask for everything newer than lastId and draw it.
    async function poll() {
      if (!me) return;
      try {
        const res = await fetch(`/messages?since=${lastId}`);
        if (!res.ok) return;
        const data = await res.json();

        if (epoch !== null && data.epoch !== epoch) {   // Somebody pressed Clear
          transcript.replaceChildren();
          typingEl = null;
          addNote("The room was cleared.", "system");
        }
        epoch = data.epoch;

        const atBottom = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 60;
        if (typingEl) { typingEl.remove(); typingEl = null; }

        for (const msg of data.messages) {
          addMessage(msg);
          lastId = Math.max(lastId, msg.id);   // Only ever moves forward
        }
        renderMembers(data.members);
        renderTyping(data.typing);

        if (atBottom || data.messages.some(m => m.speaker === me)) {
          transcript.scrollTop = transcript.scrollHeight;
        }
      } catch (err) {
        status.textContent = "offline";
      }
    }

    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = textInput.value.trim();
      if (!text) return;

      textInput.value = "";
      sendBtn.disabled = true;
      try {
        // Note what is *not* here: no reply is drawn from this response. Both
        // your message and every bot answer arrive through poll().
        await postJSON("/messages", { text });
        await poll();
      } catch (err) {
        if (err.message.includes("Join the room")) {   // 401 — session expired
          me = null;
          chatPane.classList.add("hidden");
          joinPane.classList.remove("hidden");
        } else {
          addNote(err.message, "error");
        }
      } finally {
        sendBtn.disabled = false;
        textInput.focus();
      }
    });

    setInterval(poll, 1500);

    // Browsers throttle timers in hidden tabs — Chrome down to roughly once a
    // minute — so a backgrounded chat quietly stops updating. Polling again the
    // moment the tab is shown means the reader never sees a stale room.
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) poll();
    });

    poll();
```

The full file, including the CSS and the member chips, is in
[`templates/index.html`](templates/index.html).

**Key points:**
- **`lastId` is the entire client state.** One integer decides what to ask for
  and what to draw, and it only ever increases — so a message is never drawn
  twice, even if two polls overlap or the POST handler calls `poll()` at the
  same moment the timer does.
- **Your own message takes the same path as everyone else's.** Rendering it from
  the POST response would be one line shorter and would create two code paths
  that drift apart — the classic version of this bug is your message appearing
  twice, once optimistically and once from the next poll.
- **Poll immediately after sending**, or your own message sits invisible for up
  to 1.5 seconds and the room feels broken.
- **Speaker colour is computed, not assigned.** The same name hashes to the same
  hue in every browser, so nobody has to coordinate a palette and new members
  need no server support.
- **Only auto-scroll if the reader was already at the bottom**, or reading back
  through the log gets yanked away every time a bot finishes a sentence.
- **`{{ member|tojson }}`, not `"{{ member }}"`.** Jinja's `tojson` emits a
  valid JavaScript literal — including `null` — and escapes quotes. Interpolating
  a name straight into a string literal breaks the script for anyone called
  `O'Brien`, and is how script injection gets in.
- **1.5 seconds is a compromise**, not a magic number. Faster feels live and
  hammers a server that is busy generating; slower is calmer and makes the room
  feel laggy. Exercise 3 replaces polling with server-sent events and the
  question disappears.
- **`visibilitychange` is not decoration.** Chrome throttles `setInterval` in
  hidden tabs to about once a minute — this workshop's polling really does stall
  when you switch tabs, and re-polling on focus is the cheap fix.

</details>

---

## Running the Server

**Start it** from inside the `group_chat/` folder:

```bash
python app.py
```

```
=== Community Chat Tutorial — Flask + Qwen/Qwen2.5-0.5B-Instruct ===

Loading Qwen/Qwen2.5-0.5B-Instruct (downloads ~1 GB on first run)...
Model ready.
Open http://127.0.0.1:5000 in your browser.
Open it a second time in a private window to join as another member.

 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

**Try it properly:** open the page twice — a normal window and a private one, so
they get different cookies — and join as two different names. Watch a message
typed in one window appear in the other, and a bot answer both.

**Stop it:** `Ctrl+C`.

**The edit-run loop:**

| You changed | What to do |
|---|---|
| `templates/index.html` | Refresh the browser — templates reload on every request in debug mode |
| CSS or JavaScript inside the template | Hard-refresh (`Cmd/Ctrl+Shift+R`) to defeat the browser cache |
| `app.py` | `Ctrl+C` and re-run — this reloads the model and empties the room |
| `chat_room.py` | Test with `python chat_room.py` (or the fake-engine checkpoint) first |
| `bot_engine.py` | Test with `python bot_engine.py` first, then restart the server |

**Change the port** if 5000 is taken. On macOS, AirPlay Receiver listens there:

```bash
PORT=5001 python app.py
```

---

## Debugging While You Develop

Three modules give you a decision procedure with two questions instead of one:

```bash
python bot_engine.py    # Is the model producing sensible text?
python chat_room.py     # Are the rules and the worker behaving?
```

If a bug shows up in the first, it is a prompt or a cleaning problem. If it only
shows up in the second, it is turn-taking or threading. If neither reproduces
it, it is `app.py` or the frontend. That splits three files into "the one it is
in" in about twenty seconds.

**The fake engine is your best instrument.** Every rule in Part 2 is pure logic
over strings, so you can drive the whole room at full speed with the
`FakeEngine` from Task 7's checkpoint — scripted replies, no model, instant
results. Loop bugs in particular are almost impossible to study at 12 seconds
per message and trivial at 12 microseconds.

### The four instruments

**1. The terminal** — Flask logs one line per request. In this app the polls are
the interesting part:

```
127.0.0.1 - - [02/Sep/2026 10:28:04] "POST /messages HTTP/1.1" 201 -
127.0.0.1 - - [02/Sep/2026 10:28:05] "GET /messages?since=0 HTTP/1.1" 200 -
127.0.0.1 - - [02/Sep/2026 10:28:07] "GET /messages?since=1 HTTP/1.1" 200 -
```

If `since` stops climbing, the client is not recording ids. If the `GET` lines
stop entirely, the browser tab is hidden and its timer has been throttled.

**2. `/health` and `?since=0`** — the two windows into server state:

```bash
curl http://127.0.0.1:5000/health
curl "http://127.0.0.1:5000/messages?since=0"   # the whole transcript, plus typing and epoch
```

`typing` is the one to watch. A name stuck there for more than 30 seconds means
a generation failed and the `finally` block in `_work()` is missing or broken.

**3. `curl` with a cookie jar** — join as a second member from the command line
and talk to the browser session:

```bash
curl -c jar.txt -X POST http://127.0.0.1:5000/join \
  -H "Content-Type: application/json" -d '{"name": "Sam"}'
curl -b jar.txt -X POST http://127.0.0.1:5000/messages \
  -H "Content-Type: application/json" -d '{"text": "hello from the terminal"}'
```

This is the fastest multi-member test there is, and it needs no second browser.

**4. Browser DevTools (F12)** — **Network** shows the poll loop: click any
`/messages` row and compare **Payload** with **Response**. **Application →
Cookies** should show one `session` cookie; no cookie means `app.secret_key` is
missing.

### Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Messages send, `typing` fills up, no bot ever speaks | `room.start()` never called | Call it next to `engine.load()` in `__main__` |
| A bot is stuck "writing…" forever | Generation raised and `_pending` was never cleaned | Put the removal in a `finally` block in `_work()` |
| Bots talk to each other endlessly | `max_bot_chain` never checked, or the counter never resets | Re-read `_wake()`; test with the fake engine |
| The whole app freezes for 15 seconds per message | The lock is held across `engine.reply()` | Snapshot under the lock, generate outside it |
| The app deadlocks on the first message | `threading.Lock` instead of `RLock`, and `post()` calls `_wake()` | Use `threading.RLock()` |
| Bot replies are empty or `…` | Cleaning removed everything | See Task 3 — a label on line 1 must be stripped, not obeyed |
| A bot answers as another bot's name | Normal for a 0.5B model | `_clean_reply()` strips it; a 1.5B model does it far less |
| Bot ignores the question and answers an older one | It was addressed before the newer messages arrived | Working as designed — the snapshot is taken when the turn is queued |
| Messages appear twice | Rendering the POST response *and* the poll | Draw only in `poll()` |
| New messages never appear | `lastId` not updated, or `since` sent as a string | Log the `GET` line in the terminal and watch `since` climb |
| Nothing updates while the tab is in the background | Chrome throttles timers in hidden tabs | Re-poll on `visibilitychange` |
| One browser still shows messages another cleared | `epoch` not compared | Redraw when `epoch` changes |
| `RuntimeError: The session is unavailable because no secret key was set` | `app.secret_key` not set | Set it before the first request |
| Everyone is joined as the same person | Two windows sharing cookies | Use a private window for the second member |
| `401 Join the room first` after restarting the server | Cookie survived; the room's `_humans` did not | Re-join — or fix it properly with Exercise 5 |
| Replies take 10–15 s each, longer with two bots | Normal — CPU generation, serialised on one worker | Lower `max_new_tokens`; see Exercise 2 |
| `Address already in use` | Port 5000 taken (macOS AirPlay Receiver) | `PORT=5001 python app.py` |

---

## Exercises

1. **Give a bot a job, not a personality.** Add a fourth persona that summarises
   the last ten messages when someone types `@Summary`. Decide whether the
   summary should be posted to the room (everyone sees it, and it enters every
   later prompt) or returned only to the asker — and notice that the second
   option does not fit the current design at all. What would have to change?
2. **Show the bots' thinking time.** Record how long `engine.reply()` takes in
   `_speak()`, store it on the `Message`, and display it under bot bubbles.
   Then use it: does a longer transcript make replies slower? By how much?
3. **Replace polling with server-sent events.** Add a `GET /stream` route that
   holds the connection open and pushes each new message as it is appended, and
   have the frontend use `EventSource`. Keep `?since=` working for reconnects —
   an `EventSource` that drops must be able to catch up.
4. **Rate-limit the room.** One member spamming messages can queue every bot and
   monopolise the worker. Add a per-member cooldown. Decide carefully which
   module it belongs in: is "one message every two seconds" a room rule or an
   HTTP concern?
5. **Make membership survive a restart.** A browser's cookie outlives the
   server, so after a restart it holds a name the room has never heard of.
   Re-join automatically from the cookie, or persist `_humans` — and while you
   are there, expire members who have not polled in five minutes so the member
   list stops growing forever.
6. **Let a human speak as a bot — and then stop them.** Try joining as `ada`
   (lower case) and watch the collision check reject it. Now remove the
   `.lower()` calls in `join()` and see what an impersonated bot does to the
   transcript and to every later prompt. Put them back.
7. **Two rooms.** Turn the single global `room` into a dict of rooms keyed by a
   name in the URL (`/r/python`, `/r/cooking`). The bots and the engine are
   shared; only the transcript is not. If Part 2's boundaries are right, `app.py`
   is the only file that changes.

---

## Beyond This Workshop

This is a development setup, not a deployment. Four things would have to change
before it faced real users — and the first two are new problems that the Web
Chatbot did not have:

- **The room lives in one process's memory.** `app.run()` is single-process, so
  a shared dict works. Run two `gunicorn` workers and you have two rooms that
  cannot see each other, each with its own worker thread and its own copy of the
  model. Shared state across processes means a database or Redis, and a
  transcript in a database means the `_lock` becomes a transaction.
- **One worker, one CPU, three bots.** Every bot message is 10–15 seconds of a
  core. A dozen active members would queue for minutes. Real deployments put
  generation behind a job queue on separate hardware — which is exactly the
  shape `_queue` and `_work()` already have, so the change is smaller than it
  sounds.
- **Nothing here is moderated or authenticated.** Anyone who can reach the page
  can join under any free name, read the whole transcript and clear it for
  everybody. A real community chat needs accounts, permissions, and a way to
  deal with what people — and models — say.
- **`debug=True` and the dev secret key must both go.** The debugger is remote
  code execution, and a published secret key lets anyone forge a membership.
