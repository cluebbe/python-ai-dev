# Workshop: Web Chatbot with Flask and Qwen2.5

---

## Introduction

### Background

This workshop takes a chatbot that runs in a terminal and puts it on the web. By
the end you will have a Flask server that loads a language model once at
startup, exposes a small JSON API, and serves a browser page that talks to it —
with each visitor getting their own private conversation.

The language model is **Qwen2.5-0.5B-Instruct**, a small instruction-tuned model
that runs locally via the Hugging Face `transformers` library. It is Apache-2.0
licensed and ungated, so no cloud API, account, or access token is required —
unlike Llama, which needs a Hugging Face login and a signed license before it
will download.

> **Why an instruction-tuned model?** An obvious alternative is DialoGPT, an
> older GPT-2-based model fine-tuned on Reddit threads. It is smaller and
> faster, but it was trained only to produce a *plausible next reply*, never to
> answer questions or stay consistent. Asked "are you a girl or a guy" it will
> happily say "Girl" and then "I am a guy" two turns later, and asked "what is a
> president" it answers "I am a woman". Instruction tuning is what fixes that,
> and it costs roughly 2.5x in generation time on CPU.

The workshop is self-contained: every piece of model code you need is spelled
out here, so you can follow it without having done the terminal chatbot first.

### What Changes When a Chatbot Moves to the Web

A terminal chatbot is a single loop with a single user. A web server is neither.
Four assumptions break, and much of this workshop is about the four fixes:

| Terminal assumption | What the web does instead | Fix |
|---|---|---|
| One conversation exists | Many browsers connect at once | Key history by conversation ID (Task 3) |
| One thing happens at a time | The dev server runs threads in parallel | Serialise with a lock (Task 3) |
| Code runs top to bottom | Requests arrive in any order | Load the model once, up front (Task 1) |
| A crash is visible in the console | A crash is a blank page for the user | Return JSON errors with status codes (Task 8) |

There is a fifth difference that is not about correctness but about feel: on a
CPU this model takes **10–15 seconds** to answer. A terminal user watching a
cursor accepts that; a browser user staring at a frozen page assumes it broke.
Task 10 addresses it with an immediate placeholder bubble and a disabled Send
button, and the streaming exercise at the end addresses it properly.

### Why Two Modules Instead of One?

It is tempting to put all of this in a single `app.py`. Don't. The code has two
jobs that change for completely unrelated reasons:

- **Chatbot logic** — loading the model, tracking conversation history,
  generating replies. This changes when you swap models or tune sampling.
- **Web logic** — routes, request validation, session cookies, status codes.
  This changes when you add an endpoint or change the API shape.

So this project splits them:

```
chatbot_engine.py   ← imports torch + transformers. Never imports Flask.
app.py              ← imports Flask + chatbot_engine. Never imports torch.
```

The dependency arrow points one way: **the web layer knows about the engine, and
the engine knows nothing about the web.** Concretely, that buys you four things:

1. **You can run and test the engine with no web server.** `python
   chatbot_engine.py` opens a terminal chat using the exact same class the
   server uses. When a reply looks wrong, you can reproduce it without HTTP,
   cookies, or a browser anywhere in the picture.
2. **Debugging gets a decision procedure.** "Is this a model bug or a web bug?"
   stops being guesswork — reproduce it in the terminal demo. If it happens
   there, it is the engine. If not, it is the web layer.
3. **The engine is unit-testable.** Testing `engine.reply("test-1", "hi")`
   needs no Flask test client and no request context.
4. **Both files stay small enough to hold in your head.** Roughly 230 and 110
   lines, each about one topic.

The boundary between them is a single idea: the engine identifies conversations
by an **opaque string ID** and does not care where it came from. The web layer
mints one per browser from a session cookie; the terminal demo just uses the
literal string `"terminal"`.

> **When is one file fine?** For a 30-line prototype, splitting is overhead. The
> moment you have two kinds of state (model + sessions) and two kinds of error
> (generation failures + HTTP errors) in one file, the split pays for itself.

### The Request Path

```
Browser                    app.py (Flask)              chatbot_engine.py
   |                            |                             |
   |-- GET / ------------------>|                             |
   |<------ index.html ---------|                             |
   |                            |                             |
   |-- POST /chat {"message"} ->|                             |
   |                            |-- engine.reply(sid, msg) -->|
   |                            |                        [generate]
   |                            |<---- (reply, tokens) -------|
   |<------ {"reply": "..."} ---|                             |
```

The browser never touches the model. It only ever sends and receives JSON.

### Project Layout

```
web_chatbot/
├── WEB_CHATBOT.md        # this workshop
├── chatbot_engine.py     # model loading, history, generation — no Flask
├── app.py                # routes, sessions, validation — no torch
└── templates/
    └── index.html        # the browser frontend
```

Flask finds HTML templates by convention, in a folder called `templates/` next
to the file that created the app.

### Setting Up the Development Environment

**1. Create and activate a virtual environment with Python 3.12**

> This project requires `torch`, which does not yet support Python 3.13+.
> Use Python 3.12 to avoid compatibility issues.

```bash
python3.12 -m venv venv

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

> The first run will download Qwen2.5-0.5B-Instruct (~1 GB). This only happens
> once — it is cached locally in `~/.cache/huggingface/` afterwards.

**3. Verify the installation**

```python
import flask, torch, transformers
print(flask.__version__)          # 3.x
print(torch.__version__)          # should print 2.2.2
print(transformers.__version__)   # should print 4.38.0
```

---

# Part 1 — The Engine

Build the chatbot first, with no web server anywhere. You will be able to talk
to it in the terminal before Flask appears at all.

---

## Task 1 — The Engine Class and Loading

Create `chatbot_engine.py` with a `ChatbotEngine` class. The constructor stores
configuration but does **not** load the model; a separate `load()` method does
that. Add an `is_ready` property so callers can tell whether loading has
finished.

Also define a system prompt that pins the assistant's identity.

<details>
<summary>Solution</summary>

```python
import logging
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly, concise chat assistant running locally on the Qwen2.5 model. "
    "If you are asked your name, say you are a Qwen2.5 assistant. "
    "Keep replies to one or two sentences."
)


class ChatbotEngine:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, system_prompt=DEFAULT_SYSTEM_PROMPT,
                 max_context_tokens=1024, max_new_tokens=128,
                 temperature=0.7, top_p=0.9):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self._model = None
        self._tokenizer = None
        self._histories = {}
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

    @property
    def active_conversations(self):
        return len(self._histories)
```

**Key points:**
- **A class instead of module-level globals.** A one-file version would use
  `MODEL` and `TOKENIZER` globals plus a `global` statement in the loader. That
  works, but `global` is easy to forget — and forgetting it creates *local*
  variables so the module-level names stay `None`, surfacing much later as
  `AttributeError: 'NoneType' object has no attribute 'encode'`. Instance
  attributes make that mistake impossible and let you hold two engines at once.
- **`load()` is separate from `__init__()`** so constructing an engine is
  instant. `app.py` builds the engine at import time and chooses when to pay the
  loading cost. It also means importing the module for a test does not download
  a gigabyte.
- **Load once, at startup.** Loading inside a request handler would re-read the
  model on every message — tens of seconds per reply, and several copies in RAM.
- **`torch_dtype=torch.float32` is a deliberate choice, not a default.** The
  obvious move on a memory-constrained machine is `bfloat16`, which halves the
  weights to ~1 GB. But CPUs without native bf16 support emulate it, and
  measured on a 1.1 GHz Intel i5 float32 ran **~11% faster** (3.34 vs 3.01
  tokens/sec). At 0.5B parameters float32 is only ~2 GB, so the memory saving
  buys nothing. Bigger models flip this trade-off: at 1.5B, bfloat16 becomes
  necessary to fit at all.
- **The system prompt pins the identity for a real reason.** Asked "what is your
  name", the bare model confidently answers *"I am Claude, created by
  Anthropic"* — a hallucination absorbed from assistant transcripts in its
  training data. Smaller models are more prone to this. Stating what it actually
  is corrects it.
- The system prompt is **not** stored in the history. It is re-added on every
  request (Task 2), so trimming old turns can never silently discard it.
- No `pad_token` assignment is needed — Qwen defines a real pad token, unlike
  GPT-2-based models where you must borrow the EOS token.
- `self._model.eval()` disables dropout layers, which are only needed during
  training — inference is faster and deterministic without them.
- The leading underscore on `_model`, `_tokenizer`, `_histories` and `_lock`
  marks them as internal. `is_ready` and `active_conversations` are the public,
  read-only way to ask about that state.
- `self._model` is assigned **last** in `load()`, so `is_ready` can never be
  `True` while the tokenizer is still missing.

</details>

---

## Task 2 — Building the Prompt and Generating

Instruction-tuned models do not take raw concatenated text. They expect a
**list of messages with roles**, rendered into the exact control tokens the
model was trained on. Getting that formatting right is what separates coherent
replies from garbage.

Write two private methods:

- `_build_prompt(self, history)` — prepend the system prompt, render the
  messages with `apply_chat_template`, and drop the oldest turns until the
  result fits `max_context_tokens`. Return `(input_ids, history)`.
- `_generate(self, input_ids)` — generate, decode only the new tokens, and fall
  back to a safe default if the reply is empty.

<details>
<summary>Solution</summary>

```python
    def _build_prompt(self, history):
        while True:
            messages = [{"role": "system", "content": self.system_prompt}] + history

            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if input_ids.shape[-1] <= self.max_context_tokens:
                return input_ids, history
            if len(history) <= 1:
                return input_ids, history

            history = history[2:]   # drop the oldest user+assistant pair

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

        reply = self._tokenizer.decode(
            output_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True,
        ).strip()

        if not reply:
            reply = "I'm not sure how to respond to that."

        return reply
```

**Key points:**
- **History is a list of `{"role", "content"}` dicts**, not a token tensor. The
  roles are what let the model tell your words from its own — which is exactly
  what an older non-instruct model lacks, and why it contradicts itself about
  who it is.
- **`apply_chat_template` writes the control tokens for you.** Qwen expects
  turns wrapped as `<|im_start|>user … <|im_end|>`. Every model family uses a
  different scheme, and the template ships with the tokenizer, so this one call
  stays correct across model swaps. Hand-formatting this string is the single
  most common cause of garbled output.
- **`add_generation_prompt=True` is essential.** It appends the opening of the
  assistant's turn, so the model continues *as the assistant*. Omit it and the
  model often invents the user's next line instead of replying.
- **Trimming drops whole turns, not tokens.** Cutting a token tensor at a fixed
  length — the natural approach without roles — can slice through the middle of
  a message and leave a dangling half-turn the model has to interpret. Removing
  `history[:2]` (one user + assistant pair) always leaves valid structure.
- The system prompt is prepended **inside the loop**, so it is never part of
  what gets trimmed. Store it in the history instead and a long conversation
  would eventually delete the assistant's own instructions.
- `len(history) <= 1` stops the loop when only the current user message remains.
  Without that guard, a single oversized message would loop forever.
- `torch.no_grad()` disables gradient computation during inference, saving
  memory and speeding up generation.
- `output_ids[:, input_ids.shape[-1]:]` slices off the input prefix, leaving
  only the newly generated reply tokens.
- `max_new_tokens` is a **ceiling, not a target**. Generation stops at the
  end-of-turn token, usually well short of it, so a generous cap costs nothing
  on normal replies and only bounds a runaway one.
- Neither method touches `self._histories`. All state mutation lives in
  `reply()` in the next task, which is what makes the locking there easy to
  reason about.

</details>

---

## Task 3 — Conversations and Thread Safety

Add the public `reply(self, conversation_id, message)` and
`reset(self, conversation_id)` methods.

`reply()` looks up that conversation's history, generates, stores the updated
history, and returns `(reply, context_tokens)`. Both methods must be safe to
call from several threads at once, because the Flask development server handles
requests in parallel.

<details>
<summary>Solution</summary>

```python
    def reply(self, conversation_id, message):
        if not self.is_ready:
            raise RuntimeError("Model is not loaded yet — call load() first.")

        with self._lock:
            history = self._histories.get(conversation_id, [])
            history = history + [{"role": "user", "content": message}]

            input_ids, history = self._build_prompt(history)
            reply = self._generate(input_ids)

            history = history + [{"role": "assistant", "content": reply}]
            self._histories[conversation_id] = history
            return reply, input_ids.shape[-1]

    def reset(self, conversation_id):
        with self._lock:
            self._histories.pop(conversation_id, None)
```

**Test it:** once the server exists, open the site in two browser windows and
send messages in both at once. Both should get sensible replies, the second
arriving after the first finishes.

**Key points:**
- **One model cannot generate two replies at once.** Concurrent `generate()`
  calls interleave and corrupt each other's state. The lock makes replies happen
  one at a time — a queue, which is the correct behaviour for a single model on
  a single CPU, not a bottleneck you invented.
- **The lock covers the whole read-modify-write cycle, not just generation.**
  This is a real fix, not extra caution. Locking only `_generate()` leaves this
  race for two messages in the *same* conversation: both read the same history,
  both generate, and the slower one's write overwrites the faster one's —
  silently dropping a turn. Reading, generating and storing must be one
  indivisible unit. Generation dominates the runtime anyway, so widening the
  lock costs effectively nothing.
- `with self._lock:` releases the lock on exit **including when an exception is
  raised**. Manual `acquire()`/`release()` leaks the lock on error and deadlocks
  the whole server.
- `_build_prompt()` and `_generate()` must **not** take the lock — `reply()`
  already holds it, and `threading.Lock` is not reentrant, so acquiring it twice
  in one thread deadlocks instantly. This is why their docstrings record who is
  responsible for locking.
- `history + [...]` builds a **new list** rather than calling `.append()` on the
  stored one. If generation raises, the stored history is left untouched instead
  of holding a user message that never got a reply.
- `_build_prompt()` returns the possibly-trimmed history, and that trimmed
  version is what gets stored — so the trimming is permanent rather than
  recomputed from an ever-growing list on every turn.
- Raising `RuntimeError` when the model is missing gives the caller something
  clear to catch, rather than an `AttributeError` on `None` from deep inside.
- `dict.pop(key, None)` removes the entry if present and does nothing
  otherwise — no `KeyError` when a visitor resets before saying anything.
- **This bug class is invisible when you test alone.** Concurrency problems in
  web apps almost always are, which is why you reason about them rather than
  wait for them to show up.

</details>

---

## Task 4 — Talk to the Engine Without a Web Server

Add an `if __name__ == "__main__":` block to `chatbot_engine.py` that loads the
engine and runs a terminal chat loop, exiting on `quit`, `stop`, or `exit`.

This is the payoff for keeping the engine web-free — and your most valuable
debugging tool for the rest of the workshop.

<details>
<summary>Solution</summary>

```python
if __name__ == "__main__":
    print("=== Chatbot Engine — terminal demo ===\n")

    engine = ChatbotEngine()
    engine.load()

    print("\nChat started. Type 'quit' to exit.\n")

    while True:
        message = input("You: ").strip()
        if not message:
            continue
        if message.lower() in ("quit", "stop", "exit"):
            print("\nBot: Goodbye!\n")
            break

        reply, tokens = engine.reply("terminal", message)
        print(f"\nBot: {reply}   [{tokens} tokens of context]\n")
```

Run it:

```bash
python chatbot_engine.py
```

```
=== Chatbot Engine — terminal demo ===

Loading Qwen/Qwen2.5-0.5B-Instruct (downloads ~1 GB on first run)...
Model ready.

Chat started. Type 'quit' to exit.

You: are you a girl or a guy

Bot: As an artificial intelligence designed by Alibaba Cloud, I do not have
gender. My purpose is to provide assistance and answer questions in natural
language.   [119 tokens of context]

You: What is a president

Bot: A president is the head of state and government of a country, usually
elected by the people through democratic processes.   [185 tokens of context]
```

Expect roughly **10–15 seconds per reply** on a modest CPU.

**Key points:**
- `if __name__ == "__main__":` runs only when the file is executed directly. When
  `app.py` does `from chatbot_engine import ChatbotEngine`, `__name__` is
  `"chatbot_engine"` and this block is skipped — so importing the engine never
  starts a chat loop.
- The single fixed conversation ID `"terminal"` is all a single-user terminal
  needs. The engine cannot tell the difference between this and a session ID —
  which is exactly the point of an opaque ID.
- **Everything the web server will do to the engine, you can now do here.** If a
  reply is nonsense, if context is not carried between turns, if generation
  crashes — reproduce it in this loop first. A bug that appears here is an
  engine bug; a bug that does not is a web bug.
- Watch the token counter climb across turns. That is the conversation history
  accumulating, and it is the same number the web UI will display.

</details>

---

# Part 2 — The Web Layer

The engine works. Now put it behind HTTP. Nothing in `chatbot_engine.py` changes
from here on.

---

## Task 5 — A Server That Runs

Before wiring in the engine, get an empty server running. A 20-line app that
starts is worth more than a 200-line app you cannot launch.

Create `app.py`: a Flask app with a `/health` route returning JSON, running on
`127.0.0.1:5000`.

<details>
<summary>Solution</summary>

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
```

Run it:

```bash
python app.py
```

```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

Check it from a second terminal:

```bash
curl http://127.0.0.1:5000/health
# {"status":"ok"}
```

**Key points:**
- `Flask(__name__)` passes the module name so Flask knows which directory to
  search for `templates/` and `static/`.
- `@app.get("/health")` registers a function to handle `GET /health`. The
  function's return value becomes the HTTP response body.
- `jsonify()` serialises a dict to JSON **and** sets the
  `Content-Type: application/json` header.
- `host="127.0.0.1"` binds to localhost only. Using `"0.0.0.0"` would expose the
  development server to your whole network — never do that with `debug=True`,
  since the debugger allows remote code execution.
- A `/health` route sounds trivial but is the single most useful debugging tool
  in this workshop: it answers "is the server even up?" without involving the
  browser, the model, or your JavaScript.

</details>

---

## Task 6 — Importing the Engine

Import `ChatbotEngine` into `app.py`, build one instance at module level, and
call `load()` from the `__main__` block before `app.run()`.

<details>
<summary>Solution</summary>

```python
import os
from flask import Flask, jsonify
from chatbot_engine import ChatbotEngine

app = Flask(__name__)
engine = ChatbotEngine()

if __name__ == "__main__":
    print(f"=== Web Chatbot Tutorial — Flask + {engine.model_name} ===\n")
    engine.load()
    print("Open http://127.0.0.1:5000 in your browser.\n")
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=True,
        use_reloader=False,
    )
```

**Key points:**
- `from chatbot_engine import ChatbotEngine` works with no packaging setup
  because Python puts the **running script's directory** first on `sys.path`.
  Since `app.py` and `chatbot_engine.py` are siblings, the import resolves. This
  holds whether you run `python app.py` from inside the folder or
  `python web_chatbot/app.py` from its parent — what matters is where `app.py`
  lives, not your shell's working directory.
- `engine = ChatbotEngine()` at module level is cheap because the constructor
  loads nothing (Task 1). One shared instance is what lets every request see the
  same model and the same conversations.
- `engine.load()` sits in `__main__`, **not** at module level, so importing
  `app.py` — for a test, or by a WSGI server — does not download and load a
  gigabyte of weights as a side effect.
- **`use_reloader=False` is the important flag.** In debug mode Flask normally
  watches your files and restarts on every save. That would re-run `load()`, so a
  one-character typo fix costs a full model reload. Worse, the reloader runs your
  module in *two* processes, so the model loads twice and uses double the RAM.
- You keep the good half of debug mode: interactive tracebacks in the browser,
  and template auto-reload. You give up automatic restarts on Python changes —
  press `Ctrl+C` and re-run instead.
- `int(os.environ.get("PORT", 5000))` reads an override from the environment.
  Environment variables are always strings, so the `int()` is required.

<details>
<summary>If you really want auto-reload during development</summary>

Guard the load so only the worker process pays for it. The reloader sets
`WERKZEUG_RUN_MAIN=true` in the child process it spawns:

```python
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        engine.load()         # only the worker process loads the model
    app.run(debug=True)       # reloader stays on
```

You still wait for a reload on every save. Useful when iterating on route
logic; annoying otherwise.

</details>

</details>

---

## Task 7 — One Conversation Per Browser

The engine keys conversations by an opaque ID. The web layer's job is to hand it
a stable ID per visitor, so two people using the site simultaneously never see
each other's messages.

Give each browser an ID, store it in a Flask session cookie, and pass it to the
engine.

<details>
<summary>Solution</summary>

```python
import os
import uuid
from flask import session

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-insecure-key")

def get_conversation_id():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]
```

**Key points:**
- **Why not put the history in the session itself?** Flask's session is a
  cookie, and browsers cap cookies at about 4 KB. A message list happens to be
  JSON-friendly, so it would *technically* fit at first — and then silently stop
  fitting a few turns in, because our context budget alone allows 1024 tokens of
  conversation. Worse, the client could edit its own history and put words in
  the assistant's mouth. So the cookie holds a short **ID**, and the transcript
  stays on the server inside the engine. This ID-in-cookie, data-on-server split
  is how server-side sessions work in general.
- **This function is the entire boundary between web and engine.** It is the
  only place that knows a conversation ID comes from a cookie. Swap cookies for
  an API key or a JWT and this one function changes; `chatbot_engine.py` does
  not.
- `app.secret_key` **signs** the session cookie so a user cannot edit their own
  session ID and read someone else's conversation. Signed is not encrypted: the
  contents are readable by the client, just not forgeable. Never put anything
  secret in a session.
- Hardcoding the dev key is deliberate: a random key at startup would silently
  invalidate every session on every restart, and you would spend an afternoon
  wondering why the bot keeps forgetting. In production, set `FLASK_SECRET_KEY`
  to a real random value.
- `uuid.uuid4().hex` produces a random 32-character ID with no meaningful
  ordering — not guessable, unlike an incrementing counter.

</details>

---

## Task 8 — The `/chat` Endpoint

Write a `POST /chat` route that reads `{"message": "..."}` from the request
body, validates it, delegates to the engine, and returns
`{"reply": "...", "history_tokens": N}`.

It must return a useful JSON error — never an HTML crash page — for: an empty
message, an over-long message, a request that arrives before the model has
loaded, and a failure inside generation.

<details>
<summary>Solution</summary>

```python
from flask import request

MAX_MESSAGE_CHARS = 1000

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify(error="Message must not be empty."), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify(error=f"Message must be at most {MAX_MESSAGE_CHARS} characters."), 400

    if not engine.is_ready:
        return jsonify(error="Model is still loading, try again shortly."), 503

    try:
        reply, history_tokens = engine.reply(get_conversation_id(), message)
    except Exception as exc:
        app.logger.exception("Generation failed")
        return jsonify(error=f"Generation failed: {exc}"), 500

    return jsonify(reply=reply, history_tokens=history_tokens)
```

Test it without a browser:

```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello there"}'
# {"history_tokens":38,"reply":"Hello! How can I assist you today?"}
```

**Key points:**
- **Notice how little this handler does.** Validate, delegate, translate the
  result to HTTP. There is no `torch` here and no tokenisation — one line does
  the actual work. That is what the split is for.
- **Validation belongs here, not in the engine.** "Is this a well-formed HTTP
  request?" is a web question. The engine's contract is that it receives a
  non-empty string; enforcing that is the caller's job.
- **`POST`, not `GET`.** The request changes server state, and the message
  belongs in a body rather than a URL that gets logged and stored in browser
  history.
- `request.get_json(silent=True)` returns `None` instead of raising when the
  body is missing or malformed. Without `silent=True`, a bad request produces a
  415 or 400 HTML page, and your frontend's `response.json()` fails with a
  confusing parse error instead of showing your message.
- `(data.get("message") or "").strip()` handles a missing key, an explicit
  `null`, and whitespace-only input in one expression.
- **Status codes carry meaning.** `400` says the client sent something invalid,
  `503` says come back shortly, `500` says the server broke and it is not the
  client's fault. A `200` with an error string inside cannot be distinguished
  from success.
- `app.logger.exception(...)` writes the full traceback to the terminal while
  the browser gets a short message. You need the detail; the user does not.
- Catching bare `Exception` around the engine call is deliberate: model failures
  are varied and hard to enumerate, and one bad request should not take out the
  endpoint for everyone else.
- Returning `history_tokens` is not decoration — it makes the context trimming
  from Task 2 visible in the UI, so you can watch the context fill up.

</details>

---

## Task 9 — Reset and Health Routes

Add `POST /reset`, which forgets the calling browser's history, and extend
`/health` to report engine state.

<details>
<summary>Solution</summary>

```python
@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model_loaded=engine.is_ready,
        active_sessions=engine.active_conversations,
    )

@app.post("/reset")
def reset():
    engine.reset(get_conversation_id())
    return jsonify(status="reset")
```

**Key points:**
- Reset drops the **history**, not the session ID. The browser keeps its
  identity and simply starts a new conversation.
- `/health` reads the engine's two public properties rather than reaching into
  `engine._histories`. The underscore-prefixed attributes are private for a
  reason: `active_conversations` is a contract, the dict is an implementation
  detail you might later replace with Redis.
- `model_loaded` and `active_sessions` turn `/health` into a live window into
  server state. When the bot "forgets" everything between messages, checking
  whether `active_sessions` climbs by one per message tells you instantly that
  cookies are not coming back.

</details>

---

## Task 10 — The Browser Frontend

Write `templates/index.html`: a scrolling transcript, a text input, a Send
button, and a Reset button. On submit it should `POST` the message to `/chat`
and append the reply to the transcript. No frameworks, no CDN — plain HTML, CSS,
and JavaScript in one file.

Requirements:
1. Show the user's message immediately, then a placeholder while waiting.
2. Disable Send while a request is in flight.
3. Display server errors in the transcript rather than failing silently.
4. Insert text safely, so a reply containing HTML cannot execute.

<details>
<summary>Solution — the JavaScript</summary>

The full file with styling is in `templates/index.html`. The logic is:

```javascript
// Append one bubble. Text goes in with .textContent, never .innerHTML — the
// model's output is untrusted text, and .innerHTML would execute HTML in it.
function addMessage(text, kind) {
  const el = document.createElement("div");
  el.className = "msg " + kind;
  el.textContent = text;
  transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
  return el;
}

// A non-2xx response still carries a JSON body with an "error" key,
// so parse first and only then decide whether to throw.
async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage(message, "user");
  input.value = "";
  sendBtn.disabled = true;
  status.textContent = "thinking…";
  const pending = addMessage("…", "bot pending");

  try {
    const data = await postJSON("/chat", { message });
    pending.textContent = data.reply;
    pending.className = "msg bot";
    status.textContent = `${data.history_tokens} tokens of context`;
  } catch (err) {
    pending.remove();
    addMessage(err.message, "error");
    status.textContent = "error";
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});
```

Add the route that serves it:

```python
from flask import render_template

@app.get("/")
def index():
    return render_template("index.html", model_name=engine.model_name)
```

In the template, render it with a Jinja placeholder:

```html
<h1>Local Web Chatbot</h1>
<span id="model">{{ model_name }}</span>
```

**Key points:**
- `render_template("index.html")` looks in `templates/` relative to the file
  that created the app. `jinja2.exceptions.TemplateNotFound` means the folder is
  missing, misspelled, or not next to `app.py` — Flask does not search your
  current working directory.
- **Keyword arguments to `render_template` become variables in the template**,
  read with `{{ model_name }}`. Passing the name in rather than hardcoding it is
  worth the extra argument: hardcoded model names in the UI are exactly the kind
  of thing that silently goes stale after a model swap, since nothing breaks and
  no test fails — the page just quietly lies. Deriving it from the engine means
  the page cannot disagree with what is actually loaded.
- With `debug=True`, Flask re-reads the template on **every request**. Edit
  `index.html`, hit refresh, no restart — which matters because restarting means
  reloading the model.
- `event.preventDefault()` stops the browser's default form submission, which
  would reload the whole page and throw away the transcript. Forgetting this
  produces the classic symptom: the page flashes and the input clears, but
  nothing else happens.
- Using a `<form>` with a submit handler — rather than a click handler on the
  button — gets Enter-to-send for free, along with correct keyboard and screen
  reader behaviour.
- **`.textContent`, not `.innerHTML`.** The reply is text generated by a model
  from user input. Assigning it with `.innerHTML` would execute any HTML or
  script inside it. `.textContent` renders it as literal characters — this is
  the whole fix for cross-site scripting here.
- `await res.json()` is called **before** checking `res.ok`, because the error
  body is where the server's message lives. Checking `res.ok` first and throwing
  a generic error would discard the useful text you returned in Task 8.
- `.catch(() => ({}))` on the JSON parse handles the case where the server
  returns HTML instead of JSON — exactly what happens on an unhandled crash.
  Without it you get `SyntaxError: Unexpected token '<'`, which tells you
  nothing about the real problem.
- The `finally` block re-enables the button whether the request succeeded or
  failed. Putting that line only in the `try` leaves the UI permanently frozen
  after the first error.
- The pending bubble is **reused** as the reply bubble on success and removed on
  failure, so the transcript never keeps a stray "…".

</details>

---

## Running the Server

**Start it** from inside the `web_chatbot/` folder:

```bash
python app.py
```

```
=== Web Chatbot Tutorial — Flask + Qwen/Qwen2.5-0.5B-Instruct ===

Loading Qwen/Qwen2.5-0.5B-Instruct (downloads ~1 GB on first run)...
Model ready.
Open http://127.0.0.1:5000 in your browser.

 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

The model load takes roughly 5–20 seconds on a warm cache, and a few minutes on
the very first run while it downloads.

**Stop it:** `Ctrl+C`.

**The edit-run loop:**

| You changed | What to do |
|---|---|
| `templates/index.html` | Just refresh the browser — templates reload on every request in debug mode |
| CSS or JavaScript inside the template | Hard-refresh (`Cmd/Ctrl+Shift+R`) to defeat the browser cache |
| `app.py` | `Ctrl+C` and re-run — this reloads the model |
| `chatbot_engine.py` | Test it with `python chatbot_engine.py` first, then restart the server |

Because Python edits are the expensive ones, it pays to test engine changes in
the terminal demo and route logic with `curl` before touching the UI. That is
why the tasks are ordered the way they are.

**Change the port** if 5000 is taken. On macOS, AirPlay Receiver listens there by
default:

```bash
PORT=5001 python app.py
```

---

## Debugging While You Develop

The split gives you a first question to ask before anything else: **can you
reproduce it in the terminal demo?**

```bash
python chatbot_engine.py
```

If the bug appears there, it is in `chatbot_engine.py`, and no amount of staring
at routes will help. If it does not, it is in `app.py` or the frontend. That one
check eliminates half the codebase in about ten seconds.

Beyond that you have four instruments. Learning which one answers which question
is most of the skill.

### 1. The terminal — every request is logged

Flask prints one line per request:

```
127.0.0.1 - - [28/Jul/2026 14:03:11] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [28/Jul/2026 14:03:19] "POST /chat HTTP/1.1" 200 -
127.0.0.1 - - [28/Jul/2026 14:03:24] "POST /chat HTTP/1.1" 400 -
```

Read the status code first. **If no line appears at all, the request never
reached the server** — the bug is in your JavaScript (wrong URL, or the form
reloaded the page), not in Python.

Add your own logging inside a handler:

```python
app.logger.info("chat: session=%s chars=%d", get_conversation_id()[:8], len(message))
```

Use `app.logger`, not `print()`. It is timestamped, level-filtered, and goes to
the same stream as Flask's own output.

### 2. `debug=True` — the interactive traceback

When a handler raises an uncaught exception, debug mode returns the full
traceback as a web page instead of a blank 500. Every frame expands, and the
console icon on the right of a frame opens a Python prompt **in that frame** —
you can inspect `input_ids.shape` at the moment things broke. The terminal
prints a PIN the first time you use it.

This only works for exceptions you do **not** catch. That is a real tension with
Task 8's `try/except`: catching everything gives users clean errors but hides
the interactive debugger from you. When you are stuck on a generation bug,
temporarily comment out the `except` block and let it crash — or better, drop
into `python chatbot_engine.py`, where nothing is catching anything.

> Never run `debug=True` on a public server. The console is remote code
> execution by design.

### 3. `curl` — test the API without the browser

This is the fastest way to split "is the bug in Python or in JavaScript?"

```bash
# Is the server up and the model loaded?
curl http://127.0.0.1:5000/health

# A normal message
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'

# Does validation reject an empty message with a 400?
curl -i -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "   "}'

# Do sessions work? -c saves cookies, -b sends them back.
curl -c jar.txt -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"message": "my name is Sam"}'
curl -b jar.txt -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"message": "what is my name?"}'
```

`-i` prints the response headers, including the status code. The cookie-jar pair
is the one reliable way to test conversation memory from the command line —
plain `curl` sends no cookies, so every call looks like a brand new visitor.
Watch `history_tokens` grow across the two calls: that is proof the session is
working.

### 4. Browser DevTools (F12) — the frontend half

- **Console** — JavaScript errors land here. A silent Send button is almost
  always an error on this tab.
- **Network** — click the `/chat` row after sending a message. **Payload** shows
  what your JavaScript actually sent; **Response** shows exactly what Flask
  returned. Comparing those two against what you expected localises nearly every
  bug to one side or the other.
- **Application → Cookies** — you should see one `session` cookie for
  `127.0.0.1`. No cookie means `app.secret_key` is missing or sessions are not
  being written.

### Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `Address already in use` | Port 5000 taken (macOS AirPlay Receiver, or an old server still running) | `PORT=5001 python app.py`, or find it with `lsof -nP -iTCP:5000 -sTCP:LISTEN` |
| `ModuleNotFoundError: No module named 'chatbot_engine'` | `app.py` moved away from its sibling module | Keep both files in the same folder; run `python app.py` or `python web_chatbot/app.py` |
| `jinja2.exceptions.TemplateNotFound: index.html` | `templates/` missing, misnamed, or not next to `app.py` | Create `templates/` in the same directory as `app.py` |
| `RuntimeError: Model is not loaded yet` | `engine.load()` never ran | Call it before `app.run()` |
| `RuntimeError: The session is unavailable because no secret key was set` | `app.secret_key` not set | Set it before the first request |
| `SyntaxError: Unexpected token '<'` in the browser console | The server returned an HTML error page, not JSON | Look at the terminal traceback — the real error is there |
| `405 Method Not Allowed` | Route declared `@app.get` but the frontend sends `POST` | Match the decorator to the method |
| `415 Unsupported Media Type` | `Content-Type: application/json` header missing on the request | Add the header in `fetch`/`curl` |
| Bot forgets everything each message | Cookies not returned (plain `curl`, or a private-window quirk) | Use `curl -c/-b`; check Application → Cookies |
| Bot remembers things you never said | You are reusing an old session ID | Click Reset, or `POST /reset` |
| Replies take 10–15s | Normal — CPU generation of a 0.5B model | Lower `max_new_tokens`, or add streaming (exercise 2) so first words appear in ~2s |
| Replies get slower over a long chat | Normal — the prompt grows every turn, so there is more to process | Watch the token counter climb; lower `max_context_tokens` |
| Bot claims to be "Claude, created by Anthropic" | Identity hallucinated from training data | Pin the name in `system_prompt` (Task 1) |
| Replies are garbled, or the bot writes your next line for you | Prompt not rendered with the model's template | Use `apply_chat_template` with `add_generation_prompt=True` — never hand-format the turns |
| The page reloads when you press Send | `event.preventDefault()` missing | Add it as the first line of the submit handler |
| Styling changes have no effect | Browser cached the page | Hard-refresh (`Cmd/Ctrl+Shift+R`) |

---

## Exercises

1. **Write tests for the engine.** Because it has no Flask dependency, you can
   test it with plain `pytest` — assert that two conversation IDs stay
   independent, and that `reset()` clears one without touching the other. Use a
   tiny model to keep tests fast.
2. **Streaming replies.** Use `TextIteratorStreamer` from `transformers` and a
   Flask streaming response to show tokens as they are generated. Note which
   file each change belongs in.
3. **Show the token budget.** The frontend already receives `history_tokens`.
   Turn it into a progress bar against `max_context_tokens`.
4. **Add temperature control.** Send a slider value with each message and pass
   it through to the engine. Validate the range in `app.py` — never trust a
   number from a browser — and decide whether it belongs on the constructor or
   as a `reply()` argument.
5. **Persist conversations.** Replace the `_histories` dict with SQLite so
   history survives a restart. Nothing in `app.py` should need to change — if it
   does, the boundary was in the wrong place.

---

## Beyond This Workshop

This is a development setup, not a deployment. Three things would have to change
before it faced real users:

- **The dev server.** `app.run()` is single-process and explicitly not built for
  production. You would run `gunicorn` or `waitress` instead — which means
  moving `engine.load()` into an app factory, and reckoning with the fact that
  each worker process loads its own copy of the model.
- **`_histories` grows forever.** Every new visitor adds an entry that is never
  removed, and the dict is per-process, so it is not shared between workers.
  Real deployments use Redis or a database with an expiry time. Because this is
  hidden behind `reply()` and `reset()`, swapping it out touches one file.
- **`debug=True` and the dev secret key must both go.** The debugger is remote
  code execution, and a published secret key lets anyone forge a session.
