# Python AI Development Workshops

A collection of hands-on Python tutorials covering speech I/O, local language models, and AI-powered applications using Claude.

Each tutorial comes as a pair: a runnable `.py` file you can execute and experiment with, and a `.md` workshop file with step-by-step tasks and collapsible solutions. The web chatbot and the community chat each span several files, so they live in their own [`web_chatbot/`](web_chatbot) and [`group_chat/`](group_chat) folders.

**New to Python?** Start with the beginner workshops in [cluebbe/python](https://github.com/cluebbe/python) — Python basics and object-oriented programming, no dependencies required. They used to live here, and the tutorials below assume you are comfortable with that material.

---

## Getting Started

**Requirements:** Python 3.12 (required for the AI tutorials — `torch` does not support 3.13+)

```bash
# Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt
```

For the code generation tutorial you also need an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

---

## Tutorials

### 1. Speech to Text
**Files:** [speech_to_text.py](speech_to_text.py) · [SPEECH_TO_TEXT.md](SPEECH_TO_TEXT.md)

Captures live microphone audio and transcribes it using Google's free Web Speech API via the `SpeechRecognition` library. Demonstrates single-shot transcription, ambient noise calibration, continuous listening with a timeout, and proper error handling for unintelligible audio and network failures.

**Required setup (macOS):** `brew install portaudio flac` *before* `pip install -r requirements.txt` — PyAudio is compiled against it. On Apple Silicon, `which brew` must show `/opt/homebrew/bin/brew`. See [SPEECH_TO_TEXT.md](SPEECH_TO_TEXT.md) for troubleshooting.

---

### 2. Text to Speech
**Files:** [text_to_speech.py](text_to_speech.py) · [TEXT_TO_SPEECH.md](TEXT_TO_SPEECH.md)

Converts written text into spoken audio using `pyttsx3`, which runs fully offline with no API key. Covers listing available voices, speaking a single phrase, and running a continuous type-and-speak loop — with control over speaking rate and volume.

---

### 3. Chatbot
**Files:** [chatbot.py](chatbot.py) · [CHATBOT.md](CHATBOT.md)

Combines speech-to-text and text-to-speech with the local **Qwen2.5-0.5B-Instruct** language model to build a fully conversational chatbot. Supports both voice and keyboard input. Responses are spoken aloud and printed to the terminal. Conversation history is kept as a list of role-tagged messages and rendered with `apply_chat_template`, trimmed by whole turns to fit the context budget, with a system prompt that pins the assistant's identity and keeps replies short enough to listen to.

**Extra setup:** Python 3.12 required · first run downloads Qwen2.5-0.5B-Instruct (~1 GB, ungated — no HF login needed) · expect 10–15s per reply on CPU

---

### 4. Web Chatbot
**Folder:** [web_chatbot/](web_chatbot) — [WEB_CHATBOT.md](web_chatbot/WEB_CHATBOT.md) · [chatbot_engine.py](web_chatbot/chatbot_engine.py) · [app.py](web_chatbot/app.py) · [templates/index.html](web_chatbot/templates/index.html)

Puts a local **Qwen2.5-0.5B-Instruct** chatbot behind a Flask web server with a browser frontend. The code is split along a one-way dependency: `chatbot_engine.py` holds the model, conversation history and generation and never imports Flask, while `app.py` holds routes, session cookies and validation and never imports torch. The engine runs standalone as a terminal chat (`python chatbot_engine.py`), which doubles as the fastest way to tell a model bug from a web bug.

Covers instruction-tuned prompting with `apply_chat_template`, a system prompt that pins the assistant's identity, trimming history by whole turns to fit the context budget, loading the model once at startup, per-visitor history keyed by a signed session cookie, a threading lock around the read-generate-store cycle, and proper HTTP status codes for every error path. Includes a full section on running and debugging while developing — Flask's request log, interactive tracebacks, `curl` with cookie jars, browser DevTools, and a table of common failures.

```bash
cd web_chatbot
python app.py            # web server at http://127.0.0.1:5000
python chatbot_engine.py # same engine, terminal only
```

**Extra setup:** Python 3.12 required · first run downloads Qwen2.5-0.5B-Instruct (~1 GB, ungated — no HF login needed) · expect 10–15s per reply on CPU · on macOS, AirPlay Receiver may occupy port 5000 — use `PORT=5001 python app.py`

---

### 5. Community Chat
**Folder:** [group_chat/](group_chat) — [GROUP_CHAT.md](group_chat/GROUP_CHAT.md) · [bot_engine.py](group_chat/bot_engine.py) · [chat_room.py](group_chat/chat_room.py) · [app.py](group_chat/app.py) · [templates/index.html](group_chat/templates/index.html)

A WhatsApp-style group chat where several people and three AI bots share one
transcript. Members join with a name, address a bot with `@Ada`, and see each
other's messages appear live. Builds directly on the Web Chatbot and adds the
problems that only appear once a chatbot has company.

The two-file split grows a middle layer: `bot_engine.py` (model and personas),
`chat_room.py` (transcript, membership, turn-taking, background worker) and
`app.py` (routes and cookies), with the dependency arrow pointing one way
throughout. All three bots share **one** loaded copy of Qwen2.5-0.5B-Instruct —
a persona is a system prompt, not a model.

Covers flattening a five-speaker transcript onto the two roles a chat template
understands, repairing the labels a small model puts on its own messages, a
shared message log with monotonic ids, deciding which bot answers an
unaddressed message, three rules that stop bots talking to each other forever, a
worker thread so sending a message never waits for a 10–15s reply, and
`GET /messages?since=N` polling — the route a one-to-one chatbot never needs.
Turn-taking is tested against a fake engine, so the tricky parts run in
milliseconds with no model loaded.

```bash
cd group_chat
python app.py            # web server at http://127.0.0.1:5000
python chat_room.py      # same room, terminal only
python bot_engine.py     # personas answering a fixed transcript
```

**Extra setup:** Python 3.12 required · shares the Qwen2.5-0.5B-Instruct download with the Web Chatbot · expect 10–15s per bot reply on CPU · open the page twice (one private window) to chat as two members

---

### 6. AI Code Generation
**Files:** [code_generation.py](code_generation.py) · [CODE_GENERATION.md](CODE_GENERATION.md)

Uses the Anthropic Python SDK to call Claude (claude-opus-4-8) for AI-powered code generation. Demonstrates generating functions and classes from descriptions, streaming long outputs, producing code alongside unit tests, refactoring messy code, explaining complex snippets, and building a multi-turn interactive coding assistant. Includes prompt caching to reduce API costs.

**Extra setup:** `export ANTHROPIC_API_KEY=your-key-here`

---

## Dependencies

All Python packages are pinned in [requirements.txt](requirements.txt). Key packages:

| Package | Used by |
|---|---|
| `SpeechRecognition` | Speech to Text, Chatbot |
| `PyAudio` | Speech to Text, Chatbot |
| `pyttsx3` | Text to Speech, Chatbot |
| `torch==2.2.2` | Chatbot, Web Chatbot, Community Chat |
| `transformers==4.38.0` | Chatbot, Web Chatbot, Community Chat |
| `numpy==1.26.4` | Chatbot, Web Chatbot, Community Chat (pinned below 2.0 for torch compatibility) |
| `Flask==3.1.3` | Web Chatbot, Community Chat |
| `anthropic` | AI Code Generation |
