# Python AI Development Workshops

A collection of hands-on Python tutorials covering the fundamentals of the language, object-oriented programming, speech I/O, and AI-powered applications using Claude.

Each tutorial comes as a pair: a runnable `.py` file you can execute and experiment with, and a `.md` workshop file with step-by-step tasks and collapsible solutions.

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

### 1. Python Basics
**Files:** [python_basics.py](python_basics.py) · [PYTHON_BASICS.md](PYTHON_BASICS.md)

Introduces the core building blocks of Python for complete beginners. Covers variables and data types, string formatting with f-strings, user input, conditionals, lists, loops, dictionaries, and functions — ending with a small interactive program that uses all of them together.

---

### 2. Object-Oriented Programming
**Files:** [oop_basics.py](oop_basics.py) · [OOP_BASICS.md](OOP_BASICS.md)

Teaches the four pillars of OOP in Python: classes, objects, inheritance, and encapsulation. Walks through creating a class with `__init__` and instance methods, readable `__str__` output, inheriting from a parent class, and using `super()` to extend it — all illustrated with a practical library system example.

---

### 3. Speech to Text
**Files:** [speech_to_text.py](speech_to_text.py) · [SPEECH_TO_TEXT.md](SPEECH_TO_TEXT.md)

Captures live microphone audio and transcribes it using Google's free Web Speech API via the `SpeechRecognition` library. Demonstrates single-shot transcription, ambient noise calibration, continuous listening with a timeout, and proper error handling for unintelligible audio and network failures.

**Extra setup (macOS):** `brew install portaudio`

---

### 4. Text to Speech
**Files:** [text_to_speech.py](text_to_speech.py) · [TEXT_TO_SPEECH.md](TEXT_TO_SPEECH.md)

Converts written text into spoken audio using `pyttsx3`, which runs fully offline with no API key. Covers listing available voices, speaking a single phrase, and running a continuous type-and-speak loop — with control over speaking rate and volume.

---

### 5. Chatbot
**Files:** [chatbot.py](chatbot.py) · [CHATBOT.md](CHATBOT.md)

Combines speech-to-text and text-to-speech with Microsoft's DialoGPT language model to build a fully conversational chatbot. Supports both voice and keyboard input. Responses are spoken aloud and printed to the terminal. Conversation history is managed as token ID tensors to stay within the model's context window.

**Extra setup:** Python 3.12 required · first run downloads the DialoGPT-medium model (~863 MB)

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
| `torch==2.2.2` | Chatbot |
| `transformers==4.38.0` | Chatbot |
| `numpy==1.26.4` | Chatbot (pinned below 2.0 for torch compatibility) |
| `anthropic` | AI Code Generation |
