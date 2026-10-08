# Workshop: AI-Based Code Generation with Claude

---

## Introduction

### Background

Large language models like Claude can understand natural language and generate
working code from plain-English descriptions. This unlocks a new kind of
development workflow: instead of writing every line by hand, you describe what
you want and let the model draft it — then review, test, and refine.

This workshop uses the **Anthropic Python SDK** to call Claude directly. You
will build utilities that generate functions, classes, and unit tests from
descriptions, refactor messy code, explain complex snippets, and hold a
multi-turn coding conversation.

**How the pieces fit together:**

```
Plain-English description
         ↓
   Anthropic Python SDK  (client.messages.create / stream)
         ↓
   Claude claude-opus-4-8
         ↓
   Generated Python code
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
python3.12 -m venv venv   # Windows: py -3.12 -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

The only new package needed for this workshop is `anthropic`:

```bash
pip install anthropic
```

**3. Set your API key**

```bash
export ANTHROPIC_API_KEY=your-key-here           # macOS / Linux
$env:ANTHROPIC_API_KEY="your-key-here"           # Windows PowerShell
set ANTHROPIC_API_KEY=your-key-here              # Windows Command Prompt
```

The variable only lasts for the current terminal window. To keep it, add the
`export` line to `~/.zshrc` on macOS, or run
`setx ANTHROPIC_API_KEY "your-key-here"` once on Windows and open a new terminal.

Get a key at [console.anthropic.com](https://console.anthropic.com).

**4. Verify the installation**

```python
import anthropic
client = anthropic.Anthropic()
print("Client ready")
```

---

## Task 1 — Basic Code Generation

Write a function `generate_code(description)` that sends a plain-English
description to Claude and returns the generated Python code as a string.

Requirements:
- Use model `claude-opus-4-8`
- Use a system prompt that tells Claude to return **only raw code** with no
  markdown fences or surrounding explanation
- Apply `cache_control: {type: "ephemeral"}` to the system prompt so it is
  only billed once across repeated calls

<details>
<summary>Solution</summary>

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are an expert Python developer.
When asked to generate code, return ONLY the raw Python code with no markdown fences,
no explanation before or after — just the code itself.
Write clean, readable code with descriptive variable names."""

def generate_code(description):
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": f"Write a Python function that: {description}",
        }],
    )
    return response.content[0].text

code = generate_code("takes a list of numbers and returns the average, handling empty lists")
print(code)
```

**Key points:**
- `client.messages.create()` sends a single request and waits for the full
  response before returning.
- `response.content` is a list of blocks. For a plain text response the first
  block is a `TextBlock`; access its text with `.text`.
- Passing the system prompt as a list with `cache_control` tells the API to
  cache those tokens — subsequent requests with the same system prefix skip
  the processing cost (~90% cheaper for the cached portion).
- `max_tokens=1024` caps the output length. Raise it if you expect longer
  functions.

</details>

---

## Task 2 — Streaming Code Generation

Write a function `generate_code_streaming(description)` that streams Claude's
response token-by-token, printing each chunk as it arrives.

After streaming completes, print the number of output tokens used.

<details>
<summary>Solution</summary>

```python
def generate_code_streaming(description):
    print(f"Generating: {description}\n")
    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=2048,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": f"Write a Python class that: {description}",
        }],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    final = stream.get_final_message()
    print(f"\n\n[Tokens used: {final.usage.output_tokens}]")
```

**Key points:**
- `client.messages.stream()` is a context manager that opens an SSE
  connection. Inside the `with` block, `stream.text_stream` is an iterator
  that yields text chunks as they arrive.
- `flush=True` on `print()` forces each chunk to appear immediately instead
  of being buffered until a newline.
- `stream.get_final_message()` returns the fully assembled `Message` object
  after streaming is complete — use it to read `usage`, `stop_reason`, etc.
- Always use streaming when `max_tokens` is large (above ~16 000) to avoid
  SDK HTTP timeouts.

</details>

---

## Task 3 — Generating Code with Unit Tests

Write a function `generate_with_tests(description)` that asks Claude to
generate both a Python function **and** its pytest unit tests in a single
request.

Enable **adaptive thinking** so Claude reasons through edge cases before
writing. Print only the final text block (skip thinking blocks).

<details>
<summary>Solution</summary>

```python
def generate_with_tests(description):
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": (
                f"Write a Python function that: {description}\n\n"
                "Then write pytest unit tests that cover normal cases, "
                "edge cases, and error cases."
            ),
        }],
    )

    for block in response.content:
        if block.type == "text":
            print(block.text)
```

**Key points:**
- `thinking: {type: "adaptive"}` lets Claude decide how much to reason
  before writing. This is the recommended mode for Opus 4.8 — it produces
  more thorough and correct code, especially for tricky edge cases.
- `response.content` may contain both `thinking` blocks and `text` blocks.
  Filtering by `block.type == "text"` ensures only the final code is printed.
- Thinking blocks are billed as output tokens but contain Claude's internal
  reasoning — useful to log for debugging but not meant to show to end users.

</details>

---

## Task 4 — Refactoring Existing Code

Write a function `refactor_code(code)` that takes a messy Python snippet and
returns a cleaner, more Pythonic version.

The prompt should ask Claude to:
- Use descriptive variable names
- Add type hints
- Raise exceptions instead of returning `None`

<details>
<summary>Solution</summary>

```python
def refactor_code(code):
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": (
                "Refactor this Python code to be cleaner and more Pythonic. "
                "Use good variable names, add type hints, and raise exceptions "
                "instead of returning None:\n\n"
                f"{code}"
            ),
        }],
    )
    return response.content[0].text
```

**Key points:**
- The user message embeds the code directly as a string — no special
  formatting is required; Claude understands inline code.
- Being explicit about what "refactor" means (`type hints`, `raise exceptions`)
  leads to more predictable results than asking generically to "improve" code.
- The system prompt's `cache_control` is shared across all functions in this
  workshop — the API recognises the same prefix and serves it from cache,
  so each subsequent call only pays for the user-message tokens.

</details>

---

## Task 5 — Code Explanation

Write a function `explain_code(code)` that takes a Python snippet and returns
a plain-English explanation suitable for beginners.

Use a different system prompt for this function — one that instructs Claude to
act as a tutor and avoid putting code in the answer.

<details>
<summary>Solution</summary>

```python
def explain_code(code):
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        system="You are a Python tutor. Explain code clearly for beginners. No code in your answer.",
        messages=[{
            "role": "user",
            "content": f"Explain what this Python code does, step by step:\n\n{code}",
        }],
    )
    return response.content[0].text
```

**Key points:**
- The system prompt is a plain string here (not a list) — this is fine for
  short prompts that do not need caching.
- Keeping the system prompt short and role-specific ("Python tutor") focuses
  Claude's voice and avoids bleeding formatting rules from other functions.
- `max_tokens=512` is enough for a concise explanation; raising it
  unnecessarily costs more and can lead to verbose answers.

</details>

---

## Task 6 — Multi-Turn Code Assistant

Write a function `code_assistant()` that runs a continuous conversation loop
where the user can ask Claude to write, extend, and fix code across multiple
turns.

Requirements:
1. Maintain a `history` list and append each user/assistant turn to it.
2. Stream Claude's responses token-by-token.
3. Apply `cache_control` to the system prompt so it stays cached across every
   turn.
4. Exit when the user types `"quit"`, `"exit"`, or `"stop"`.

<details>
<summary>Solution</summary>

```python
def code_assistant():
    history = []

    print("Code Assistant ready. Describe what you want to build.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("quit", "exit", "stop"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        with client.messages.stream(
            model="claude-opus-4-8",
            max_tokens=2048,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=history,
        ) as stream:
            print("\nClaude: ", end="")
            response_text = ""
            for text in stream.text_stream:
                print(text, end="", flush=True)
                response_text += text

        print("\n")
        history.append({"role": "assistant", "content": response_text})
```

**Key points:**
- The API is stateless — `history` is the entire conversation. Appending
  both user and assistant turns means Claude can refer back to any earlier
  message ("add error handling to the function you wrote above").
- The system prompt's `cache_control` marker is at the same prefix position
  on every turn, so the API reads from the same cache entry across the whole
  session — very cost-efficient for long conversations.
- `response_text += text` accumulates the streamed chunks so the full
  response can be stored in history as a single string.

</details>

---

## Task 7 — Putting It All Together

Write a `__main__` block that:

1. Generates a utility function from a description.
2. Generates a class with streaming.
3. Generates a function with unit tests using adaptive thinking.
4. Refactors a messy code snippet.
5. Explains a complex one-liner.
6. Launches the interactive multi-turn assistant.

<details>
<summary>Solution</summary>

```python
if __name__ == "__main__":
    print("=== AI Code Generation Tutorial ===\n")

    # 1. Basic generation
    code = generate_code("takes a list of numbers and returns the average")
    print("Generated function:\n", code)

    # 2. Streaming generation
    generate_code_streaming("represents a bank account with deposit and withdraw methods")

    # 3. Generate with tests
    generate_with_tests("validates an email address and returns True/False")

    # 4. Refactor
    messy = "def f(x,y,op):\n  if op=='add': return x+y\n  elif op=='sub': return x-y"
    print(refactor_code(messy))

    # 5. Explain
    snippet = "result = {x: x**2 for x in range(10) if x % 2 == 0}"
    print(explain_code(snippet))

    # 6. Interactive assistant
    print("--- Launching interactive assistant ---")
    code_assistant()
```

**Key points:**
- The `if __name__ == "__main__"` guard ensures the demonstration only runs
  when the script is executed directly, not when imported as a module.
- The system prompt is cached across every function call — Claude processes
  it once and every subsequent request reads from cache, making the full
  demonstration significantly cheaper than six independent uncached calls.
- The interactive assistant at the end lets you freely experiment with
  everything covered in the previous tasks.

</details>
