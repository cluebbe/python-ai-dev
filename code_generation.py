# AI-Based Code Generation with Claude
#
# This tutorial shows how to use Claude's API to generate, refactor,
# and explain Python code using natural language descriptions.
#
# SETUP
# -----
# Install dependencies:
#   pip install anthropic
#
# Set your API key:
#   export ANTHROPIC_API_KEY=your-key-here
#
# Run with:
#   python code_generation.py

import anthropic  # Anthropic SDK for calling Claude

# Create a single client instance to reuse across all requests
client = anthropic.Anthropic()

# The model to use for all requests — Opus 4.8 is the most capable
MODEL = "claude-opus-4-8"

# System prompt shared across all requests — cached so it is only billed once
SYSTEM_PROMPT = """You are an expert Python developer.
When asked to generate code, return ONLY the raw Python code with no markdown fences,
no explanation before or after — just the code itself.
Write clean, readable code with descriptive variable names."""


# ---------------------------------------------------------------------------
# 1. BASIC CODE GENERATION
# ---------------------------------------------------------------------------
# The simplest usage: send a description, get back Python code.

print("=" * 50)
print("  AI Code Generation with Claude")
print("=" * 50)

print("\n--- 1. Basic Code Generation ---")

def generate_code(description):
    """Ask Claude to generate Python code from a plain-English description."""
    response = client.messages.create(        # Call the Claude API
        model=MODEL,                          # Which Claude model to use
        max_tokens=1024,                      # Maximum length of the response
        system=[{                             # System prompt as a list for cache_control support
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},  # Cache the system prompt to save cost on repeated calls
        }],
        messages=[{                           # The user's request
            "role": "user",
            "content": f"Write a Python function that: {description}",
        }],
    )
    return response.content[0].text          # Extract the generated code string from the response

# Generate a simple utility function
code = generate_code("takes a list of numbers and returns the average, handling empty lists gracefully")
print(code)                                  # Print the raw generated code


# ---------------------------------------------------------------------------
# 2. GENERATING CODE WITH STREAMING
# ---------------------------------------------------------------------------
# Streaming prints each token as it arrives instead of waiting for the full response.
# Use this for longer generations to avoid timeouts and give instant feedback.

print("\n--- 2. Streaming Code Generation ---")

def generate_code_streaming(description):
    """Generate code and print it token-by-token as Claude writes it."""
    print(f"Generating: {description}\n")
    with client.messages.stream(             # Open a streaming connection
        model=MODEL,
        max_tokens=2048,                     # Larger limit — streaming handles long outputs safely
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},  # Cache is shared with generate_code() above
        }],
        messages=[{
            "role": "user",
            "content": f"Write a Python class that: {description}",
        }],
    ) as stream:
        for text in stream.text_stream:      # Iterate over text chunks as they arrive
            print(text, end="", flush=True)  # flush=True ensures each chunk appears immediately

    final = stream.get_final_message()       # Retrieve the complete assembled message after streaming
    print(f"\n\n[Tokens used: {final.usage.output_tokens}]")  # Show how many tokens were generated

generate_code_streaming(
    "represents a simple bank account with deposit, withdraw, and balance methods. "
    "Raise a ValueError if a withdrawal exceeds the balance."
)


# ---------------------------------------------------------------------------
# 3. GENERATING CODE WITH TESTS
# ---------------------------------------------------------------------------
# Ask Claude to generate both the implementation and its unit tests in one call.

print("\n--- 3. Generating Code with Unit Tests ---")

def generate_with_tests(description):
    """Generate a Python function AND its pytest unit tests together."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "adaptive"},       # Let Claude reason before writing — improves correctness
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": (
                f"Write a Python function that: {description}\n\n"
                "Then write pytest unit tests that cover normal cases, edge cases, and error cases."
            ),
        }],
    )

    for block in response.content:          # The response may contain thinking blocks and text blocks
        if block.type == "text":            # Only print the final text, not the internal reasoning
            print(block.text)

generate_with_tests("validates an email address and returns True/False")


# ---------------------------------------------------------------------------
# 4. REFACTORING EXISTING CODE
# ---------------------------------------------------------------------------
# Paste in code you want improved and Claude rewrites it cleanly.

print("\n--- 4. Refactoring Existing Code ---")

MESSY_CODE = """
def calc(x,y,op):
    if op=="add":
        r=x+y
    elif op=="sub":
        r=x-y
    elif op=="mul":
        r=x*y
    elif op=="div":
        if y==0:
            r=None
        else:
            r=x/y
    return r
"""  # Poorly written code with bad naming and style

def refactor_code(code):
    """Ask Claude to clean up and improve existing code."""
    response = client.messages.create(
        model=MODEL,
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
                "Use good variable names, add type hints, and raise exceptions instead of returning None:\n\n"
                f"{code}"
            ),
        }],
    )
    return response.content[0].text         # Return the refactored code as a string

refactored = refactor_code(MESSY_CODE)
print("Original code refactored to:")
print(refactored)


# ---------------------------------------------------------------------------
# 5. CODE EXPLANATION
# ---------------------------------------------------------------------------
# Claude can explain what a piece of code does in plain English.

print("\n--- 5. Explaining Code ---")

COMPLEX_CODE = """
from functools import reduce
def mystery(lst):
    return reduce(lambda acc, x: acc | {x: acc.get(x, 0) + 1}, lst, {})
"""  # A compact one-liner that may be hard to read at first glance

def explain_code(code):
    """Ask Claude to explain what a piece of code does in plain English."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system="You are a Python tutor. Explain code clearly for beginners. No code in your answer.",  # Plain string system prompt
        messages=[{
            "role": "user",
            "content": f"Explain what this Python code does, step by step:\n\n{code}",
        }],
    )
    return response.content[0].text         # Return the explanation as a string

explanation = explain_code(COMPLEX_CODE)
print(explanation)


# ---------------------------------------------------------------------------
# 6. MULTI-TURN CODE ASSISTANT
# ---------------------------------------------------------------------------
# A simple loop that keeps conversation history so Claude remembers earlier turns.

print("\n--- 6. Interactive Code Assistant ---")

def code_assistant():
    """
    Run a multi-turn conversation where Claude helps you write code.
    Type 'quit' to exit.
    """
    history = []  # Stores the conversation so Claude has context across turns

    print("Code Assistant ready. Describe what you want to build.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()  # Read the user's next request

        if user_input.lower() in ("quit", "exit", "stop"):  # Check for exit commands
            print("Goodbye!")
            break

        if not user_input:                   # Skip empty input
            continue

        history.append({                     # Add the user's message to the conversation log
            "role": "user",
            "content": user_input,
        })

        with client.messages.stream(         # Stream Claude's response so it appears as it's written
            model=MODEL,
            max_tokens=2048,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # System prompt is cached across every turn
            }],
            messages=history,                # Pass the full history so Claude remembers earlier turns
        ) as stream:
            print("\nClaude: ", end="")
            response_text = ""
            for text in stream.text_stream:  # Print each token as it arrives
                print(text, end="", flush=True)
                response_text += text        # Accumulate the full response to save in history

        print("\n")

        history.append({                     # Add Claude's reply to history for the next turn
            "role": "assistant",
            "content": response_text,
        })


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # Only run when this file is executed directly
    print("\n--- Starting Interactive Code Assistant ---")
    code_assistant()        # Launch the multi-turn assistant
