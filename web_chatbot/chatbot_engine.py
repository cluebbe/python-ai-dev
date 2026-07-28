# Chatbot Engine — Qwen2.5 conversation logic, with no web framework involved
#
# This module owns everything that makes the chatbot a chatbot: loading the
# model, tracking each conversation's message history, and generating replies.
# It deliberately imports nothing from Flask. It does not know what HTTP is,
# what a cookie is, or that a browser exists.
#
# Conversations are identified by an opaque string ID chosen by the caller.
# The web layer uses a session ID; the terminal demo at the bottom of this file
# uses the literal string "terminal". The engine does not care which.
#
# SETUP
# -----
#   pip install transformers torch
#
# Try it standalone, without a web server:
#   python chatbot_engine.py

import logging                                                         # Standard library logging — used to silence the transformers logger
import threading                                                       # Provides the Lock that serialises access to the model
import torch                                                           # PyTorch — required to run the model
from transformers import AutoTokenizer, AutoModelForCausalLM           # Load the tokenizer and model directly

logging.getLogger("transformers").setLevel(logging.ERROR)              # Suppress transformers warnings — they use their own logger, not Python's warnings module

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # ~1 GB download, ~2 GB in memory as float32; Apache-2.0 and ungated, so no login is needed

# The system prompt is prepended to every request and is never stored in the
# history, so trimming old turns can never discard it. Pinning the identity here
# matters: asked "what is your name", the bare model confidently answers "I am
# Claude, created by Anthropic" — a hallucination picked up from assistant
# transcripts in its training data. Stating what it actually is fixes that.
DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly, concise chat assistant running locally on the Qwen2.5 model. "
    "If you are asked your name, say you are a Qwen2.5 assistant. "
    "Keep replies to one or two sentences."
)


class ChatbotEngine:
    """
    A chat assistant that can hold several independent conversations.

    The model is loaded once by load() and reused for every reply. All public
    methods are safe to call from multiple threads.

    Typical use:
        engine = ChatbotEngine()
        engine.load()
        reply, tokens = engine.reply("some-conversation-id", "Hello!")
    """

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_context_tokens=1024,  # Qwen2.5 supports far more, but every extra token of context slows CPU generation down
        max_new_tokens=128,       # A ceiling, not a target — generation stops at the end-of-turn token, usually well before this
        temperature=0.7,          # Lower = more focused; higher = more creative but less coherent
        top_p=0.9,                # Nucleus sampling: only consider tokens whose cumulative probability reaches 90%
    ):
        """Store the configuration. The model is *not* loaded here — call load()."""
        self.model_name = model_name                  # Which checkpoint load() should fetch
        self.system_prompt = system_prompt            # Prepended to every request, never stored in history
        self.max_context_tokens = max_context_tokens  # Budget enforced in _build_prompt()
        self.max_new_tokens = max_new_tokens          # Reply length cap, passed to generate()
        self.temperature = temperature                # Sampling temperature, passed to generate()
        self.top_p = top_p                            # Nucleus sampling threshold, passed to generate()

        self._model = None      # Populated by load(); None means "not ready yet"
        self._tokenizer = None  # Populated by load(); must always match the model
        self._histories = {}    # Maps conversation ID -> list of {"role", "content"} dicts, held in memory only
        self._lock = threading.Lock()  # Serialises reply() and reset() — see the note in reply()

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def load(self):
        """
        Load the model and tokenizer into memory.

        Kept separate from __init__ so that constructing an engine is instant
        and the caller decides when to pay the loading cost.
        """
        print(f"Loading {self.model_name} (downloads ~1 GB on first run)...")  # Warn the user that this may take a moment
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)      # Load the tokenizer that matches the model's vocabulary
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # float32 measured ~11% faster than bfloat16 on CPUs without native bf16, and 0.5B still only needs ~2 GB
        )
        self._model.eval()                                                    # Switch to inference mode — disables dropout for deterministic output
        print("Model ready.")                                                 # Confirm the model has loaded successfully

    @property
    def is_ready(self):
        """True once load() has finished. Callers use this to answer requests that arrive too early."""
        return self._model is not None  # The model is assigned last in load(), so this is only True when everything is in place

    @property
    def active_conversations(self):
        """How many conversations currently hold history. Useful for health checks and debugging."""
        return len(self._histories)  # Grows by one per new conversation ID, shrinks on reset()

    # -----------------------------------------------------------------------
    # Public conversation API
    # -----------------------------------------------------------------------

    def reply(self, conversation_id, message):
        """
        Generate a reply within the given conversation and remember it.

        Args:
            conversation_id: Opaque string identifying the conversation. Any
                             new value starts a fresh conversation.
            message:         The user's message as a plain string.

        Returns:
            A tuple (reply: str, context_tokens: int), where context_tokens is
            the size of the rendered prompt that produced this reply.

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        if not self.is_ready:                                              # Guard against use before loading
            raise RuntimeError("Model is not loaded yet — call load() first.")  # A clear error beats an AttributeError on None

        # The lock covers the whole read-modify-write cycle, not just generation.
        # Reading the history, generating, and storing the result must happen as
        # one unit: if two messages for the same conversation overlap, the slower
        # one would otherwise overwrite the faster one's history and silently
        # drop a turn. Generation is the slow part anyway, so widening the lock
        # to include the dictionary access costs effectively nothing.
        with self._lock:
            history = self._histories.get(conversation_id, [])                 # [] on the first turn of this conversation
            history = history + [{"role": "user", "content": message}]         # Build a new list rather than mutating the stored one

            input_ids, history = self._build_prompt(history)                   # Render to tokens, dropping old turns if over budget
            reply = self._generate(input_ids)                                  # Run the model

            history = history + [{"role": "assistant", "content": reply}]      # Record the reply so the next turn has context
            self._histories[conversation_id] = history                         # Store the updated history
            return reply, input_ids.shape[-1]                                  # Return the text plus the prompt size

    def reset(self, conversation_id):
        """Forget one conversation's history. Safe to call for an unknown ID."""
        with self._lock:                                       # Mutating the dict under the same lock keeps it consistent with reply()
            self._histories.pop(conversation_id, None)         # Remove the entry if present; the default None means "no error if absent"

    # -----------------------------------------------------------------------
    # Internal prompt building and generation
    # -----------------------------------------------------------------------

    def _build_prompt(self, history):
        """
        Render history into model-ready token IDs, trimming until it fits.

        Returns (input_ids, history), where history is the possibly-trimmed list
        that should be stored. The caller must hold self._lock.
        """
        while True:
            messages = [{"role": "system", "content": self.system_prompt}] + history  # System prompt is re-added each time, so it can never be trimmed away

            # apply_chat_template renders the role structure into the exact control
            # tokens Qwen was trained on (<|im_start|>user ... <|im_end|>). Writing
            # that formatting by hand is the most common source of garbled replies.
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,  # Append the opening of the assistant turn so the model continues as the assistant
                return_tensors="pt",         # Return a PyTorch tensor rather than a plain list
            )

            if input_ids.shape[-1] <= self.max_context_tokens:  # Fits within the budget
                return input_ids, history                       # Done
            if len(history) <= 1:                               # Only the current user message left — cannot trim further
                return input_ids, history                       # Send it anyway rather than looping forever

            history = history[2:]  # Drop the oldest user+assistant pair, keeping turns intact instead of cutting mid-message

    def _generate(self, input_ids):
        """
        Generate a reply for an already-rendered prompt and decode it.

        The caller must hold self._lock — this method does not acquire it.
        """
        with torch.no_grad():                                # Disable gradient tracking — not needed for inference, saves memory
            output_ids = self._model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),        # All 1s = every token is real, none is padding
                max_new_tokens=self.max_new_tokens,               # Cap the reply length
                pad_token_id=self._tokenizer.pad_token_id,        # Qwen defines a real pad token, unlike GPT-2-based models
                do_sample=True,                                   # Use sampling for more natural, varied replies
                temperature=self.temperature,                     # Sampling temperature from the constructor
                top_p=self.top_p,                                 # Nucleus sampling threshold from the constructor
            )

        reply = self._tokenizer.decode(                      # Convert the newly generated token IDs back into a string
            output_ids[:, input_ids.shape[-1]:][0],          # Slice off the input prefix — we only want the new tokens
            skip_special_tokens=True,                        # Strip <|im_end|> and friends from the output string
        ).strip()

        if not reply:                                        # Guard against an empty reply
            reply = "I'm not sure how to respond to that."   # Fall back to a safe default so the conversation doesn't stall

        return reply


# ---------------------------------------------------------------------------
# Terminal demo — proof that the engine does not need a web server
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # Only runs when this file is executed directly, not when app.py imports it
    print("=== Chatbot Engine — terminal demo ===\n")  # Print a title banner

    engine = ChatbotEngine()  # Build the engine with default settings
    engine.load()             # Pay the loading cost once, up front

    print("\nChat started. Type 'quit' to exit.\n")  # Tell the user how to stop

    while True:  # Keep the conversation going until a stop command is received
        message = input("You: ").strip()  # Read a line from the terminal
        if not message:                   # The user just pressed Enter
            continue                      # Prompt again without calling the model
        if message.lower() in ("quit", "stop", "exit"):  # Check for a stop command (case-insensitive)
            print("\nBot: Goodbye!\n")                   # Say farewell
            break                                        # Exit the loop and end the session

        reply, tokens = engine.reply("terminal", message)  # One fixed conversation ID is enough for a single-user terminal
        print(f"\nBot: {reply}   [{tokens} tokens of context]\n")  # Show the reply and the current context size
