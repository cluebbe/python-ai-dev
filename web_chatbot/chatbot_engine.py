# Chatbot Engine — DialoGPT conversation logic, with no web framework involved
#
# This module owns everything that makes the chatbot a chatbot: loading the
# model, tracking each conversation's token history, and generating replies.
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
import torch                                                           # PyTorch — required to run the DialoGPT model
from transformers import AutoTokenizer, AutoModelForCausalLM           # Load the tokenizer and model directly

logging.getLogger("transformers").setLevel(logging.ERROR)              # Suppress transformers warnings — they use their own logger, not Python's warnings module

DEFAULT_MODEL_NAME = "microsoft/DialoGPT-medium"  # The Hugging Face model ID; downloaded once and cached in ~/.cache/huggingface/


class ChatbotEngine:
    """
    A DialoGPT chatbot that can hold several independent conversations.

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
        max_history_tokens=512,  # Keep only the most recent 512 tokens — DialoGPT-medium's hard limit is 1024; staying at half leaves room for each new reply
        max_new_tokens=100,      # Generate at most 100 new tokens so replies stay concise
        temperature=0.7,         # Lower = more focused; higher = more creative but less coherent
        top_p=0.9,               # Nucleus sampling: only consider tokens whose cumulative probability reaches 90%
    ):
        """Store the configuration. The model is *not* loaded here — call load()."""
        self.model_name = model_name                  # Which checkpoint load() should fetch
        self.max_history_tokens = max_history_tokens  # Context window budget, enforced in _generate()
        self.max_new_tokens = max_new_tokens          # Reply length cap, passed to generate()
        self.temperature = temperature                # Sampling temperature, passed to generate()
        self.top_p = top_p                            # Nucleus sampling threshold, passed to generate()

        self._model = None      # Populated by load(); None means "not ready yet"
        self._tokenizer = None  # Populated by load(); must always match the model
        self._histories = {}    # Maps conversation ID -> token history tensor, held in memory only
        self._lock = threading.Lock()  # Serialises reply() and reset() — see the note in reply()

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def load(self):
        """
        Load the model and tokenizer into memory.

        Kept separate from __init__ so that constructing an engine is instant
        and the caller decides when to pay the multi-second loading cost.
        """
        print(f"Loading {self.model_name} (downloads ~863 MB on first run)...")  # Warn the user that this may take a moment
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # Load the tokenizer that matches the model's vocabulary
        self._tokenizer.pad_token = self._tokenizer.eos_token             # DialoGPT has no pad token; reuse EOS so padding works correctly
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)  # Load the model weights
        self._model.eval()                                                # Switch to inference mode — disables dropout for deterministic output
        print("Model ready.")                                             # Confirm the model has loaded successfully

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
            A tuple (reply: str, history_tokens: int), where history_tokens is
            the size of the stored context after this turn.

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
            history_ids = self._histories.get(conversation_id)             # None on the first turn of this conversation
            reply, history_ids = self._generate(message, history_ids)      # Run the model
            self._histories[conversation_id] = history_ids                 # Store the updated history for the next turn
            return reply, history_ids.shape[-1]                            # Return the text plus the current context size

    def reset(self, conversation_id):
        """Forget one conversation's history. Safe to call for an unknown ID."""
        with self._lock:                                       # Mutating the dict under the same lock keeps it consistent with reply()
            self._histories.pop(conversation_id, None)         # Remove the entry if present; the default None means "no error if absent"

    # -----------------------------------------------------------------------
    # Internal generation
    # -----------------------------------------------------------------------

    def _generate(self, message, history_ids):
        """
        Encode the message, append it to the history, generate, and decode.

        Returns (reply_text, updated_history_ids). The caller must hold
        self._lock — this method does not acquire it.
        """
        new_ids = self._tokenizer.encode(                    # Tokenise the user's message into a tensor of integer IDs
            message + self._tokenizer.eos_token,             # Append EOS so the model knows this turn has ended
            return_tensors="pt",                             # Return a PyTorch tensor rather than a plain list
        )

        input_ids = (                                        # Build the full prompt by prepending any previous turns
            torch.cat([history_ids, new_ids], dim=-1)        # Concatenate along the sequence dimension if history exists
            if history_ids is not None                       # On the very first turn there is no history yet
            else new_ids                                     # So just use the new message on its own
        )

        if input_ids.shape[-1] > self.max_history_tokens:        # If the conversation has grown too long for the model's context window
            input_ids = input_ids[:, -self.max_history_tokens:]  # Discard the oldest tokens, keeping only the most recent ones

        attention_mask = torch.ones_like(input_ids)          # All 1s = every token is real, none is padding

        with torch.no_grad():                                # Disable gradient tracking — not needed for inference, saves memory
            output_ids = self._model.generate(
                input_ids,
                attention_mask=attention_mask,                    # Explicitly tell the model which tokens to attend to (all of them)
                max_new_tokens=self.max_new_tokens,               # Cap the reply length
                pad_token_id=self._tokenizer.eos_token_id,        # Use EOS as the pad token ID
                do_sample=True,                                   # Use sampling for more natural, varied replies
                temperature=self.temperature,                     # Sampling temperature from the constructor
                top_p=self.top_p,                                 # Nucleus sampling threshold from the constructor
            )

        reply = self._tokenizer.decode(                      # Convert the newly generated token IDs back into a string
            output_ids[:, input_ids.shape[-1]:][0],          # Slice off the input prefix — we only want the new tokens
            skip_special_tokens=True,                        # Remove EOS and other special tokens from the output string
        ).strip()

        if not reply:                                        # Guard against an empty reply (can happen on very short inputs)
            reply = "I'm not sure how to respond to that."   # Fall back to a safe default so the conversation doesn't stall

        return reply, output_ids  # Return the reply and the full token history (input + reply) for the next turn


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
