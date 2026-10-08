# Bot Engine — turning a group transcript into one bot's next message
#
# This module owns the language model and nothing else. It does not know that a
# chat room exists, it stores no history, and it never imports Flask.
#
# The important difference from a one-to-one chatbot engine: this one is
# *stateless*. A private chatbot can own its history, because there is exactly
# one history per conversation and only the bot appends to it. In a group chat
# the transcript is shared — humans and other bots write to it too — so a single
# owner must hold it, and that owner is the room (chat_room.py). The engine is
# handed a transcript and asked one question: "what would this persona say next?"
#
# SETUP
# -----
#   pip install transformers torch
#
# Try it standalone, without a room and without a web server:
#   python bot_engine.py

import logging                                                         # Standard library logging — used to silence the transformers logger
import threading                                                       # Provides the Lock that serialises access to the model
import torch                                                           # PyTorch — required to run the model
from transformers import AutoTokenizer, AutoModelForCausalLM           # Load the tokenizer and model directly

logging.getLogger("transformers").setLevel(logging.ERROR)              # Suppress transformers warnings — they use their own logger, not Python's warnings module

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # ~1 GB download, ~2 GB in memory as float32; Apache-2.0 and ungated, so no login is needed

# Rules appended to every persona's own prompt. A model that has only ever seen
# two-party dialogue will happily write everybody else's lines too, so the group
# setting has to be spelled out. _clean_reply() below repairs what the prompt
# fails to prevent — with a 0.5B model you need both.
GROUP_RULES = (
    "You are one participant in a group chat with several people. "
    "Each message you are shown is labelled with the name of whoever sent it. "
    "Write ONLY your own next message. "
    "Do not put your name in front of it and never write anyone else's lines. "
    "Keep it to one or two short sentences, like a real chat message."
)


class Persona:
    """
    One bot's identity: a display name plus the system prompt that produces it.

    A persona is *not* a separate model. Every persona in the room shares the
    one set of weights loaded by BotEngine.load(); all that differs is the
    system prompt prepended to the request. Three bots therefore cost the same
    ~2 GB of memory as one.
    """

    def __init__(self, name, description, system_prompt):
        self.name = name                  # How the persona appears in the transcript; also what members @mention
        self.description = description    # One line shown next to the name in the UI
        self.system_prompt = system_prompt  # The character, prepended to every request from this persona


# The starting cast. Each prompt states the identity outright, because asked
# "what is your name" the bare model confidently answers "I am Claude, created
# by Anthropic" — a hallucination absorbed from assistant transcripts in its
# training data.
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
    """
    Generates one persona's next chat message from a shared transcript.

    The model is loaded once by load() and shared by every persona. All public
    methods are safe to call from multiple threads.

    Typical use:
        engine = BotEngine()
        engine.load()
        text = engine.reply(persona, [{"speaker": "Sam", "text": "hi all"}])
    """

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        max_context_tokens=1024,  # Qwen2.5 supports far more, but every extra token of context slows CPU generation down
        max_new_tokens=96,        # Lower than a one-to-one chatbot: group messages should be short, and every bot pays this cost
        temperature=0.8,          # Slightly higher than a solo assistant, so three personas sharing one model sound less alike
        top_p=0.9,                # Nucleus sampling: only consider tokens whose cumulative probability reaches 90%
    ):
        """Store the configuration. The model is *not* loaded here — call load()."""
        self.model_name = model_name                  # Which checkpoint load() should fetch
        self.max_context_tokens = max_context_tokens  # Budget enforced in _build_prompt()
        self.max_new_tokens = max_new_tokens          # Reply length cap, passed to generate()
        self.temperature = temperature                # Sampling temperature, passed to generate()
        self.top_p = top_p                            # Nucleus sampling threshold, passed to generate()

        self._model = None      # Populated by load(); None means "not ready yet"
        self._tokenizer = None  # Populated by load(); must always match the model
        self._lock = threading.Lock()  # Serialises generation — the model is one shared resource, and CPU inference gains nothing from overlapping it

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

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reply(self, persona, transcript, member_names=()):
        """
        Generate the next message this persona would send.

        Args:
            persona:      The Persona to speak as.
            transcript:   List of {"speaker": str, "text": str} dicts, oldest
                          first. Plain dicts, not room objects — the engine must
                          not depend on the room's classes.
            member_names: Everyone currently in the room. Used to tell the model
                          who it is talking to, and to cut the reply short if it
                          starts writing someone else's line anyway.

        Returns:
            A tuple (text: str, context_tokens: int).

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        if not self.is_ready:                                              # Guard against use before loading
            raise RuntimeError("Model is not loaded yet — call load() first.")  # A clear error beats an AttributeError on None

        # Only generation needs the lock. Unlike a chatbot engine that owns its
        # history, there is no stored state here to protect — the room passed in
        # a snapshot and holds its own lock over the real transcript.
        with self._lock:
            input_ids = self._build_prompt(persona, transcript, member_names)  # Render to tokens, dropping old messages if over budget
            raw = self._generate(input_ids)                                    # Run the model

        return self._clean_reply(raw, persona, member_names), input_ids.shape[-1]  # Repair outside the lock — it is pure string work

    # -----------------------------------------------------------------------
    # Internal prompt building and generation
    # -----------------------------------------------------------------------

    def _system_prompt(self, persona, member_names):
        """Build this persona's full system prompt: character, group rules, cast list."""
        parts = [persona.system_prompt, GROUP_RULES]                 # Who you are, then how a group chat works
        others = [name for name in member_names if name != persona.name]  # Everyone except the speaker
        if others:                                                   # Skip the sentence entirely in an empty room
            parts.append("The other people in this chat are: " + ", ".join(others) + ".")  # Naming them makes @mentions and replies land on real people
        return " ".join(parts)

    def _build_prompt(self, persona, transcript, member_names):
        """
        Render a multi-speaker transcript into model-ready token IDs.

        Chat templates only know two conversational roles, so the group has to
        be flattened onto them: whatever this persona said is "assistant", and
        everything anyone else said — human or bot — is "user", prefixed with
        the speaker's name so the model can still tell them apart.
        """
        while True:
            messages = [{"role": "system", "content": self._system_prompt(persona, member_names)}]  # Re-added every time, so trimming can never discard it

            for entry in transcript:
                if entry["speaker"] == persona.name:                                  # This persona's own earlier messages
                    messages.append({"role": "assistant", "content": entry["text"]})  # No name prefix — the model must not learn to write one
                else:                                                                 # Everybody else, humans and other bots alike
                    messages.append({"role": "user", "content": f'{entry["speaker"]}: {entry["text"]}'})  # The prefix is the only thing carrying speaker identity

            # apply_chat_template renders the role structure into the exact control
            # tokens Qwen was trained on (<|im_start|>user ... <|im_end|>). Writing
            # that formatting by hand is the most common source of garbled replies.
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,  # Append the opening of the assistant turn so the model continues as this persona
                return_tensors="pt",         # Return a PyTorch tensor rather than a plain list
            )

            if input_ids.shape[-1] <= self.max_context_tokens:  # Fits within the budget
                return input_ids                                # Done
            if len(transcript) <= 1:                            # Only the triggering message left — cannot trim further
                return input_ids                                # Send it anyway rather than looping forever

            transcript = transcript[1:]  # Drop the single oldest message — group messages are not user/assistant pairs, so there is no pair to drop

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

        return self._tokenizer.decode(                       # Convert the newly generated token IDs back into a string
            output_ids[:, input_ids.shape[-1]:][0],          # Slice off the input prefix — we only want the new tokens
            skip_special_tokens=True,                        # Strip <|im_end|> and friends from the output string
        ).strip()

    def _clean_reply(self, text, persona, member_names):
        """
        Repair the two things a small model does wrong in a group chat.

        1. It labels its own message with a name — sometimes its own, and
           surprisingly often *another* bot's, because every message in the
           prompt was labelled that way and it is imitating the format.
        2. It keeps going and writes the other participants' replies too,
           because it has seen thousands of transcripts that continue.

        The position of the label is what tells the two apart. A label on the
        very first line is a mislabelled own turn, so only the label is dropped.
        A label on any later line means the model has moved on to someone
        else's turn, so everything from there is cut.
        """
        names = {n.lower() for n in member_names} | {persona.name.lower()}  # The persona may not be in member_names; it is still a label

        kept = []
        for index, line in enumerate(text.splitlines()):
            head, separator, rest = line.partition(":")           # "Newton: Verily" -> ("Newton", ":", " Verily")
            if separator and head.strip().lower() in names:       # This line starts with a member's name
                if index == 0:
                    kept.append(rest.strip())                     # Mislabelled own turn — keep what was said, drop the label
                    continue
                break                                             # A later labelled line is someone else's turn — stop here
            kept.append(line)

        text = "\n".join(kept).strip()                    # Reassemble what survived
        return text or "…"                                # Never return an empty string; the room would post a blank bubble


# ---------------------------------------------------------------------------
# Terminal demo — proof that the engine needs neither a room nor a web server
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # Only runs when this file is executed directly, not when chat_room.py imports it
    print("=== Bot Engine — terminal demo ===\n")  # Print a title banner

    engine = BotEngine()  # Build the engine with default settings
    engine.load()         # Pay the loading cost once, up front

    # A hand-written transcript. In the real application the room builds this;
    # here it is a literal, which is exactly what makes the engine easy to test.
    transcript = [
        {"speaker": "Sam", "text": "Hi everyone! What should we build this weekend?"},
        {"speaker": "Ada", "text": "Something small you can finish. A weekend is shorter than it looks."},
        {"speaker": "Sam", "text": "Newton, what do you think?"},
    ]

    names = [p.name for p in DEFAULT_PERSONAS] + ["Sam"]  # Everyone in this imaginary room

    print("\n--- transcript ---")
    for entry in transcript:                                   # Show the input the personas are answering
        print(f'{entry["speaker"]}: {entry["text"]}')

    print("\n--- each persona answers the same transcript ---")
    for persona in DEFAULT_PERSONAS:                                       # One model, three characters
        text, tokens = engine.reply(persona, transcript, names)            # The only thing that differs is persona.system_prompt
        print(f"\n{persona.name}: {text}   [{tokens} tokens of context]")
