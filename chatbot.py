# Conversational Chatbot with Speech I/O and Qwen2.5
#
# This tutorial builds a chatbot that accepts input by voice or keyboard,
# generates natural-language replies with Qwen2.5-0.5B-Instruct, and delivers
# each response both as spoken audio (pyttsx3) and printed text.
#
# SETUP
# -----
# Install dependencies:
#   pip install SpeechRecognition pyaudio pyttsx3 transformers torch
#
# On macOS, if pyaudio fails to install:
#   brew install portaudio
#   pip install pyaudio
#
# On Linux:
#   sudo apt-get install python3-pyaudio portaudio19-dev espeak
#
# The first run will download Qwen2.5-0.5B-Instruct (~1 GB).
# No API key is required for any component.
#
# Expect roughly 10-15 seconds per reply on a CPU. That is slower than a small
# non-instruct model, but it is the difference between a bot that answers your
# question and one that free-associates.

import logging                                                         # Standard library logging — used to silence the transformers logger
import torch                                                           # PyTorch — required to run the model
import speech_recognition as sr                                        # Library for capturing and transcribing microphone audio
import pyttsx3                                                         # Offline text-to-speech engine
from transformers import AutoTokenizer, AutoModelForCausalLM           # Load the tokenizer and model directly

logging.getLogger("transformers").setLevel(logging.ERROR)              # Suppress transformers warnings — they use their own logger, not Python's warnings module

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # ~1 GB download; Apache-2.0 and ungated, so no Hugging Face login is needed
MAX_CONTEXT_TOKENS = 1024  # Budget for the rendered prompt — every extra token of context slows CPU generation down
MAX_NEW_TOKENS = 128       # A ceiling, not a target — generation stops at the end-of-turn token, usually well before this

# The system prompt is prepended to every request and never stored in the
# history, so trimming old turns can never discard it. Two things matter here:
# pinning the identity (asked its name, the bare model claims to be "Claude,
# created by Anthropic" — a hallucination from its training data), and demanding
# short replies, because every reply gets read aloud by the speech engine.
SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant running locally on the Qwen2.5 model. "
    "If you are asked your name, say you are a Qwen2.5 assistant. "
    "Keep replies to one or two short sentences, since they will be read aloud."
)


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def build_chatbot():
    """Load Qwen2.5-0.5B-Instruct and return (model, tokenizer)."""
    print(f"Loading {MODEL_NAME} (downloads ~1 GB on first run)...")  # Warn the user that this may take a moment
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)             # Load the tokenizer that matches the model's vocabulary
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # float32 measured ~11% faster than bfloat16 on CPUs without native bf16 support, and 0.5B still only needs ~2 GB
    )
    model.eval()                                                      # Switch to inference mode — disables dropout for deterministic output
    print("Model ready.\n")                                           # Confirm the model has loaded successfully
    return model, tokenizer                                           # Return both so callers can use them together


def build_recognizer(mic_index=None):
    """Calibrate a Recognizer and return it alongside the Microphone object."""
    recognizer = sr.Recognizer()                          # Create the object that analyses audio and calls Google Web Speech
    mic = sr.Microphone(device_index=mic_index)           # Open the chosen microphone (None = system default)
    with mic as source:                                   # Briefly open the mic stream just for calibration
        print("Calibrating microphone for ambient noise...")  # Inform the user to stay quiet
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Measure background noise for 1 second to set the silence threshold
    print("Microphone ready.\n")                          # Confirm calibration is done
    return recognizer, mic                                # Return both objects so the caller can use them for listening


# ---------------------------------------------------------------------------
# Input: voice or keyboard
# ---------------------------------------------------------------------------

def listen(recognizer, mic):
    """
    Record one spoken phrase from the microphone and return its transcription.

    Returns the transcribed string, or None if speech was not understood or
    no speech was detected within the timeout.
    """
    with mic as source:                                           # Open the mic stream for this turn
        print("Listening... (speak now)")                         # Prompt the user to speak
        try:
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=12)  # Wait up to 6 s for speech to start, record up to 12 s
        except sr.WaitTimeoutError:                               # No speech was detected within the 6-second window
            print("(no speech detected)")                         # Inform the user nothing was heard
            return None                                           # Signal that no input was captured

    try:
        text = recognizer.recognize_google(audio)  # Send the audio to Google Web Speech and receive a transcription
        print(f"You said: {text}")                 # Echo what was heard so the user can verify
        return text                                # Return the transcribed string to the caller
    except sr.UnknownValueError:                   # Audio was captured but Google could not decode it
        print("(could not understand, try again)") # Prompt the user to repeat themselves
        return None                                # Signal that transcription failed
    except sr.RequestError as e:                   # Network or API-level failure
        print(f"Speech API error: {e}")            # Print the specific error for debugging
        return None                                # Signal that the API call failed


def get_text_input():
    """Read a line of text from the keyboard and return it, or None if empty."""
    text = input("You: ")   # Wait for the user to type something and press Enter
    return text.strip() or None  # Return the stripped text, or None if the user just pressed Enter


# ---------------------------------------------------------------------------
# Output: speak + print
# ---------------------------------------------------------------------------

def respond(text, rate=160, volume=1.0):
    """
    Print the bot's reply as text and speak it aloud.

    A fresh pyttsx3 engine is created for each call rather than reused
    across turns. On Windows, the SAPI5 driver's event loop only runs
    correctly once per engine instance — reusing one instance across
    multiple say()/runAndWait() calls silently produces no audio after
    the first turn, even though say() queues the text without error.
    """
    print(f"\nBot: {text}\n")         # Print the response so the user can read it
    engine = pyttsx3.init()           # Create a fresh speech driver instance for this turn
    engine.setProperty("rate", rate)      # Set speaking speed
    engine.setProperty("volume", volume)  # Set playback volume
    engine.say(text)                  # Queue the text to be spoken
    engine.runAndWait()               # Play the queued audio and block until playback finishes
    engine.stop()                     # Release the audio driver so it can be used again next turn


# ---------------------------------------------------------------------------
# Language model: generate a reply with Qwen2.5
# ---------------------------------------------------------------------------

def build_prompt(history, tokenizer):
    """
    Render the conversation into model-ready token IDs, trimming if too long.

    Args:
        history:   List of {"role", "content"} dicts, ending with the user's
                   latest message.
        tokenizer: The AutoTokenizer returned by build_chatbot().

    Returns:
        A tuple (input_ids: torch.Tensor, history: list) — the history is
        returned because it may have been trimmed.
    """
    while True:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history  # Re-added every time, so trimming can never remove it

        # apply_chat_template renders the role structure into the exact control
        # tokens Qwen was trained on (<|im_start|>user ... <|im_end|>). Writing
        # that formatting by hand is the most common cause of garbled replies.
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # Append the opening of the assistant turn so the model replies instead of inventing your next line
            return_tensors="pt",         # Return a PyTorch tensor rather than a plain list
        )

        if input_ids.shape[-1] <= MAX_CONTEXT_TOKENS:  # The prompt fits within the budget
            return input_ids, history                   # Done
        if len(history) <= 1:                           # Only the current user message left — cannot trim any further
            return input_ids, history                   # Send it anyway rather than looping forever

        history = history[2:]  # Drop the oldest user+assistant pair, keeping whole turns intact instead of cutting mid-message


def get_reply(user_input, history, model, tokenizer):
    """
    Append user_input to the conversation, generate a reply, and return
    (reply_text, updated_history).

    Args:
        user_input: The user's latest message as a plain string.
        history:    List of {"role", "content"} dicts from previous turns,
                    or [] at the start of a conversation.
        model:      The AutoModelForCausalLM returned by build_chatbot().
        tokenizer:  The AutoTokenizer returned by build_chatbot().

    Returns:
        A tuple (reply: str, history: list).
    """
    history = history + [{"role": "user", "content": user_input}]  # Build a new list rather than mutating the caller's

    input_ids, history = build_prompt(history, tokenizer)  # Render to tokens, dropping old turns if over budget

    with torch.no_grad():                                # Disable gradient tracking — not needed for inference, saves memory
        output_ids = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),   # All 1s = every token is real, none is padding
            max_new_tokens=MAX_NEW_TOKENS,               # Cap the reply length
            pad_token_id=tokenizer.pad_token_id,         # Qwen defines a real pad token, unlike GPT-2-based models
            do_sample=True,                              # Use sampling for more natural, varied replies
            temperature=0.7,                             # Lower = more focused; higher = more creative but less coherent
            top_p=0.9,                                   # Nucleus sampling: only consider tokens whose cumulative probability reaches 90%
        )

    reply = tokenizer.decode(                            # Convert the newly generated token IDs back into a string
        output_ids[:, input_ids.shape[-1]:][0],          # Slice off the input prefix — we only want the new tokens
        skip_special_tokens=True,                        # Strip <|im_end|> and other special tokens from the output string
    ).strip()

    if not reply:                                        # Guard against an empty reply
        reply = "I'm not sure how to respond to that."   # Fall back to a safe default so the conversation doesn't stall

    return reply, history + [{"role": "assistant", "content": reply}]  # Record the reply so the next turn has context


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def chat_loop(model, tokenizer, recognizer=None, mic=None, use_voice=True, tts_rate=160, tts_volume=1.0):
    """
    Run the conversation loop until the user says or types 'quit', 'stop',
    or 'exit'.

    Args:
        model:       The AutoModelForCausalLM returned by build_chatbot().
        tokenizer:   The AutoTokenizer returned by build_chatbot().
        recognizer:  sr.Recognizer instance (required when use_voice=True).
        mic:         sr.Microphone instance (required when use_voice=True).
        use_voice:   True = accept spoken input; False = accept keyboard input.
        tts_rate:    Words per minute for spoken replies.
        tts_volume:  Volume level from 0.0 (silent) to 1.0 (full) for spoken replies.
    """
    history = []  # No conversation history at the start; an empty list means get_reply sends only the system prompt plus the first message

    print("Chat started. Say or type 'quit' to exit.\n")  # Tell the user how to stop

    while True:  # Keep the conversation going until a stop command is received
        if use_voice:                              # Voice input mode
            user_input = listen(recognizer, mic)   # Record and transcribe the user's speech
        else:                                      # Keyboard input mode
            user_input = get_text_input()          # Read a line from the terminal

        if user_input is None:  # No input was captured (timeout, unintelligible, or empty)
            continue            # Skip this iteration and prompt the user again

        if user_input.strip().lower() in ("quit", "stop", "exit"):  # Check for a stop command (case-insensitive)
            respond("Goodbye!", tts_rate, tts_volume)  # Speak and print the farewell
            break                                      # Exit the while loop and end the session

        print("(thinking...)")                                                # Generation takes 10-15 s on a CPU; without this the terminal looks frozen
        reply, history = get_reply(user_input, history, model, tokenizer)      # Generate the bot's response
        respond(reply, tts_rate, tts_volume)                                   # Speak and print the response


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # Only run when this file is executed directly (not imported)
    print("=== Chatbot Tutorial — Speech + Qwen2.5 ===\n")  # Print a title banner

    model, tokenizer = build_chatbot()  # Load the model and its tokenizer

    mode = input("Input mode — type 'v' for voice or 't' for text: ").strip().lower()  # Ask the user which input method to use
    use_voice = (mode == "v")  # True if the user chose voice, False for keyboard

    if use_voice:                                                     # Only set up the microphone if voice mode was selected
        recognizer, mic = build_recognizer()                          # Calibrate the mic and create the recognizer
        chat_loop(model, tokenizer, recognizer, mic, use_voice=True)  # Start the voice-input chat loop
    else:
        chat_loop(model, tokenizer, use_voice=False)                  # Start the text-input chat loop (no mic needed)
