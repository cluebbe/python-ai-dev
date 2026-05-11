# Conversational Chatbot with Speech I/O and DialoGPT
#
# This tutorial builds a chatbot that accepts input by voice or keyboard,
# generates natural-language replies with Microsoft's DialoGPT, and delivers
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
# The first run will download the DialoGPT-medium model (~863 MB).
# No API key is required for any component.

import speech_recognition as sr              # Library for capturing and transcribing microphone audio
import pyttsx3                               # Offline text-to-speech engine
from transformers import pipeline            # Hugging Face helper that wraps models into simple callable objects


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

EOS = "<|endoftext|>"  # Special token DialoGPT uses to separate conversation turns


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def build_chatbot():
    """Load DialoGPT-medium and return a text-generation pipeline."""
    print("Loading DialoGPT-medium (downloads ~863 MB on first run)...")  # Warn the user that this may take a moment
    bot = pipeline("text-generation", model="microsoft/DialoGPT-medium")  # Wrap the model in a text-generation pipeline
    print("Model ready.\n")  # Confirm the model has loaded successfully
    return bot  # Return the pipeline so the caller can store and reuse it


def build_tts():
    """Create and return a configured pyttsx3 TTS engine."""
    engine = pyttsx3.init()              # Initialise the platform's default speech driver
    engine.setProperty("rate", 160)      # Set speaking speed to 160 words per minute (slightly slower than default for clarity)
    engine.setProperty("volume", 1.0)    # Set volume to maximum
    return engine                        # Return the engine so the caller can reuse it across turns


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

def respond(text, engine):
    """Print the bot's reply as text and speak it aloud."""
    print(f"\nBot: {text}\n")  # Print the response so the user can read it
    engine.say(text)           # Queue the text to be spoken
    engine.runAndWait()        # Play the queued audio and block until playback finishes


# ---------------------------------------------------------------------------
# Language model: generate a reply with DialoGPT
# ---------------------------------------------------------------------------

def get_reply(user_input, history, chatbot):
    """
    Append user_input to the conversation history, generate a reply with
    DialoGPT, and return (reply_text, updated_history).

    Args:
        user_input: The user's latest message as a plain string.
        history:    List of alternating user/bot strings from this session.
        chatbot:    The text-generation pipeline returned by build_chatbot().

    Returns:
        A tuple (reply: str, history: list).
    """
    history.append(user_input)                    # Add the user's message to the running conversation log
    prompt = EOS.join(history) + EOS              # Join all turns with the EOS token; DialoGPT uses this format to track context

    output = chatbot(
        prompt,
        max_new_tokens=100,    # Generate at most 100 new tokens so replies stay concise
        pad_token_id=50256,    # 50256 is the EOS token ID for GPT-2 / DialoGPT; avoids a pad-token warning
        do_sample=True,        # Use sampling instead of greedy decoding for more natural, varied replies
        temperature=0.7,       # Lower temperature → more focused replies; higher → more creative but less coherent
        top_p=0.9,             # Nucleus sampling: only consider tokens whose cumulative probability reaches 90%
    )

    generated = output[0]["generated_text"]         # The pipeline returns the full string including the original prompt
    reply = generated[len(prompt):]                 # Strip the prompt prefix to isolate only the newly generated text
    reply = reply.split(EOS)[0].strip()             # DialoGPT may generate multiple turns; take only the first one

    if not reply:                                   # Guard against an empty reply (can happen on very short inputs)
        reply = "I'm not sure how to respond to that."  # Fall back to a safe default so the conversation doesn't stall

    history.append(reply)  # Add the bot's reply to the history so future turns have full context
    return reply, history  # Return the reply text and the updated history list


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def chat_loop(chatbot, tts_engine, recognizer=None, mic=None, use_voice=True):
    """
    Run the conversation loop until the user says or types 'quit', 'stop',
    or 'exit'.

    Args:
        chatbot:     The DialoGPT text-generation pipeline.
        tts_engine:  The pyttsx3 engine used for audio output.
        recognizer:  sr.Recognizer instance (required when use_voice=True).
        mic:         sr.Microphone instance (required when use_voice=True).
        use_voice:   True = accept spoken input; False = accept keyboard input.
    """
    history = []  # Start with an empty conversation history for this session

    print("Chat started. Say or type 'quit' to exit.\n")  # Tell the user how to stop

    while True:  # Keep the conversation going until a stop command is received
        if use_voice:                              # Voice input mode
            user_input = listen(recognizer, mic)   # Record and transcribe the user's speech
        else:                                      # Keyboard input mode
            user_input = get_text_input()          # Read a line from the terminal

        if user_input is None:  # No input was captured (timeout, unintelligible, or empty)
            continue            # Skip this iteration and prompt the user again

        if user_input.strip().lower() in ("quit", "stop", "exit"):  # Check for a stop command (case-insensitive)
            print("Goodbye!")          # Print a farewell message
            respond("Goodbye!", tts_engine)  # Speak the farewell so the user hears it too
            break                      # Exit the while loop and end the session

        reply, history = get_reply(user_input, history, chatbot)  # Generate the bot's response using DialoGPT
        respond(reply, tts_engine)                                 # Speak and print the response


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # Only run when this file is executed directly (not imported)
    print("=== Chatbot Tutorial — Speech + DialoGPT ===\n")  # Print a title banner

    bot = build_chatbot()    # Load the DialoGPT model
    tts = build_tts()        # Set up the text-to-speech engine

    mode = input("Input mode — type 'v' for voice or 't' for text: ").strip().lower()  # Ask the user which input method to use
    use_voice = (mode == "v")  # True if the user chose voice, False for keyboard

    if use_voice:                                    # Only set up the microphone if voice mode was selected
        recognizer, mic = build_recognizer()         # Calibrate the mic and create the recognizer
        chat_loop(bot, tts, recognizer, mic, use_voice=True)   # Start the voice-input chat loop
    else:
        chat_loop(bot, tts, use_voice=False)         # Start the text-input chat loop (no mic needed)
