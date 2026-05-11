# Text to Speech with pyttsx3
#
# This tutorial shows how to convert text into spoken audio using the
# `pyttsx3` library, which works fully offline with no API key required.
#
# SETUP
# -----
# Install dependencies:
#   pip install pyttsx3
#
# On macOS, pyttsx3 uses the built-in "say" command — no extra setup needed.
#
# On Linux:
#   sudo apt-get install espeak
#
# On Windows, pyttsx3 uses the built-in SAPI5 engine — no extra setup needed.

import pyttsx3  # Import the text-to-speech library


def list_voices(engine):
    """Print all available voices and their index numbers."""
    print("Available voices:")  # Print a header label
    for index, voice in enumerate(engine.getProperty("voices")):  # Loop over every installed voice with a numeric index
        print(f"  [{index}] {voice.name} — {voice.id}")  # Print the index, human-readable name, and system ID of each voice


def speak_once(text, voice_index=None, rate=150, volume=1.0):
    """
    Convert a string of text to spoken audio and play it immediately.

    Args:
        text:        The string to speak.
        voice_index: Index of the voice to use (None = system default).
        rate:        Words per minute (default 150).
        volume:      Volume level from 0.0 (silent) to 1.0 (full, default).
    """
    engine = pyttsx3.init()  # Create a TTS engine instance using the platform's default driver

    engine.setProperty("rate", rate)      # Set how fast the voice speaks (words per minute)
    engine.setProperty("volume", volume)  # Set the playback volume (0.0 = mute, 1.0 = max)

    if voice_index is not None:  # Only change the voice if the caller specified one
        voices = engine.getProperty("voices")  # Fetch the list of installed voices
        engine.setProperty("voice", voices[voice_index].id)  # Switch to the chosen voice by its system ID

    engine.say(text)       # Queue the text to be spoken
    engine.runAndWait()    # Block until all queued speech has finished playing
    engine.stop()          # Release the audio driver so it can be used again


def speak_loop(voice_index=None, rate=150, volume=1.0):
    """
    Repeatedly prompt the user to type text and speak it aloud.
    Type 'quit' or 'stop' to exit.
    """
    engine = pyttsx3.init()  # Create a single engine instance to reuse across all iterations

    engine.setProperty("rate", rate)      # Set the speaking rate for the whole session
    engine.setProperty("volume", volume)  # Set the volume for the whole session

    if voice_index is not None:  # Only swap the voice if one was specified
        voices = engine.getProperty("voices")  # Retrieve all available voices
        engine.setProperty("voice", voices[voice_index].id)  # Apply the chosen voice

    print("\nText-to-Speech loop. Type 'quit' or 'stop' to exit.\n")  # Explain how to end the loop

    while True:  # Keep prompting until the user types a stop word
        text = input("Enter text to speak: ")  # Wait for the user to type something and press Enter

        if text.strip().lower() in ("quit", "stop", "exit"):  # Check for a stop command (ignoring case and surrounding spaces)
            print("Stopping.")  # Confirm the loop is ending
            break  # Exit the while loop

        if not text.strip():  # Skip empty input so the engine doesn't try to speak nothing
            print("(nothing typed, try again)")  # Prompt the user to type something
            continue  # Go back to the top of the loop

        engine.say(text)     # Queue the typed text to be spoken
        engine.runAndWait()  # Play the speech and block until it finishes

    engine.stop()  # Release the audio driver when the loop ends


# ---------------------------------------------------------------------------
# Main entry point — demonstrates both single-shot and continuous modes
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # Only run when this file is executed directly (not imported)
    print("=== Text-to-Speech Tutorial ===\n")  # Print a title banner

    # Uncomment the next two lines to see all available voices:
    _engine = pyttsx3.init()
    list_voices(_engine)

    # --- Single-shot mode ---
    print("--- Single-shot mode ---")  # Label the single-shot section
    speak_once("Hello! This is a text to speech demonstration.")  # Speak a fixed example sentence

    # --- Continuous mode ---
    print("\n--- Continuous mode ---")  # Label the continuous section
    speak_loop()  # Start the interactive type-and-speak loop
