# Speech to Text with Google Web Speech API
#
# This tutorial shows how to capture live microphone audio and transcribe it
# using Google's free Web Speech API via the `speech_recognition` library.
#
# SETUP
# -----
# Install dependencies:
#   pip install SpeechRecognition pyaudio
#
# On macOS, if pyaudio fails to install:
#   brew install portaudio
#   pip install pyaudio
#
# On Linux:
#   sudo apt-get install python3-pyaudio portaudio19-dev
#
# No API key is required — Google Web Speech is free for low-volume use.

import speech_recognition as sr


def list_microphones():
    """Print all available microphone devices and their index numbers."""
    print("Available microphones:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{index}] {name}")


def transcribe_once(mic_index=None, language="en-US"):
    """
    Listen for a single spoken phrase and return its transcription.

    Args:
        mic_index: Index of the microphone to use (None = system default).
        language:  BCP-47 language tag, e.g. "en-US", "es-ES", "de-DE".

    Returns:
        The transcribed text string, or None if recognition failed.
    """
    recognizer = sr.Recognizer()

    # Adjust for ambient noise so quiet environments don't cause false triggers
    # and loud environments don't get cut off too early.
    mic = sr.Microphone(device_index=mic_index)

    with mic as source:
        print("Calibrating for ambient noise... (hold still)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... (speak now)")
        audio = recognizer.listen(source)

    print("Transcribing...")
    try:
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        # Google could not understand the audio
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        # Network or API error
        print(f"Google API request failed: {e}")
        return None


def transcribe_loop(mic_index=None, language="en-US"):
    """
    Continuously listen and transcribe until the user says 'quit' or 'stop'.

    Each recognised phrase is printed as it comes in.
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=mic_index)

    # Calibrate once before the loop starts
    with mic as source:
        print("Calibrating for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

    print("\nListening continuously. Say 'quit' or 'stop' to exit.\n")

    while True:
        with mic as source:
            try:
                # timeout=5   → give up waiting for speech after 5 s
                # phrase_time_limit=10 → stop recording after 10 s of speech
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                # No speech detected within the timeout — loop and try again
                print("(no speech detected, still listening...)")
                continue

        try:
            text = recognizer.recognize_google(audio, language=language)
            print(f"You said: {text}")

            if text.lower() in ("quit", "stop", "exit"):
                print("Stopping.")
                break

        except sr.UnknownValueError:
            print("(could not understand)")
        except sr.RequestError as e:
            print(f"API error: {e}")
            break


# ---------------------------------------------------------------------------
# Main entry point — demonstrates both single-shot and continuous modes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Google Speech-to-Text Tutorial ===\n")

    # Uncomment the next line to see all available microphones:
    # list_microphones()

    # --- Single-shot transcription ---
    print("--- Single-shot mode ---")
    result = transcribe_once()
    if result:
        print(f"Transcription: {result}\n")

    # --- Continuous transcription ---
    print("--- Continuous mode ---")
    transcribe_loop()
