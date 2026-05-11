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

import speech_recognition as sr  # Import the speech_recognition library under the alias "sr"


def list_microphones():
    """Print all available microphone devices and their index numbers."""
    print("Available microphones:")  # Print a header label
    for index, name in enumerate(sr.Microphone.list_microphone_names()):  # Loop over every mic name, giving each a numeric index
        print(f"  [{index}] {name}")  # Print the index and name of each microphone


def transcribe_once(mic_index=None, language="en-US"):
    """
    Listen for a single spoken phrase and return its transcription.

    Args:
        mic_index: Index of the microphone to use (None = system default).
        language:  BCP-47 language tag, e.g. "en-US", "es-ES", "de-DE".

    Returns:
        The transcribed text string, or None if recognition failed.
    """
    recognizer = sr.Recognizer()  # Create a Recognizer object that handles audio analysis and API calls

    # Adjust for ambient noise so quiet environments don't cause false triggers
    # and loud environments don't get cut off too early.
    mic = sr.Microphone(device_index=mic_index)  # Create a Microphone object tied to the chosen device (or default if None)

    with mic as source:  # Open the microphone stream; "source" is the active audio input
        print("Calibrating for ambient noise... (hold still)")  # Inform the user to stay quiet during calibration
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Sample 1 second of background noise to set the silence threshold
        print("Listening... (speak now)")  # Prompt the user to start speaking
        audio = recognizer.listen(source)  # Record audio until a pause is detected, returning an AudioData object

    print("Transcribing...")  # Inform the user that the audio is being sent for recognition
    try:
        text = recognizer.recognize_google(audio, language=language)  # Send the audio to Google Web Speech API and get back the transcription
        return text  # Return the transcribed string to the caller
    except sr.UnknownValueError:
        # Google could not understand the audio
        print("Could not understand audio.")  # Notify the user that the speech was unintelligible
        return None  # Return None to signal failure
    except sr.RequestError as e:
        # Network or API error
        print(f"Google API request failed: {e}")  # Print the specific network or API error message
        return None  # Return None to signal failure


def transcribe_loop(mic_index=None, language="en-US"):
    """
    Continuously listen and transcribe until the user says 'quit' or 'stop'.

    Each recognised phrase is printed as it comes in.
    """
    recognizer = sr.Recognizer()  # Create a fresh Recognizer for this loop session
    mic = sr.Microphone(device_index=mic_index)  # Create the Microphone object for the chosen device

    # Calibrate once before the loop starts
    with mic as source:  # Open the mic briefly just for noise calibration
        print("Calibrating for ambient noise...")  # Inform the user calibration is happening
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Measure background noise for 1 second to tune the silence threshold

    print("\nListening continuously. Say 'quit' or 'stop' to exit.\n")  # Tell the user how to end the loop

    while True:  # Loop forever until broken by a stop word or API error
        with mic as source:  # Open the mic for each listening iteration
            try:
                # timeout=5   → give up waiting for speech after 5 s
                # phrase_time_limit=10 → stop recording after 10 s of speech
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)  # Wait up to 5 s for speech to start, record up to 10 s
            except sr.WaitTimeoutError:
                # No speech detected within the timeout — loop and try again
                print("(no speech detected, still listening...)")  # Inform the user nothing was heard
                continue  # Skip the rest of the loop body and listen again

        try:
            text = recognizer.recognize_google(audio, language=language)  # Transcribe the captured audio via Google Web Speech API
            print(f"You said: {text}")  # Display what the user said

            if text.lower() in ("quit", "stop", "exit"):  # Check if the transcription is a stop command (case-insensitive)
                print("Stopping.")  # Confirm the loop is ending
                break  # Exit the while loop

        except sr.UnknownValueError:
            print("(could not understand)")  # Audio was captured but Google couldn't decode it
        except sr.RequestError as e:
            print(f"API error: {e}")  # A network or API failure occurred
            break  # Stop the loop because further requests will also fail


# ---------------------------------------------------------------------------
# Main entry point — demonstrates both single-shot and continuous modes
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # Only run the following code when this file is executed directly (not imported)
    print("=== Google Speech-to-Text Tutorial ===\n")  # Print a title banner

    # Uncomment the next line to see all available microphones:
    # list_microphones()

    # --- Single-shot transcription ---
    print("--- Single-shot mode ---")  # Label the single-shot section
    result = transcribe_once()  # Call the function to listen once and get a transcription
    if result:  # Only print if a transcription was successfully returned (not None)
        print(f"Transcription: {result}\n")  # Display the transcribed text

    # --- Continuous transcription ---
    print("--- Continuous mode ---")  # Label the continuous section
    transcribe_loop()  # Start the continuous listen-and-transcribe loop
