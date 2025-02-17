import os
import torch
from TTS.api import TTS
from TTS.utils.text.tokenizer import Tokenizer
import soundfile as sf
import librosa
import logging

# Enable verbose logging
logging.basicConfig(level=logging.DEBUG)

# Paths to required files
MODEL_PATH = "./run 57862 9 epoch/checkpoint_56781.pth"  # Fine-tuned model path
CONFIG_PATH = "./run 57862 9 epoch/config.json"          # Path to your model's config
TOKENIZER_FILE = "./run/training/XTTS_v1.1_original_model_files/tokenizer.json"
MEL_NORM_FILE = "./run/training/XTTS_v1.1_original_model_files/mel_stats.pth"
DVAE_CHECKPOINT = "./run/training/XTTS_v1.1_original_model_files/dvae.pth"
VOCODER_PATH = "./vocoder/hifigan.pth"  # Updated vocoder path

# Verify paths
required_files = [
    MODEL_PATH,
    CONFIG_PATH,
    TOKENIZER_FILE,
    MEL_NORM_FILE,
    DVAE_CHECKPOINT,
    VOCODER_PATH
]

missing_files = [file for file in required_files if not os.path.isfile(file)]
if missing_files:
    raise FileNotFoundError(f"The following required files are missing: {missing_files}")
else:
    print("All required files are present.")

# Function to verify and convert WAV file to mono and required sample rate
def verify_and_convert_wav(wav_path, required_sr=22050, min_duration=3, output_path=None):
    """
    Verifies and converts a WAV file to meet the required sample rate and mono channel.

    Args:
        wav_path (str): Path to the input WAV file.
        required_sr (int): Required sample rate (default: 22050 Hz).
        min_duration (float): Minimum duration in seconds (default: 3 seconds).
        output_path (str, optional): Path to save the converted WAV file. 
                                     If None, overwrites the original file.

    Raises:
        ValueError: If the WAV file does not meet the required sample rate or duration.
    """
    # Load the WAV file
    data, sr = sf.read(wav_path)
    duration = len(data) / sr
    print(f"Original Sample Rate: {sr} Hz")
    print(f"Original Duration: {duration:.2f} seconds")

    # Resample if necessary
    if sr != required_sr:
        print(f"Resampling from {sr} Hz to {required_sr} Hz.")
        data = librosa.resample(data, orig_sr=sr, target_sr=required_sr)
        sr = required_sr
        print(f"Resampled Sample Rate: {sr} Hz")

    # Check duration
    if duration < min_duration:
        raise ValueError(f"Duration of {wav_path} is {duration:.2f} seconds, which is shorter than the required {min_duration} seconds.")

    # Convert to mono if necessary
    if len(data.shape) > 1:
        print("Converting to mono.")
        data = data.mean(axis=1)
        if output_path is None:
            output_path = wav_path  # Overwrite the original file
        else:
            print(f"Saving converted WAV to {output_path}.")
        sf.write(output_path, data, sr)
        print(f"Converted WAV saved to {output_path}.")
    else:
        if sr != required_sr:
            # If resampling occurred but no channel conversion was needed
            if output_path is None:
                output_path = wav_path
            print(f"Saving resampled WAV to {output_path}.")
            sf.write(output_path, data, sr)
            print(f"Resampled WAV saved to {output_path}.")
        else:
            print(f"WAV file {wav_path} already meets the requirements.")

def synthesize_speech(text, speaker_wav_path, output_wav_path, language="fr"):
    """
    Synthesizes speech from text using a reference speaker's WAV file.

    Args:
        text (str): The text to synthesize.
        speaker_wav_path (str): Path to the reference speaker's WAV file.
        output_wav_path (str): Path to save the synthesized audio.
        language (str): Language code (e.g., "fr" for French).
    """
    # Verify and possibly convert the speaker WAV file
    if speaker_wav_path:
        if not os.path.isfile(speaker_wav_path):
            raise ValueError("Reference speaker WAV file does not exist.")
        print(f"Verifying and converting speaker WAV file: {speaker_wav_path}")
        verify_and_convert_wav(speaker_wav_path, required_sr=22050, min_duration=3, output_path=speaker_wav_path)
    else:
        speaker_wav_path = None

    # Debugging: Print parameter types and values
    print(f"Text: {text}")
    print(f"Speaker WAV Path: {speaker_wav_path}")
    print(f"Output WAV Path: {output_wav_path}")
    print(f"Language: {language}")

    # Perform synthesis using the standard API
    try:
        print("Starting synthesis...")
        tts.tts_to_file(
            text=text,
            file_path=output_wav_path,
            speaker_wav=speaker_wav_path,  # Pass the file path string
            language=language              # Use the appropriate language code
        )
        print(f"Synthesized audio saved to {output_wav_path}")
    except Exception as e:
        print(f"An error occurred during synthesis: {e}")

def test_tokenizer(tokenizer_path):
    """
    Tests the tokenizer by encoding a sample text.
    """
    try:
        tokenizer = Tokenizer(tokenizer_path)
        sample_text = "Test"
        encoded = tokenizer.encode(sample_text)
        print(f"Tokenizer encoded '{sample_text}' as: {encoded.ids}")
        print("Tokenizer loaded and functioning correctly.")
    except Exception as e:
        print(f"Failed to load or use the tokenizer: {e}")

# Initialize the TTS model using the standard API
print("Loading TTS model...")
try:
    tts = TTS(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        progress_bar=False,
        gpu=torch.cuda.is_available()
    )
    print("TTS model loaded successfully.")

    # Test the tokenizer
    test_tokenizer(TOKENIZER_FILE)
except Exception as e:
    print(f"Failed to load TTS model: {e}")
    exit(1)

# Example usage
if __name__ == "__main__":
    input_text = "Bonjour tout le monde, ceci est une démonstration de synthèse vocale."
    speaker_wav = "inp.wav"          # Path to reference speaker audio
    output_wav = "./output_synthesized.wav"  # Path to save the synthesized audio

    if not os.path.isfile(speaker_wav):
        raise FileNotFoundError(f"Speaker reference audio not found: {speaker_wav}")

    synthesize_speech(input_text, speaker_wav, output_wav, language="fr")
