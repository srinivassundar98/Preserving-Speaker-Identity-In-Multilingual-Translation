import os
os.environ["COQUI_TTS_LICENSE"] = "non-commercial"  # or "commercial" if you have a license
import streamlit as st
import time
import whisper
from transformers.models.marian.modeling_marian import MarianMTModel
from transformers.models.marian.tokenization_marian import MarianTokenizer
from TTS.api import TTS
import tempfile


# ---------------------------
# Core Pipeline Functions
# ---------------------------
def load_models(target_language="fr"):
    """
    Load STT, NMT, and TTS models dynamically based on the target language.
    """
    stt_model = whisper.load_model("base")
    nmt_model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"
    nmt_tokenizer = MarianTokenizer.from_pretrained(nmt_model_name)
    nmt_model = MarianMTModel.from_pretrained(nmt_model_name)
    tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    return stt_model, nmt_tokenizer, nmt_model, tts_model

def transcribe_audio(audio_path, stt_model):
    """
    Transcribes audio to text using the Whisper model.
    """
    result = stt_model.transcribe(audio_path)
    transcribed_text = result['text']
    return transcribed_text

def translate_text(text, nmt_tokenizer, nmt_model):
    """
    Translates text to the target language using MarianMT.
    """
    inputs = nmt_tokenizer.encode(text, return_tensors="pt", truncation=True)
    translated_tokens = nmt_model.generate(inputs, max_length=512)
    translated_text = nmt_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

def synthesize_speech_with_voice_cloning(text, output_path, tts_model, speaker_wav, language="fr"):
    """
    Synthesizes speech with voice cloning using the TTS model.
    """
    tts_model.tts_to_file(text=text, speaker_wav=speaker_wav, file_path=output_path, language=language)

def speech_translation_pipeline(audio_path, speaker_wav, output_audio_path, target_language="fr"):
    """
    End-to-end pipeline for speech translation with voice cloning.
    """
    stt_model, nmt_tokenizer, nmt_model, tts_model = load_models(target_language=target_language)
    transcribed_text = transcribe_audio(audio_path, stt_model)
    translated_text = translate_text(transcribed_text, nmt_tokenizer, nmt_model)
    synthesize_speech_with_voice_cloning(translated_text, output_audio_path, tts_model, speaker_wav, language=target_language)
    return transcribed_text, translated_text

# ---------------------------
# Streamlit User Interface
# ---------------------------
st.title("Speech Translation with Voice Cloning Demo")
st.markdown("""
This demo allows you to **upload an audio file** (in WAV format), 
transcribe it using an advanced speech-to-text model, translate it to a target language,
and then synthesize the translated speech with voice cloning.  
You can view the transcribed text, translated text, listen to the output, and download the synthesized audio.
""")

# Sidebar: Choose target language
target_language = st.sidebar.selectbox(
    "Select target language for translation",
    options=["fr", "es", "de"],
    index=0
)

# Main area: Audio file upload
st.header("Upload Your Audio File")
audio_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if audio_file is not None:
    # Save the uploaded file to a temporary file so it can be passed to the pipeline
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        input_audio_path = temp_audio.name
    st.audio(audio_file, format="audio/wav")
else:
    st.info("Please upload a WAV audio file to begin.")

# Run pipeline button
if st.button("Run Speech Translation Pipeline") and audio_file is not None:
    start_time = time.time()
    
    # For this demo, we use the uploaded audio file as the speaker reference
    speaker_wav = input_audio_path
    output_audio_path = "synthesized_output.wav"
    
    with st.spinner("Processing..."):
        transcribed_text, translated_text = speech_translation_pipeline(
            input_audio_path, speaker_wav, output_audio_path, target_language=target_language
        )
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    st.success("Pipeline completed successfully!")
    
    # Display results
    st.subheader("Transcription")
    st.write(transcribed_text)
    
    st.subheader("Translation")
    st.write(translated_text)
    
    st.subheader("Time Taken")
    st.write(f"{time_taken:.2f} seconds")
    
    # Display synthesized audio
    if os.path.exists(output_audio_path):
        with open(output_audio_path, "rb") as f:
            synthesized_audio = f.read()
        st.subheader("Synthesized Speech")
        st.audio(synthesized_audio, format="audio/wav")
        
        # Download button
        st.download_button(
            label="Download Synthesized Audio",
            data=synthesized_audio,
            file_name=output_audio_path,
            mime="audio/wav"
        )
