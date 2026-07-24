# app.py
import os
import sys
import time
import re
import logging
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

if not os.environ.get("STREAMLIT_SERVER_FILE_WATCHER_TYPE"):
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    
def _is_running_with_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

# Self-launch via Streamlit if run via standard Python
if __name__ == "__main__" and not _is_running_with_streamlit():
    print("Bootstrapping Streamlit...")
    os.system(f"python -m streamlit run {sys.argv[0]}")
    sys.exit(0)

import streamlit as st
from llm import rewrite_script
from tts import create_tts_pipeline, synthesize
from stt import create_stt_pipeline, transcribe
from content_fetcher import fetch_content

os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)

if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
# Initialize logger for the current specific session
logger = logging.getLogger(st.session_state.session_id)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"logs/asmr_session_{st.session_state.session_id}.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

def sanitize_text(text: str) -> str:
    """Cleans up LLM artifacts and strictly converts pause tags to pure silence markers."""
    # Convert duration pause tags: [pause:2s] -> <<SILENCE2000MS>>
    text = re.sub(
        r'\[pause(?:[:=](\d+)s)?\]', 
        lambda m: f"<<SILENCE{min(5000, max(200, int(m.group(1))*1000 if m.group(1) else 1500))}MS>>", 
        text, 
        flags=re.IGNORECASE
    )
    # Convert action tags to 1-second silence markers
    text = re.sub(r'\[(?:soft breath|whisper|breath|sigh)\]', '<<SILENCE1000MS>>', text, flags=re.IGNORECASE)
    
    # Strip any remaining unhandled bracket tags and formatting asterisks to prevent TTS noise
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[\*\_]', '', text) 
    return text.strip()

def prompt_save_location(default_filename):
    """Opens a native OS 'Save As' window to let the user browse their file system."""
    try:
        root = tk.Tk()
        root.withdraw()
        # Bring the dialog to the front of all windows
        root.attributes('-topmost', True)
        file_path = filedialog.asksaveasfilename(
            initialfile=default_filename,
            title="Save ASMR Audio",
            defaultextension=".wav",
            filetypes=[("WAV Audio", "*.wav")]
        )
        root.destroy()
        return file_path
    except Exception as e:
        logger.error(f"Tkinter dialog failed: {e}")
        return None

def main():
    st.set_page_config(page_title="AI ASMR Generator", layout="centered")
    
    # Inject Custom CSS to fix input text colors and general look & feel
    st.markdown("""
    <style>
    /* Improve contrast for input controls */
    .stTextInput input, .stTextArea textarea {
        color: #1F2937 !important;
        background-color: #F9FAFB !important;
        border-radius: 6px;
    }
    div[data-baseweb="select"] > div {
        color: #1F2937 !important;
        background-color: #F9FAFB !important;
    }
    .stMultiSelect span {
        color: #1F2937 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎧 AI ASMR Generator")
    st.markdown("Provide a source topic and instructions to generate a calming ASMR audio track.")
    
    url = st.text_input("Source URL Context (Optional)")
    audio_file = st.audio_input("Voice Prompt (Instruct the ASMR topic & style)")
    
    # Humanized multi-voice selection mapping
    VOICE_MAP = {
        "Bella (American Female)": "af_bella",
        "Sarah (American Female)": "af_sarah",
        "Alloy (American Female)": "af_alloy",
        "Nicole (American Female)": "af_nicole",
        "Adam (American Male)": "am_adam",
        "Michael (American Male)": "am_michael",
        "Emma (British Female)": "bf_emma",
        "Isabella (British Female)": "bf_isabella",
        "George (British Male)": "bm_george",
        "Lewis (British Male)": "bm_lewis",
    }
    
    selected_voice_names = st.multiselect(
        "Select Voice(s) (Will alternate per paragraph)", 
        list(VOICE_MAP.keys()), 
        default=["Bella (American Female)"]
    )
    selected_voice_ids = [VOICE_MAP[name] for name in selected_voice_names]
    
    # explicit Whispering vs Soft Spoken preference control
    vocal_tone = st.radio("Vocal Tone Preference", ["Soft Spoken", "Whispering"], horizontal=True)
    duration = st.selectbox("Target Duration (Minutes)", [1, 2, 3, 4, 5])
    
    col1, col2 = st.columns(2)
    generate_new = col1.button("✨ Generate New ASMR")
    
    # Re-synthesize audio from existing script
    regenerate_audio = col2.button("🔄 Re-Synthesize (Use Existing Script)", disabled="clean_script" not in st.session_state)

    if generate_new or regenerate_audio:
        logger.info("--- New Generation/Synthesis Started ---")
        
        # Re-use existing script if requested
        if regenerate_audio and "clean_script" in st.session_state:
            clean_script = st.session_state.clean_script
            st.info("Using previously generated script for synthesis...")
        else:
            with st.spinner("Processing inputs & transcribing..."):
                user_text = ""
                if audio_file:
                    stt_pipe = create_stt_pipeline()
                    user_text = transcribe(stt_pipe, audio_file.getvalue())
                    logger.info(f"Transcribed User Prompt: {user_text}")
                    st.info(f"**Transcribed Instructions:** {user_text}")
                
                context = fetch_content(url) if url else ""
                if context:
                    logger.info(f"Fetched context from URL, length: {len(context)}")
                
            with st.spinner("Rewriting script for ASMR pacing..."):
                # Pass the requested tone down to the LLM prompt
                script = rewrite_script(context_text=context, user_prompt=user_text, target_minutes=duration, vocal_tone=vocal_tone)
                logger.info(f"Raw Generated Script:\n{script}")
                
                clean_script = sanitize_text(script)
                # Store it in session state to allow Re-synthesis without hitting LLM again
                st.session_state.clean_script = clean_script
                logger.info(f"Sanitized TTS Script:\n{clean_script}")
                
        with st.expander("View TTS-Ready Script", expanded=True):
            # Make it readable for the UI by temporarily swapping markers back to text
            st.write(clean_script.replace("<<SILENCE", "\n\n*(Silence: ").replace("MS>>", " ms)*\n\n"))
            
        with st.spinner("Synthesizing audio (this may take a moment)..."):
            tts_pipe = create_tts_pipeline()
            try:
                # Pass the tone and selected voice IDs down to the TTS engine
                audio_bytes = synthesize(tts_pipe, clean_script, voices=selected_voice_ids, vocal_tone=vocal_tone)
                st.session_state.audio_bytes = audio_bytes
                logger.info("Audio synthesis complete.")
                
                st.success("ASMR Audio Generated Successfully!")
                st.audio(audio_bytes, format="audio/wav")
                
            except Exception as e:
                logger.error(f"TTS Synthesis Failed: {str(e)}")
                st.error(f"Failed to synthesize audio: {str(e)}")
                
    # Replacing Streamlit download with native Browse/Save window
    if "audio_bytes" in st.session_state:
        st.markdown("---")
        if st.button("💾 Browse & Save Audio Locally"):
            default_filename = f"asmr_audio_{st.session_state.session_id}.wav"
            # Launch native OS browse/save window
            save_path = prompt_save_location(default_filename)
            
            if save_path:
                try:
                    with open(save_path, "wb") as f:
                        f.write(st.session_state.audio_bytes)
                    st.success(f"Successfully saved to: {save_path}")
                    logger.info(f"Saved local file to {save_path}")
                except Exception as e:
                    st.error(f"Error saving file: {e}")
                    logger.error(f"Error saving file: {e}")
        
if __name__ == "__main__":
    main()