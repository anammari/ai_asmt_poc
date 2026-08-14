# app.py
import os
import sys
import time
import re
import logging
from datetime import datetime

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
from tts import synthesize
from stt import create_stt_pipeline, transcribe
from content_fetcher import fetch_content
from preprocess_arabic import normalize_arabic

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
    """Cleans up LLM artifacts, inserts silence breaks, and normalizes Arabic for TTS."""
    # Insert silence at sentence boundaries (periods, question marks, exclamation marks,
    # new paragraphs) so the TTS has natural pacing even without [pause] markers.
    # Three consecutive newlines = paragraph break → longer pause.
    text = re.sub(r'\n{3,}', ' <<SILENCE3000MS>> ', text)
    # Two newlines = section break → medium pause.
    text = re.sub(r'\n{2,}', ' <<SILENCE2000MS>> ', text)
    # Single newline = line break → short pause.
    text = re.sub(r'\n', ' <<SILENCE1000MS>> ', text)

    # Sentence-ending punctuation → short pause if not already followed by a marker.
    text = re.sub(r'(?<=[.!?])\s+(?!<<SILENCE)', ' <<SILENCE800MS>> ', text)

    # Convert duration pause tags: [pause:2s] -> <<SILENCE2000MS>>
    text = re.sub(
        r'\[pause(?:[:=](\d+)s)?\]',
        lambda m: f"<<SILENCE{min(5000, max(200, int(m.group(1))*1000 if m.group(1) else 1500))}MS>>",
        text,
        flags=re.IGNORECASE
    )
    # Convert action tags to 1-second silence markers (English + Arabic)
    text = re.sub(
        r'\[(?:soft breath|whisper|breath|sigh|همس|تنفس|توقف)\]',
        '<<SILENCE1000MS>>',
        text,
        flags=re.IGNORECASE,
    )

    # Strip elongated onomatopoeias (shhhhh, sssss, hmmmm) to prevent TTS from spelling them letter-by-letter
    text = re.sub(r'\b(shh+|sss+|mmm+|zzz+|hmm+)\b', '<<SILENCE1000MS>>', text, flags=re.IGNORECASE)

    # Strip YouTube-style call-to-action phrases that the model learned from training data.
    text = re.sub(
        r'(?i)(لا\s*تنسوا\s*(اشتراك|لايك|متابعه|تعليق).*?[.!]|'
        r'اشتركوا\s*في\s*القناه.*?[.!]|'
        r'شاركونا\s*[^.!]*[.!]|'
        r'اعملوا\s*(لايك|متابعه|اشتراك).*?[.!]|'
        r'what\s*do\s*you\s*think.*?[.!]|'
        r'let\s*me\s*know.*?[.!])',
        '<<SILENCE1500MS>>',
        text,
    )

    # Strip any remaining unhandled bracket tags and formatting asterisks to prevent TTS noise
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[\*\_]', '', text)

    # Normalize Arabic orthography (unifies Alifs, removes tatweel, emoji, etc.)
    text = normalize_arabic(text)

    # Collapse multiple consecutive silence markers.
    while re.search(r'<<SILENCE\d+MS>>\s*<<SILENCE\d+MS>>', text):
        text = re.sub(
            r'<<SILENCE(\d+)MS>>\s*<<SILENCE(\d+)MS>>',
            lambda m: f"<<SILENCE{min(5000, int(m.group(1)) + int(m.group(2)))}MS>>",
            text,
        )

    return text.strip()

ENGLISH_VOICE_CATEGORIES = {
    "Soft Spoken": {
        "ASMR English Female 1": "fish_ca3007f96ae7499ab87d27ea3599956a",
        "ASMR English Female 2": "fish_3c97242aa71c44bdbc48713365768b2a",
    },
    "Whispering": {
        "ASMR English Female 5": "fish_efee6c804185420fb89955186451df85",
        "ASMR English Female 3": "fish_f06431add405433ba777a323911f0dfd",
        "ASMR English Female 4": "fish_084446cf1e7a4b16bafd19459ef12a06",
    },
}

ARABIC_VOICES = {
    "Whispering": {
        "ASMR 1 (Arabic Female)": "fish_0de68eaa0cc5438389b82bba728c8e39",
        "ASMR 2 (Arabic Female)": "fish_2689bc84ab944610af10bf64e586684a",
        "ASMR Syrian Female": "fish_4866140585d14d8aaaebcea8c7f0876f",
        "ASMR Egyptian Female 1": "fish_26bf953181e14b92806a4b2754513fea",
        "ASMR Egyptian Female 2": "fish_f276af83bc3c421595e27b80304f8ba0",
    },
    "Soft Spoken": {
        "ASMR 4 (Arabic Female)": "fish_7eee0787bf1a476fb0864270853e344a",
        "ASMR 5 (Arabic Female)": "fish_4ac8915eb1e04bb5a46d1e1889222f75",
    },
}

# Whispering voices are dialect-specific; Soft Spoken voices are not.
_ARABIC_WHISPERING_BY_DIALECT = {
    "Arabic": {
        "ASMR 1 (Arabic Female)": "fish_0de68eaa0cc5438389b82bba728c8e39",
        "ASMR 2 (Arabic Female)": "fish_2689bc84ab944610af10bf64e586684a",
    },
    "Syrian": {
        "ASMR Syrian Female": "fish_4866140585d14d8aaaebcea8c7f0876f",
    },
    "Egyptian": {
        "ASMR Egyptian Female 1": "fish_26bf953181e14b92806a4b2754513fea",
        "ASMR Egyptian Female 2": "fish_f276af83bc3c421595e27b80304f8ba0",
    },
}


def get_available_voices(selected_language: str, vocal_tone: str, dialect: str = "Arabic") -> dict[str, str]:
    if selected_language.startswith("Arabic"):
        if vocal_tone == "Whispering":
            return _ARABIC_WHISPERING_BY_DIALECT.get(dialect, _ARABIC_WHISPERING_BY_DIALECT["Arabic"])
        return ARABIC_VOICES.get(vocal_tone, ARABIC_VOICES["Whispering"])
    return ENGLISH_VOICE_CATEGORIES[vocal_tone]


def _resolve_user_prompt(selected_language: str, arabic_prompt: str, english_prompt: str, audio_file) -> tuple[str, str]:
    normalized_language = selected_language.strip().lower()
    if normalized_language.startswith("arabic"):
        return arabic_prompt.strip(), "typed"

    if english_prompt.strip():
        return english_prompt.strip(), "typed"

    if audio_file:
        stt_pipe = create_stt_pipeline()
        transcribed = transcribe(stt_pipe, audio_file.getvalue(), language="English")
        return transcribed, "transcribed"

    return "", "empty"


def _synthesize_and_display(clean_script: str, selected_voice_ids, vocal_tone: str) -> None:
    with st.expander("View TTS-Ready Script", expanded=False):
        st.write(clean_script.replace("<<SILENCE", "\n\n*(Silence: ").replace("MS>>", " ms)*\n\n"))

    with st.spinner("Synthesizing audio (this may take a moment)..."):
        try:
            audio_bytes = synthesize(clean_script, voices=selected_voice_ids, vocal_tone=vocal_tone)
            st.session_state.audio_bytes = audio_bytes
            logger.info("Audio synthesis complete.")
            st.success("ASMR Audio Generated Successfully!")
            st.audio(audio_bytes, format="audio/wav")
        except Exception as e:
            logger.error(f"TTS Synthesis Failed: {str(e)}")
            st.error(f"Failed to synthesize audio: {str(e)}")


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
    
    # Improvement 1: Personal Info Area for Personal Attention
    user_info = st.text_area(
        "Optional: About You (Name, age, work, hobbies, etc.)", 
        placeholder="Share a little about yourself if you'd like personalized attention in the ASMR script..."
    )
    
    st.markdown("---")
    selected_language = st.radio(
        "1. Script Language",
        ["English", "Arabic (العربية)"],
        horizontal=True,
    )

    dialect = "Arabic"
    if selected_language.startswith("Arabic"):
        dialect = st.selectbox("Dialect", ["Arabic", "Syrian", "Egyptian"])

    arabic_prompt = ""
    english_prompt = ""
    audio_file = None
    if selected_language.startswith("Arabic"):
        arabic_prompt = st.text_area(
            "Arabic Written Prompt (النص العربي)",
            placeholder="اكتب التعليمات العربية هنا...",
        )
    else:
        english_prompt = st.text_area(
            "English Written Prompt",
            placeholder="e.g. Explain the soothing sound of gentle rain falling on a window and a show relaxing whisper",
        )
        audio_file = st.audio_input("Or record a Voice Prompt (Instruct the ASMR topic & style)")

    # Fix 1: Vocal tone selection strictly precedes and filters the voice selection
    vocal_tone = st.radio("2. Vocal Tone Preference", ["Whispering", "Soft Spoken"], horizontal=True)
    
    available_voices = get_available_voices(selected_language, vocal_tone, dialect)
    
    selected_voice_names = st.multiselect(
        "3. Select Voice(s) (Filtered by tone; will alternate per paragraph)", 
        list(available_voices.keys()), 
        default=[list(available_voices.keys())[0]]
    )
    selected_voice_ids = [available_voices[name] for name in selected_voice_names]
    
    duration = st.selectbox("4. Target Duration (Minutes)", [1, 2, 3, 4, 5])
    
    st.markdown("---")
    
    generate_new = st.button("✨ Generate New ASMR")

    if generate_new:
        logger.info("--- New Generation Started ---")
        with st.spinner("Processing inputs & transcribing..."):
            user_text, prompt_mode = _resolve_user_prompt(selected_language, arabic_prompt, english_prompt, audio_file)
            if prompt_mode == "typed":
                if not user_text:
                    st.error("Please enter written instructions before generating.")
                    return
                is_arabic_ui = selected_language.startswith("Arabic")
                label = "Arabic" if is_arabic_ui else "English"
                logger.info(f"{label} Written Prompt: {user_text}")
                st.info(f"**{label} Written Instructions:** {user_text}")
            elif prompt_mode == "transcribed":
                logger.info(f"Transcribed User Prompt: {user_text}")
                st.info(f"**Transcribed Instructions:** {user_text}")

            context = fetch_content(url) if url else ""
            if context:
                logger.info(f"Fetched context from URL, length: {len(context)}")

        with st.spinner("Rewriting script for ASMR pacing..."):
            try:
                script = rewrite_script(
                    context_text=context,
                    user_prompt=user_text,
                    target_minutes=duration,
                    vocal_tone=vocal_tone,
                    user_info=user_info, # Pass the new personal info down to the LLM
                    language=selected_language,
                    dialect=dialect,
                )
            except Exception as e:
                logger.error(f"Script generation failed: {str(e)}")
                st.error(f"Failed to generate script: {str(e)}")
                return
            logger.info(f"Raw Generated Script:\n{script}")

            clean_script = sanitize_text(script)
            st.session_state.clean_script = clean_script
            logger.info(f"Sanitized TTS Script:\n{clean_script}")

        _synthesize_and_display(clean_script, selected_voice_ids, vocal_tone)

    # Rendered after generation so it reflects the current session state (enabled once a script exists).
    regenerate_audio = st.button("🔄 Re-Synthesize (Use Existing Script)", disabled="clean_script" not in st.session_state)
    if regenerate_audio:
        logger.info("--- Re-Synthesis Started ---")
        clean_script = st.session_state.clean_script
        st.info("Using previously generated script for synthesis...")
        _synthesize_and_display(clean_script, selected_voice_ids, vocal_tone)

    # Fix 2 & Improvement 3: Replaced Tkinter OS dialog with Streamlit's native web-safe Download Button
    if "audio_bytes" in st.session_state:
        st.markdown("---")
        default_filename = f"asmr_audio_{st.session_state.session_id}.wav"
        
        st.download_button(
            label="💾 Download Audio",
            data=st.session_state.audio_bytes,
            file_name=default_filename,
            mime="audio/wav",
            use_container_width=True
        )
        
if __name__ == "__main__":
    main()