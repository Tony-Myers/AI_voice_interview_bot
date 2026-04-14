import streamlit as st
from openai import OpenAI
import pandas as pd
import base64
import tempfile
import os
import numpy as np
import threading
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from elevenlabs.client import ElevenLabs
import av

# ── Secrets & Configuration ────────────────────────────────────────────
PASSWORD = st.secrets["password"]
DEEPSEEK_API_KEY = st.secrets["deepseek_api_key"]
GROQ_API_KEY = st.secrets["groq_api_key"]
ELEVENLABS_API_KEY = st.secrets["elevenlabs_api_key"]
# Optional: override the default ElevenLabs voice in secrets
ELEVENLABS_VOICE_ID = st.secrets.get("elevenlabs_voice_id", "6fZce9LFNG3iEITDfqZZ")

# ── API Clients ────────────────────────────────────────────────────────
# DeepSeek uses the OpenAI-compatible endpoint
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

# Groq hosts Whisper with an OpenAI-compatible API
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ── Interview Topics ──────────────────────────────────────────────────
INTERVIEW_TOPICS = [
    "Introduction and personal background",
    "Shifting perspectives on generative AI",
    "Perceptions on workflow using generative AI tools in research contexts",
    "Perceived benefits of generative AI in research",
    "Challenges faced when using generative AI in quantitive research",
    "Challenges faced when using generative AI in qualitative research",
    "Impact on academic development and research training"
]
TOTAL_TOPICS = len(INTERVIEW_TOPICS)

OPENING_QUESTION = (
    "Let us begin the interview. Could you please introduce yourself, "
    "your role in higher education, and your interest in AI?"
)


# ── Thread-safe Audio Recorder ────────────────────────────────────────
class AudioRecorder:
    """Accumulates raw audio frames from a WebRTC callback thread."""

    def __init__(self):
        self._frames: list[np.ndarray] = []
        self._sample_rate: int | None = None
        self._lock = threading.Lock()

    def callback(self, frame: av.AudioFrame) -> av.AudioFrame:
        """WebRTC audio_frame_callback — runs on the WebRTC thread.

        av.AudioFrame.to_ndarray() returns shape (channels, samples) for
        planar formats (e.g. fltp) or (1, samples) for packed formats.
        Calling .flatten() on a multi-channel array interleaves L/R samples
        and produces garbled audio. We must take only the first channel.
        """
        with self._lock:
            arr = frame.to_ndarray()
            # arr shape is typically (channels, samples) — take first channel only
            if arr.ndim == 2:
                mono = arr[0]
            else:
                mono = arr
            self._frames.append(mono.copy())
            if self._sample_rate is None:
                self._sample_rate = frame.sample_rate
        return frame

    def has_audio(self) -> bool:
        with self._lock:
            return len(self._frames) > 0

    def get_audio_and_clear(self) -> tuple[np.ndarray | None, int | None]:
        """Return accumulated audio as a 1-D numpy array and clear buffer."""
        with self._lock:
            if not self._frames:
                return None, None
            audio = np.concatenate(self._frames)
            sr = self._sample_rate
            self._frames = []
            self._sample_rate = None
            return audio, sr


# ── Helper Functions ──────────────────────────────────────────────────

def generate_response(prompt: str, conversation_history: list[dict] | None = None) -> str:
    """Generate an interviewer follow-up via the DeepSeek API."""
    try:
        if conversation_history is None:
            conversation_history = []

        system_content = (
            "You are an experienced and considerate interviewer in higher education, "
            "focusing on AI applications. Use British English in your responses, "
            "including spellings like 'democratised'. Ensure your responses are "
            "complete and not truncated.\n\n"
            "After each user response, provide brief feedback and ask a relevant "
            "follow-up question based on their answer. Tailor your questions to the "
            "user's previous responses, avoiding repetition and exploring areas they "
            "have not covered. Be adaptive and create a natural flow of conversation."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "system", "content": f"Interview topics: {INTERVIEW_TOPICS}"},
            # Include up to the last 10 exchanges for richer context
            *conversation_history[-10:],
            {"role": "user", "content": prompt},
        ]

        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=500,
            n=1,
            temperature=0.6,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred in generate_response: {e}"


def synthesize_speech(text: str) -> bytes | None:
    """Convert text to speech via ElevenLabs. Returns raw MP3 bytes."""
    try:
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        chunks = b"".join(audio_generator)
        return chunks if chunks else None
    except Exception as e:
        st.error(f"ElevenLabs TTS error: {e}")
        return None


# Whisper prompt — biases the decoder toward expected vocabulary.
# Include terms that are likely to appear in the interview so that
# Whisper does not substitute common-sounding alternatives.
WHISPER_PROMPT = (
    "This is an academic interview about generative AI in higher education. "
    "The participant may mention: quantitative methods, qualitative methods, "
    "mixed methods, Bayesian statistics, sport science, research methodology, "
    "programme, module, lecturer, professor, pedagogy, assessment, "
    "prompt engineering, large language models, ChatGPT, generative AI, "
    "higher education, Birmingham Newman University."
)


def transcribe_audio(audio_file_path: str) -> str | None:
    """Transcribe audio via the Groq-hosted Whisper model."""
    try:
        with open(audio_file_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3",
                language="en",
                prompt=WHISPER_PROMPT,
            )
        return transcription.text
    except Exception as e:
        st.error(f"Groq transcription error: {e}")
        return None


def save_audio_to_wav(audio_data: np.ndarray, sample_rate: int) -> str:
    """Save a 1-D numpy array to a temporary WAV file and return the path."""
    if audio_data.dtype in (np.float32, np.float64):
        audio_data = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
    else:
        audio_data = audio_data.astype(np.int16)

    segment = AudioSegment(
        data=audio_data.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    segment.export(tmp.name, format="wav")
    return tmp.name


def get_transcript_csv(conversation: list[dict]) -> str:
    """Return a base64-encoded CSV download link for the transcript."""
    df = pd.DataFrame(conversation)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return (
        f'<a href="data:file/csv;base64,{b64}" '
        f'download="interview_transcript.csv">Download Transcript (CSV)</a>'
    )


# ── Main App ──────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="AI Interview Bot", page_icon="🎙️")

    # ── Authentication ─────────────────────────────────────────────────
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("AI Interview Bot")
        password = st.text_input(
            "Enter password to access the interview app:", type="password"
        )
        if st.button("Submit"):
            if password == PASSWORD:
                st.session_state.authenticated = True
                st.success("Access granted.")
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    # ── Session State Initialisation ───────────────────────────────────
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = OPENING_QUESTION
    if "recorder" not in st.session_state:
        st.session_state.recorder = AudioRecorder()

    recorder = st.session_state.recorder

    # ── Header ─────────────────────────────────────────────────────────
    st.title("🎙️ AI Interview Bot")

    # ── Consent ────────────────────────────────────────────────────────
    st.write(
        "Before we begin, please read the information sheet provided and "
        "understand that by ticking yes, you will be giving your written "
        "informed consent for your responses to be used for research purposes "
        "and may be anonymously quoted in publications.\n\n"
        "You can choose to end the interview at any time and request your data "
        "be removed by emailing [tony.myers@staff.newman.ac.uk]. This interview will "
        "be conducted by an AI assistant who, along with asking set questions, "
        "will ask additional probing questions depending on your response."
    )

    consent = st.checkbox(
        "I have read the information sheet and give my consent to participate."
    )

    if not consent:
        return

    # ── Progress ───────────────────────────────────────────────────────
    questions_asked = len(
        [e for e in st.session_state.conversation if e["role"] == "assistant"]
    )
    progress = min(questions_asked / TOTAL_TOPICS, 1.0)
    st.progress(
        progress,
        text=f"Progress: approximately {questions_asked}/{TOTAL_TOPICS} topics covered",
    )

    # ── Display Current Question ───────────────────────────────────────
    st.subheader("Interviewer")
    st.write(st.session_state.current_question)

    # Synthesise and cache audio for the current question to avoid
    # re-generating on every Streamlit rerun.
    audio_cache_key = f"tts_{hash(st.session_state.current_question)}"
    if audio_cache_key not in st.session_state:
        with st.spinner("Generating interviewer audio…"):
            st.session_state[audio_cache_key] = synthesize_speech(
                st.session_state.current_question
            )

    tts_audio = st.session_state[audio_cache_key]
    if tts_audio:
        st.audio(tts_audio, format="audio/mp3")

    # ── Audio Recording (WebRTC) ───────────────────────────────────────
    st.divider()
    st.subheader("Your Response")
    st.write(
        "Click **START** below to begin recording your answer. "
        "Click **STOP** when you are finished, then press **Submit Response**."
    )

    webrtc_ctx = webrtc_streamer(
        key="interview-recorder",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=recorder.callback,
    )

    # Show recording indicator
    if webrtc_ctx.state.playing:
        st.info("🔴 Recording in progress…")

    # ── Submit / Process ───────────────────────────────────────────────
    col_submit, col_text = st.columns([1, 1])

    with col_submit:
        submit_disabled = webrtc_ctx.state.playing
        if st.button("Submit Response", disabled=submit_disabled):
            if recorder.has_audio():
                audio_data, sample_rate = recorder.get_audio_and_clear()
                if audio_data is not None and sample_rate is not None:
                    wav_path = save_audio_to_wav(audio_data, sample_rate)

                    with st.spinner("Transcribing your response…"):
                        user_text = transcribe_audio(wav_path)
                    os.remove(wav_path)

                    if user_text:
                        _process_user_response(user_text)
                    else:
                        st.warning(
                            "Could not transcribe the audio. "
                            "Please try again or use text input."
                        )
            else:
                st.warning("No audio recorded. Please record or type your response.")

    # ── Text Fallback ──────────────────────────────────────────────────
    with col_text:
        st.write("Or type your response:")

    text_response = st.text_area(
        "Type here if microphone is unavailable:",
        key="text_input",
        label_visibility="collapsed",
    )
    if st.button("Submit Text Response"):
        if text_response and text_response.strip():
            _process_user_response(text_response.strip())
        else:
            st.warning("Please enter a response before submitting.")

    # ── Controls ───────────────────────────────────────────────────────
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("End Interview"):
            st.success(
                "Interview completed. Thank you for your insights on AI in education."
            )
            st.session_state.current_question = "Interview ended."

    with col2:
        if st.button("Restart Interview"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Transcript ─────────────────────────────────────────────────────
    if st.session_state.conversation:
        with st.expander("View Interview Transcript"):
            for entry in st.session_state.conversation:
                if entry["role"] == "assistant":
                    st.markdown(f"**🎙️ Interviewer:** {entry['content']}")
                else:
                    st.markdown(f"**👤 Participant:** {entry['content']}")
                st.write("---")

            st.markdown(
                get_transcript_csv(st.session_state.conversation),
                unsafe_allow_html=True,
            )


def _process_user_response(user_text: str):
    """Add user response to history, generate follow-up, and rerun."""
    st.session_state.conversation.append({"role": "user", "content": user_text})

    with st.spinner("Generating follow-up question…"):
        ai_prompt = (
            f"User's answer: {user_text}\n"
            f"Provide brief feedback and ask a follow-up question."
        )
        ai_response = generate_response(ai_prompt, st.session_state.conversation)

    st.session_state.conversation.append(
        {"role": "assistant", "content": ai_response}
    )
    st.session_state.current_question = ai_response
    st.rerun()


if __name__ == "__main__":
    main()
