import streamlit as st
from openai import OpenAI  # Updated import for client-based interface
import pandas as pd
import base64
import io
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
from pydub import AudioSegment

# Retrieve the password and OpenAI API key from Streamlit secrets
PASSWORD = st.secrets["password"]
OPENAI_API_KEY = st.secrets["openai_api_key"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# List of interview topics
interview_topics = [
    "Introduction and personal background",
    "Experiences using generative AI tools",
    "Perceived benefits of generative AI in coursework",
    "Challenges faced when integrating AI tools",
    "Impact on academic development and assessment performance",
    "Strategies for incorporating AI outputs into work",
    "Effectiveness of guidance on prompt design and AI competencies",
    "Generative AI's role in creating accessible educational environments"
]

total_questions = len(interview_topics)  # Total number of interview topics for progress bar

def generate_response(conversation_history):
    try:
        system_content = """You are an experienced and considerate interviewer in higher education, focusing on AI applications. Use British English in your responses, including spellings like 'democratised'. Ensure your responses are complete and not truncated. 
After each user response, provide brief feedback and ask a relevant follow-up question based on their answer. Tailor your questions to the user's previous responses, avoiding repetition and exploring areas they haven't covered. Be adaptive and create a natural flow of conversation."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "system", "content": f"Interview topics: {interview_topics}"},
            *conversation_history[-6:],  # Include the last 6 exchanges for more context
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=300,
            n=1,
            temperature=0.6,
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"An error occurred in generate_response: {str(e)}"

def get_transcript_download_link(conversation):
    df = pd.DataFrame(conversation)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="interview_transcript.csv">Download Transcript</a>'
    return href

# Custom Audio Processor to capture audio data
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = []

    def recv(self, frame):
        self.audio_data.append(frame.to_ndarray())
        return frame

def main():
    # Password authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Secure AI Interview Bot Access")
        password = st.text_input("Enter password to access the interview app:", type="password")
        if st.button("Submit"):
            if password == PASSWORD:
                st.session_state.authenticated = True
                st.success("Access granted.")
                st.experimental_rerun()
            else:
                st.error("Incorrect password.")
        return  # Stop the app here if not authenticated

    # Interview app content (only shown if authenticated)
    st.title("AI Interview Bot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = "Let's begin the interview. Can you please introduce yourself, your role in higher education, and your interest in AI?"
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "interview_ended" not in st.session_state:
        st.session_state.interview_ended = False

    if st.session_state.interview_ended:
        st.write("The interview has ended. Thank you for your participation!")
        if st.session_state.conversation:
            st.markdown(get_transcript_download_link(st.session_state.conversation), unsafe_allow_html=True)
        return

    st.write("""
    **Informed Consent:**

    Before we begin, please read the information sheet provided and understand that by ticking yes, you will be giving your written informed consent for your responses to be used for research purposes and may be anonymously quoted in publications.

    You can choose to end the interview at any time and request your data be removed by emailing [tony.myers@staff.ac.uk](mailto:tony.myers@staff.ac.uk). This interview will be conducted by an AI assistant who, along with asking set questions, will ask additional probing questions depending on your response.
    """)

    consent = st.checkbox("I have read the information sheet and give my consent to participate in this interview.")

    if consent:
        st.write(f"**Question:** {st.session_state.current_question}")

        st.write("**You can respond by recording your voice or typing your answer below:**")

        # Voice input section using streamlit-webrtc
        webrtc_ctx = webrtc_streamer(
            key="audio_recorder",
            mode="audio",
            client_settings=ClientSettings(
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"audio": True, "video": False},
            ),
            audio_processor_factory=AudioProcessor,
            async_processing=True,
        )

        user_answer = ""

        # Process recorded audio
        if webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.audio_data:
            # Concatenate all audio frames
            audio_frames = webrtc_ctx.audio_processor.audio_data
            audio_np = np.concatenate(audio_frames, axis=0)
            # Convert numpy array to bytes
            audio_bytes = audio_np.tobytes()

            # Convert raw audio to WAV format using pydub
            audio_segment = AudioSegment(
                data=audio_bytes,
                sample_width=2,  # 16-bit audio
                frame_rate=16000,
                channels=1
            )
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            # Transcribe audio using OpenAI's Whisper API
            try:
                with st.spinner("Transcribing audio..."):
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=wav_io
                    )
                user_answer = transcript['text']
                st.write(f"**Transcribed Text:** {user_answer}")
            except Exception as e:
                st.error(f"An error occurred during transcription: {str(e)}")

        # Text input as an alternative
        text_input = st.text_area("Or type your response:", key=f"user_input_{len(st.session_state.conversation)}")

        # Use text input if provided
        if text_input:
            user_answer = text_input

        # Progress bar with a label indicating interview progress
        completed_questions = len([entry for entry in st.session_state.conversation if entry['role'] == "user"])
        progress_percentage = min(completed_questions / total_questions, 1.0)  # Ensure it doesn't exceed 100%
        st.write(f"**Interview Progress: {completed_questions} out of {total_questions} questions answered**")
        st.progress(progress_percentage)

        if st.button("Submit Answer"):
            if user_answer.strip():
                # Add user's answer to conversation history
                st.session_state.conversation.append({"role": "user", "content": user_answer.strip()})

                # Generate AI response
                ai_response = generate_response(st.session_state.conversation)

                # Add AI's response to conversation history
                st.session_state.conversation.append({"role": "assistant", "content": ai_response})

                # Update current question with AI's follow-up
                st.session_state.current_question = ai_response

                # Reset flags
                st.session_state.submitted = True
                webrtc_ctx.audio_processor.audio_data = []

                st.experimental_rerun()
            else:
                st.warning("Please provide an answer before submitting.")

        # Option to end the interview
        if st.button("End Interview"):
            st.success("Interview completed! Thank you for your insights into AI use in your education.")
            st.session_state.interview_ended = True
            st.experimental_rerun()

    else:
        st.warning("You must provide consent to participate in the interview.")

    # Display transcript download link if interview has ended
    if st.session_state.interview_ended and st.session_state.conversation:
        st.markdown(get_transcript_download_link(st.session_state.conversation), unsafe_allow_html=True)

if __name__ == "__main__":
    import numpy as np  # Added import for numpy
    main()
