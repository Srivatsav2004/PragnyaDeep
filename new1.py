import streamlit as st
import os
from typing import List
import pyttsx3
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableSequence
from langchain.prompts import PromptTemplate
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import tempfile

# ------------------ Helper Functions ------------------ #

@st.cache_resource
def initialize_google_api():
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "Your_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_sandhi_principles(file_path: str) -> List[Document]:
    documents = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found!")
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            cleaned_line = line.strip()
            if cleaned_line:
                documents.append(Document(page_content=cleaned_line))
    if not documents:
        raise ValueError("No valid Sandhi principles found in the file!")
    return documents

def create_vector_store(samples: List[Document], save_path: str) -> FAISS:
    embeddings = initialize_google_api()
    vector_store = FAISS.from_documents(samples, embeddings)
    vector_store.save_local(save_path)
    st.success(f"Vector store saved to {save_path}")
    return vector_store

def get_relevant_principles(vector_store: FAISS, input_text: str, k: int = 3) -> List[str]:
    similar_docs = vector_store.similarity_search(input_text, k=k)
    return [doc.page_content for doc in similar_docs]

@st.cache_resource
def setup_sandhi_chain() -> RunnableSequence:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt_template = '''
        You are a Sanskrit linguistics assistant.

        Given a Sanskrit word or phrase, perform a complete Sandhi Vigraha (word separation). 
        The input may contain **multiple Sandhi formations**, so please analyze the **entire word or phrase thoroughly**.

        Follow these steps:
        1. Split the Sanskrit correctly (Vigraha).
        2. Explain each Sandhi:
            - Combined form
            - Separated form
            - Type of Sandhi
            - Rule applied
        3. Provide the overall English translation.

        Use your knowledge of the following Sandhi principles:
        {principles}

        Input:
        {input_text}
    '''
    prompt = PromptTemplate(template=prompt_template, input_variables=["principles", "input_text"])
    return prompt | llm

def analyze_sandhi(input_text: str, vector_store: FAISS, sandhi_chain: RunnableSequence, k: int = 3):
    relevant_principles = get_relevant_principles(vector_store, input_text, k=k)
    if not relevant_principles:
        raise ValueError("No relevant Sandhi principles found for the given input.")
    result = sandhi_chain.invoke({
        "principles": "\n".join(relevant_principles),
        "input_text": input_text
    })
    return result, relevant_principles

def transcribe_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Recording... Speak now!")
        audio = recognizer.listen(source)

    try:
        st.success("Transcribing...")
        text = recognizer.recognize_google(audio, language="hi-IN")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, could not understand the audio.")
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
        return ""

from gtts import gTTS
from playsound import playsound
import tempfile

def speak_text(text: str, lang='hi'):
    # Remove any markdown or symbols
    clean_text = (
        text.replace("*", "")
            .replace("•", "")
            .replace(":", "")
            .replace("`", "")
            .replace("##", "")
            .replace("**", "")
            .replace("-", "")
            .strip()
    )

    try:
        tts = gTTS(text=clean_text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
            tts.save(tmp_path)
            playsound(tmp_path)
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")


# ------------------ Main App ------------------ #

st.set_page_config(
    page_title="Prajñādīp",
    page_icon="🪔",
    layout="wide"
)

FILE_PATH = "sandhi_samples_v2.txt"
SAVE_PATH = "sandhi_vigraha_index"
K_VALUE = 3

if 'history' not in st.session_state:
    st.session_state.history = []

def main():
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px; background-color: #f9f9f9;">
        <h1 style="margin: 0; font-size: 24px;">
            <span style="color: #FFA500;">Pragnya</span> <span style="color: #0000FF;">Deep</span>
        </h1>
        <nav>
            <a href="#" style="margin-right: 15px; text-decoration: none; color: #555;">Home</a>
            <a href="#" style="margin-right: 15px; text-decoration: none; color: #555;">Model</a>
            <a href="#" style="text-decoration: none; color: #555;">About</a>
        </nav>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    This application analyzes Sanskrit compound words (Sandhi) and splits them into their constituent parts.
    You can either **type** your input or **record** using a microphone.
    """)

    embeddings = initialize_google_api()

    if 'vector_store' not in st.session_state:
        with st.spinner("Initializing vector store..."):
            try:
                if os.path.exists(SAVE_PATH):
                    st.session_state.vector_store = FAISS.load_local(SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
                else:
                    st.info("Creating new vector store...")
                    samples = load_sandhi_principles(FILE_PATH)
                    st.session_state.vector_store = create_vector_store(samples, SAVE_PATH)
            except Exception as e:
                st.error(f"Error initializing vector store: {str(e)}")
                return

    if 'sandhi_chain' not in st.session_state:
        with st.spinner("Setting up AI model..."):
            st.session_state.sandhi_chain = setup_sandhi_chain()

    input_mode = st.radio("Choose input mode:", ("Text", "Audio"))

    input_text = ""
    if input_mode == "Text":
        input_text = st.text_input("Enter Sanskrit Sandhi word (Devanagari):", placeholder="e.g. देवालयः")
        if st.button("Analyze", type="primary") and input_text:
            with st.spinner("Analyzing Sandhi..."):
                try:
                    result, principles = analyze_sandhi(
                        input_text,
                        st.session_state.vector_store,
                        st.session_state.sandhi_chain,
                        k=K_VALUE
                    )
                    st.session_state.history.append({
                        "input": input_text,
                        "result": result.content,
                        "principles": principles
                    })
                    st.success("Analysis complete!")
                    st.markdown("## Sandhi Analysis Result")
                    st.markdown("""<div class="result-box" style="background-color: #f0f2f6; border-radius: 10px; padding: 20px;">""" + result.content + "</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error analyzing Sandhi: {str(e)}")

    elif input_mode == "Audio":
        if st.button("Record Audio", type="primary"):
            input_text = transcribe_audio_input()
            if input_text:
                st.success(f"Transcribed Text: {input_text}")
                with st.spinner("Analyzing Sandhi..."):
                    try:
                        result, principles = analyze_sandhi(
                            input_text,
                            st.session_state.vector_store,
                            st.session_state.sandhi_chain,
                            k=K_VALUE
                        )
                        st.session_state.history.append({
                            "input": input_text,
                            "result": result.content,
                            "principles": principles
                        })
                        st.success("Analysis complete!")
                        st.markdown("## Sandhi Analysis Result")
                        st.markdown("""<div class="result-box" style="background-color: #f0f2f6; border-radius: 10px; padding: 20px;">""" + result.content + "</div>", unsafe_allow_html=True)

                        speak_text(result.content)

                    except Exception as e:
                        st.error(f"Error analyzing Sandhi: {str(e)}")

    if st.session_state.history:
        st.header("Analysis History")
        history_tabs = st.tabs([item["input"] for item in reversed(st.session_state.history)])
        for i, (tab, item) in enumerate(zip(history_tabs, reversed(st.session_state.history))):
            with tab:
                st.markdown(f"### Analysis for: {item['input']}")
                st.markdown(item['result'])
                if st.button(f"Reanalyze", key=f"reanalyze_{i}"):
                    st.session_state.selected_sample = item["input"]
                    st.rerun()

    # Hide footer
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div.stButton > button:first-child {
        background-color: #FFA500;
        color: white;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF8C00;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

