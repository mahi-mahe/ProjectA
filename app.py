# app.py
import os
import tempfile
from datetime import datetime
from typing import Optional

import streamlit as st

# LangChain / connectors
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -------------------------
# Streamlit page & CSS
# -------------------------
st.set_page_config(
    page_title="Corporate Training RAG Bot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("RAG Chatbot with Streamlit")

st.markdown(
    """
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; font-weight: bold; text-align: center; margin-bottom: 1rem; }
    .sub-header { text-align: center; color: #666; margin-bottom: 2rem; }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid; }
    .user-message { background-color: #e3f2fd; border-left-color: #2196f3; }
    .bot-message { background-color: #f5f5f5; border-left-color: #4caf50; }
    .info-box { background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; margin: 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# How we load the Gemini key (no user input)
# -------------------------
def get_gemini_key() -> Optional[str]:
    """
    Priority:
      1. Streamlit secrets: st.secrets["GEMINI_API_KEY"]
      2. Environment variable: os.environ["GEMINI_API_KEY"]
      3. None if not found
    """
    # Preferred: Streamlit secrets (works on Streamlit Cloud / deployed apps)
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
        if secret_key:
            return secret_key
    except Exception:
        # st.secrets may raise if not present (older streamlit). ignore and fallback.
        pass

    # Fallback: environment variable
    return os.environ.get("GEMINI_API_KEY")


# Display a small sidebar note about secrets
st.sidebar.markdown(
    """
**Authentication (Gemini)**  
This app reads the Gemini API key from **Streamlit secrets** (`GEMINI_API_KEY`) or from the environment variable `GEMINI_API_KEY`.
- On Streamlit Cloud: add it in **Settings → Secrets** or create `.streamlit/secrets.toml`.
- Locally: either set an env var or create `.streamlit/secrets.toml`.
"""
)

# -------------------------
# In-memory conversation state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Training documents (paste your full TRAINING_DOCUMENTS here)
# -------------------------
TRAINING_DOCUMENTS = [
    # Paste your TRN001 - TRN008 dicts here (same as your original data).
    # Example:
    # {
    #    "title": "Python Programming Fundamentals",
    #    "id": "TRN001",
    #    "category": "Technical Skills",
    #    "level": "Beginner",
    #    "duration": "40 hours",
    #    "content": "..."
    # },
]

# -------------------------
# Load RAG pipeline (uses Gemini LLM + HuggingFace embeddings)
# -------------------------
@st.cache_resource
def load_rag_pipeline(gemini_api_key: Optional[str]):
    """
    Build RAG pipeline. Requires a Gemini API key (api passed directly to the LLM).
    Embeddings use a local Hugging Face model (sentence-transformers/all-MiniLM-L6-v2).
    Vector DB: Chroma (in-memory/default directory).
    """
    if not gemini_api_key:
        # If no key, return None — caller will handle
        return None

    # Use a compact HF embedding model (local)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma vectorstore (uses embeddings)
    vectorstore = Chroma(embedding_function=embeddings, collection_name="my_rag_collection")

    retriever = vectorstore.as_retriever()

    # Create Gemini-based LLM via langchain_google_genai
    llm = ChatGoogleGenerativeAI(api_key=gemini_api_key)  # pass key directly

    # Create a simple RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return {"qa_chain": qa_chain, "vectorstore": vectorstore, "embeddings": embeddings}


# -------------------------
# Acquire key (securely) and initialize pipeline
# -------------------------
gemini_key = get_gemini_key()

if not gemini_key:
    st.error(
        "Gemini API key not found. Please set `GEMINI_API_KEY` in Streamlit secrets or as an environment variable."
    )
    st.info(
        "Local setup example: create a file `.streamlit/secrets.toml` with:\n\n"
        'GEMINI_API_KEY = "your_gemini_key_here"\n\n'
        "Or run locally:\n\nexport GEMINI_API_KEY='your_gemini_key_here'  (macOS / Linux)\nsetx GEMINI_API_KEY \"your_gemini_key_here\" (Windows)"
    )
    st.stop()

pipeline = load_rag_pipeline(gemini_key)
if pipeline is None or pipeline.get("qa_chain"_
