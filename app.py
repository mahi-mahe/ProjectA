# app.py
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, List

import streamlit as st

# LangChain / connectors (adapt imports if your environment uses different package names)
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
    initial_sidebar_state="expanded",
)
st.title("RAG Chatbot with Streamlit (Gemini 2.0)")

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
      1. st.secrets["GEMINI_API_KEY"]
      2. os.environ["GEMINI_API_KEY"]
      3. None if not found
    """
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
        if secret_key:
            return secret_key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


# -------------------------
# In-memory conversation state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Training documents (paste your TRN001-TRN008 here)
# -------------------------
TRAINING_DOCUMENTS = [
    # (same training docs content as before)
]

# -------------------------
# Load RAG pipeline (forces Gemini 2.0)
# -------------------------
@st.cache_resource
def load_rag_pipeline(gemini_api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Build RAG pipeline using Gemini 2.0 only.
    Tries a few constructor signatures but always uses model='gemini-2.0'.
    """
    if not gemini_api_key:
        return None

    # Embeddings (local HF)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Chroma vectorstore (in-memory by default). Add persist_directory for disk persistence.
    vectorstore = Chroma(embedding_function=embeddings, collection_name="my_rag_collection")
    retriever = vectorstore.as_retriever()

    model_name = "gemini-2.0"
    llm = None
    errors = []

    # Try common constructor patterns while forcing model_name
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, api_key=gemini_api_key)
    except Exception as e:
        errors.append(f"model={model_name}, api_key -> {type(e).__name__}: {e}")

    if llm is None:
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, credentials={"api_key": gemini_api_key})
        except Exception as e:
            errors.append(f"model={model_name}, credentials(api_key) -> {type(e).__name__}: {e}")

    if llm is None:
        try:
            # Some wrappers expect `credentials` with nested structure or different key names
            llm = ChatGoogleGenerativeAI(model=model_name, credentials={"key": gemini_api_key})
        except Exception as e:
            errors.append(f"model={model_name}, credentials(key) -> {type(e).__name__}: {e}")

    if llm is None:
        try:
            # last-resort: pass dictionary-like single arg (some versions may accept this)
            llm = ChatGoogleGenerativeAI({"model": model_name, "api_key": gemini_api_key})
        except Exception as e:
            errors.append(f"fallback dict constructor -> {type(e).__name__}: {e}")

    if llm is None:
        # None of the tried signatures worked — raise a helpful error with attempts
        raise RuntimeError(
            "Could not construct ChatGoogleGenerativeAI with model='gemini-2.0'. "
            "Tried several constructor patterns. Errors:\n\n" + "\n".join(errors)
            + "\n\nAction: verify your langchain_google_genai version and its constructor signature. "
            "Run `pip show langchain-google-genai` and share the version if you want me to adapt the constructor."
        )

    # Build the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
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
        "Local example: create `.streamlit/secrets.toml` with:\n\nGEMINI_API_KEY = \"your_gemini_key_here\"\n\n"
        "Or set an env var and run streamlit: \n\n"
        "macOS / Linux:\nexport GEMINI_API_KEY='your_gemini_key_here'\nstreamlit run app.py\n\n"
        "Windows PowerShell:\n$env:GEMINI_API_KEY='your_gemini_key_here'\nstreamlit run app.py"
    )
    st.stop()

pipeline = load_rag_pipeline(gemini_key)
if pipeline is None or pipeline.get("qa_chain") is None:
    st.error("Failed to initialize the RAG pipeline. Check your Gemini API key and langchain-google-genai installation.")
    st.stop()

rag_chain = pipeline["qa_chain"]
vectorstore = pipeline["vectorstore"]
embeddings = pipeline["embeddings"]

# -------------------------
# Index training docs into Chroma (if present)
# -------------------------
def index_training_docs(vectorstore: Any, docs: List[Dict[str, Any]]):
    try:
        from langchain_core.documents import Document

        to_index = []
        for d in docs:
            meta = {"title": d.get("title"), "id": d.get("id"), "category": d.get("category"), "level": d.get("level")}
            content = f"{d.get('title')}\n\n{d.get('content')}"
            to_index.append(Document(page_content=content, metadata=meta))
        try:
            vectorstore.add_documents(to_index)
            return
        except Exception:
            pass
    except Exception:
        to_index = []
        for d in docs:
            meta = {"title": d.get("title"), "id": d.get("id"), "category": d.get("category"), "level": d.get("level")}
            content = f"{d.get('title')}\n\n{d.get('content')}"
            to_index.append((content, meta))

    try:
        texts = [t[0] if isinstance(t, tuple) else t.page_content for t in to_index]
        metadatas = [t[1] if isinstance(t, tuple) else t.metadata for t in to_index]
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
    except Exception as e:
        st.warning(f"Indexing documents failed: {e}")


if TRAINING_DOCUMENTS:
    index_training_docs(vectorstore, TRAINING_DOCUMENTS)

# -------------------------
# File uploader (sidebar)
# -------------------------
def handle_file_upload():
    uploaded_file = st.sidebar.file_uploader("Upload a document (txt, pdf)", type=["txt", "pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        st.sidebar.success("Document uploaded. You can implement ingestion to index it into Chroma.")
        try:
            os.remove(tmp_file_path)
        except Exception:
            pass


handle_file_upload()

# -------------------------
# Display chat history and accept user input
# -------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your training bot anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):
            try:
                response = rag_chain.run(prompt)
                response_text = str(response) if not isinstance(response, str) else response
            except Exception as e:
                response_text = f"Error calling Gemini 2.0 LLM: {e}"

            full_response += response_text
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
