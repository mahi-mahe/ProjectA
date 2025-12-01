# app.py
import os
import tempfile
from datetime import datetime

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
# Advanced settings: Gemini key (password input)
# -------------------------
with st.sidebar.expander("Advanced settings", expanded=True):
    st.markdown("**Gemini / Google Generative AI Key**")
    # password style input; not stored to disk - stored only in session_state
    typed_key = st.text_input(
        "Enter Gemini API key (kept in session only)", type="password", help="Provide your Google Gemini API key here (do not hardcode in the script)."
    )
    if st.button("Use this key for this session"):
        # store securely in session (only in-memory for this session)
        st.session_state["GEMINI_API_KEY"] = typed_key
        st.success("Gemini key saved to session (in-memory).")

# Also show a reminder about secrets
st.sidebar.markdown(
    """
**Tip:** For persistent secure storage use `secrets.toml` or your environment, not the code.
See: https://docs.streamlit.io/streamlit-cloud/get-started/deploy/advanced/secrets-management
"""
)

# -------------------------
# In-memory conversation state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Training documents (unchanged)
# -------------------------
TRAINING_DOCUMENTS = [
    # ... paste your training dicts here (TRN001 - TRN008) ...
    # For brevity I assume you keep the same TRAINING_DOCUMENTS list you had.
]

# -------------------------
# Load RAG pipeline (uses Gemini LLM + HuggingFace embeddings)
# -------------------------
@st.cache_resource
def load_rag_pipeline(gemini_api_key: str | None):
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

    # NOTE: you should index your documents into Chroma before querying.
    # For this simplified example we won't persist a disk index; if you want persistence,
    # configure Chroma() with persist_directory="path/to/db" and call .persist() after adding docs.

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


# Grab key from session (preferred)
gemini_key = st.session_state.get("GEMINI_API_KEY", None)

# If user entered a key in the current sidebar box but didn't click the "Use this key" button
# allow immediate usage without saving to session
if not gemini_key and "typed_key" in locals() and typed_key:
    gemini_key = typed_key

if not gemini_key:
    st.warning("Please provide your Gemini API key in the sidebar Advanced settings to enable the RAG functionality.")
    st.info("You can still upload documents which will be processed locally, but the LLM queries require the API key.")
    # Let the user still upload files or prepare docs; but stop before creating the LLM.
    st.stop()

# Build pipeline (cached)
pipeline = load_rag_pipeline(gemini_key)
if pipeline is None or pipeline.get("qa_chain") is None:
    st.error("Failed to initialize the RAG pipeline. Check your Gemini API key and retry.")
    st.stop()

rag_chain = pipeline["qa_chain"]
vectorstore = pipeline["vectorstore"]
embeddings = pipeline["embeddings"]

# -------------------------
# (Optional) Index the in-memory TRAINING_DOCUMENTS into Chroma
# -------------------------
# A simple indexer: convert TRAINING_DOCUMENTS to small documents and add to vectorstore.
# This is necessary so retriever has something to retrieve from.
def index_training_docs(vectorstore, docs):
    from langchain_core.documents import Document  # local import for clarity
    to_index = []
    for d in docs:
        meta = {"title": d.get("title"), "id": d.get("id"), "category": d.get("category"), "level": d.get("level")}
        content = f"{d.get('title')}\n\n{d.get('content')}"
        to_index.append(Document(page_content=content, metadata=meta))
    # Use vectorstore.add_documents or equivalent (Chroma API may differ based on version)
    try:
        vectorstore.add_documents(to_index)
    except Exception:
        # Some Chroma wrappers use .add_texts([...], metadatas=[...])
        texts = [doc.page_content for doc in to_index]
        metadatas = [doc.metadata for doc in to_index]
        vectorstore.add_texts(texts=texts, metadatas=metadatas)

# Index (only once — Chroma in-memory collection will be empty initially)
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
        st.sidebar.success("Document uploaded. (You can index it in Chroma in your ingestion flow.)")
        # In a real ingestion: load text, split, embed and add to Chroma using embeddings.add_texts(...)
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
            # Use the RAG chain to get an answer
            try:
                response = rag_chain.run(prompt)
            except Exception as e:
                response = f"Error calling Gemini LLM: {e}"
            full_response += response
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
