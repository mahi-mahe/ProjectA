# app.py
import os
import tempfile
from typing import Optional, List, Dict, Any
from datetime import datetime

import streamlit as st

# LangChain / connectors (adapt if your installation uses slightly different names)
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

# For PDF parsing
import PyPDF2

# -------------------------
# Config
# -------------------------
CHROMA_DB_DIR = "./chroma_db"  # persistent directory for vector DB
CHROMA_COLLECTION_NAME = "training_docs_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash"  # change if your account requires another exact string

# -------------------------
# Streamlit UI setup
# -------------------------
st.set_page_config(page_title="RAG Chatbot (Gemini 2.0)", page_icon="🎓", layout="wide")
st.title("Corporate Training RAG Bot — RAG + Gemini 2.0")

st.markdown(
    """
This app indexes uploaded documents into Chroma and answers queries with a Gemini 2.0 LLM.
**Steps**
1. Add your Gemini API key to Streamlit secrets (`GEMINI_API_KEY`) or environment variable.  
2. Upload documents (PDF or TXT).  
3. Click *Index documents* to store embeddings.  
4. Ask questions — answers come from the RAG pipeline.
"""
)

st.sidebar.header("Authentication & Settings")
st.sidebar.markdown(
    "- Provide `GEMINI_API_KEY` in Streamlit secrets or as an environment variable.\n"
    "- Chroma persistence: `./chroma_db` (change in CHROMA_DB_DIR variable)."
)

# -------------------------
# Securely get Gemini key (no UI password input)
# -------------------------
def get_gemini_key() -> Optional[str]:
    try:
        if hasattr(st, "secrets") and st.secrets.get("GEMINI_API_KEY"):
            return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


gemini_key = get_gemini_key()
if not gemini_key:
    st.sidebar.error("Gemini API key missing. Set GEMINI_API_KEY in Streamlit secrets or environment.")
    st.stop()

# -------------------------
# Utility: extract text from uploaded file
# -------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(fileobj=bytes_to_fileobj(file_bytes))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                # continue gracefully if a page fails
                continue
        return "\n".join(texts).strip()
    except Exception as e:
        st.warning(f"PDF text extraction error: {e}")
        return ""


def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return str(file_bytes)


def bytes_to_fileobj(b: bytes):
    # helper to create a file-like object for PyPDF2
    from io import BytesIO

    return BytesIO(b)


# -------------------------
# RAG pipeline builder (cached)
# -------------------------
@st.cache_resource
def build_rag_pipeline(api_key: str) -> Dict[str, Any]:
    """
    Returns: dict with keys: embeddings, vectorstore, retriever, qa_chain
    """
    # embeddings (local HF)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Chroma vectorstore with persistence
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Build LLM wrapper for Gemini 2.0 (try common signatures)
    # Use the model name that works for your account; we'll force GEMINI_MODEL_NAME
    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, api_key=api_key)
    except TypeError:
        # fallback signatures in case of wrapper differences
        try:
            llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, credentials={"api_key": api_key})
        except Exception as e:
            raise RuntimeError(f"Failed to construct ChatGoogleGenerativeAI: {e}")

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return {"embeddings": embeddings, "vectorstore": vectorstore, "retriever": retriever, "qa_chain": qa_chain}


# Initialize pipeline
try:
    pipeline = build_rag_pipeline(gemini_key)
except Exception as e:
    st.error(f"Error initializing RAG pipeline: {e}")
    st.stop()

embeddings = pipeline["embeddings"]
vectorstore = pipeline["vectorstore"]
retriever = pipeline["retriever"]
rag_chain = pipeline["qa_chain"]

# -------------------------
# Upload & index UI
# -------------------------
st.sidebar.header("Upload & Index Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

index_now = st.sidebar.button("Index uploaded documents")

if uploaded_files:
    st.sidebar.write(f"{len(uploaded_files)} file(s) ready to index:")
    for f in uploaded_files:
        st.sidebar.write(f"- {f.name} ({f.type})")

if index_now:
    if not uploaded_files:
        st.sidebar.warning("No files uploaded.")
    else:
        total_chunks = 0
        all_texts = []
        all_metadatas = []

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        progress_bar = st.sidebar.progress(0)
        for idx, up in enumerate(uploaded_files):
            name = up.name
            raw = up.getvalue()
            if name.lower().endswith(".pdf"):
                text = extract_text_from_pdf(raw)
            else:
                text = extract_text_from_txt(raw)
            if not text:
                st.sidebar.warning(f"No text extracted from {name}. Skipping.")
                continue

            # create simple metadata
            metadata = {"source": name, "uploaded_at": datetime.utcnow().isoformat()}

            # split into chunks
            chunks = splitter.split_text(text)
            total_chunks += len(chunks)
            all_texts.extend(chunks)
            all_metadatas.extend([metadata.copy() for _ in chunks])

            progress_bar.progress(int((idx + 1) / len(uploaded_files) * 100))

        if all_texts:
            # add to chroma
            try:
                vectorstore.add_texts(texts=all_texts, metadatas=all_metadatas)
                # persist to disk if Chroma wrapper supports it (langchain_chroma typically does)
                try:
                    vectorstore.persist()
                except Exception:
                    # not all wrappers expose persist(); ignore if absent
                    pass
                st.success(f"Indexed {len(all_texts)} chunks from {len(uploaded_files)} file(s).")
            except Exception as e:
                st.error(f"Failed to index documents: {e}")
        else:
            st.sidebar.info("No text chunks to index after extraction.")

# -------------------------
# Query UI & Chat history
# -------------------------
st.markdown("---")
st.header("Ask the RAG bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Query input
query = st.chat_input("Ask a question about the uploaded documents...")

if query:
    # store user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # run RAG
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Querying RAG pipeline..."):
            try:
                answer = rag_chain.run(query)
            except Exception as e:
                answer = f"Error while running RAG: {e}"
            placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# -------------------------
# Optional: quick retriever debug & preview
# -------------------------
with st.expander("Indexer info & debug"):
    try:
        count = vectorstore._collection.count() if hasattr(vectorstore, "_collection") else "unknown"
    except Exception:
        count = "unknown"
    st.write(f"Chroma collection: {CHROMA_COLLECTION_NAME}")
    st.write(f"Persist directory: {CHROMA_DB_DIR}")
    st.write(f"Indexed documents (estimate): {count}")
