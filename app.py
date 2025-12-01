# app.py
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, List

import streamlit as st

# LangChain / connectors (these module names reflect your previous imports;
# if your installed langchain packages use different names, adapt accordingly)
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
        # st.secrets behaves like a dict; use get for safety
        secret_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
        if secret_key:
            return secret_key
    except Exception:
        # If for any reason st.secrets access fails, fall back to env var
        pass

    # Fallback: environment variable
    return os.environ.get("GEMINI_API_KEY")


# -------------------------
# In-memory conversation state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Training documents (full content included)
# -------------------------
TRAINING_DOCUMENTS = [
    {
        "title": "Python Programming Fundamentals",
        "id": "TRN001",
        "category": "Technical Skills",
        "level": "Beginner",
        "duration": "40 hours",
        "content": """
Python Programming Fundamentals (TRN001) is a comprehensive beginner-level course
designed for employees with little to no programming experience. This 40-hour training
covers essential Python concepts including data types, variables, control structures,
functions, and object-oriented programming basics. Participants will learn to write
clean, efficient Python code through hands-on exercises and real-world projects.

Key topics include: Variables and data types, Conditional statements and loops,
Functions and modules, File handling, Basic OOP concepts, Error handling and debugging.

Prerequisites: None. This course is suitable for complete beginners.
Delivery: Online self-paced with weekly live Q&A sessions.
Instructor: Dr. Sarah Johnson, Senior Python Developer with 10+ years experience.
Certification: Upon completion, participants receive a Python Fundamentals Certificate.
""",
    },
    {
        "title": "Leadership and Management Excellence",
        "id": "TRN002",
        "category": "Soft Skills",
        "level": "Intermediate",
        "duration": "20 hours",
        "content": """
Leadership and Management Excellence (TRN002) is an intermediate-level program
designed for current and aspiring managers. This 20-hour course develops essential
leadership skills including team management, conflict resolution, strategic thinking,
and effective communication. Participants learn proven leadership frameworks and
management techniques used by successful leaders across industries.

Key topics include: Leadership styles and when to use them, Building high-performing teams,
Conflict resolution strategies, Effective delegation, Performance management,
Strategic decision-making, Change management, Emotional intelligence in leadership.

Prerequisites: Minimum 2 years of work experience, preferably in a supervisory role.
Delivery: Hybrid format with in-person workshops and online modules.
Instructor: Prof. Michael Chen, MBA, Executive Leadership Coach.
Benefits: Improved team productivity, Better decision-making skills, Enhanced employee engagement.
""",
    },
    {
        "title": "Data Analysis with Excel and SQL",
        "id": "TRN003",
        "category": "Technical Skills",
        "level": "Intermediate",
        "duration": "30 hours",
        "content": """
Data Analysis with Excel and SQL (TRN003) is an intermediate-level course that teaches
employees how to extract insights from data using industry-standard tools. This 30-hour
training covers advanced Excel functions, pivot tables, data visualization, SQL queries,
and database management. Participants will work with real datasets to solve business problems.

Key topics include: Advanced Excel formulas and functions, Pivot tables and charts,
Data cleaning and preparation, SQL fundamentals (SELECT, JOIN, GROUP BY),
Database design basics, Creating dashboards, Statistical analysis in Excel.

Prerequisites: Basic Excel knowledge (formulas, charts, basic functions).
Delivery: Online instructor-led with hands-on lab exercises.
Instructor: Jane Williams, Data Analytics Manager with 8 years experience.
Tools used: Microsoft Excel 2019/365, MySQL or PostgreSQL.
""",
    },
    {
        "title": "Cybersecurity Awareness Training",
        "id": "TRN004",
        "category": "Compliance",
        "level": "Beginner",
        "duration": "8 hours",
        "content": """
Cybersecurity Awareness Training (TRN004) is a mandatory beginner-level course for
all employees to protect company data and systems. This 8-hour training covers essential
security practices including password management, phishing detection, secure data handling,
and incident reporting. Employees learn to identify and respond to common cyber threats.

Key topics include: Password security best practices, Identifying phishing emails and scams,
Safe browsing and email habits, Data classification and protection, Mobile device security,
Social engineering awareness, Incident reporting procedures, GDPR and data privacy basics.

Prerequisites: None. Required for all employees.
Delivery: Online self-paced with interactive scenarios and quizzes.
Compliance: This training fulfills annual security awareness requirements.
Certificate: Valid for 12 months, renewal required annually.
""",
    },
    {
        "title": "Machine Learning Fundamentals",
        "id": "TRN005",
        "category": "Technical Skills",
        "level": "Advanced",
        "duration": "50 hours",
        "content": """
Machine Learning Fundamentals (TRN005) is an advanced-level course introducing employees
to ML algorithms and practical applications. This intensive 50-hour training covers
supervised and unsupervised learning, model evaluation, feature engineering, and deployment.
Participants build ML models using Python and scikit-learn.

Key topics include: Introduction to ML concepts and terminology, Supervised learning
(regression, classification), Unsupervised learning (clustering, dimensionality reduction),
Model evaluation and validation, Feature engineering and selection, Hyperparameter tuning,
Introduction to neural networks, ML model deployment basics.

Prerequisites: Python programming experience, Basic statistics knowledge.
Delivery: Online instructor-led with weekly assignments and a final project.
Instructor: Dr. Robert Lee, Machine Learning Researcher, PhD in Computer Science.
Tools: Python, Jupyter Notebooks, scikit-learn, pandas, numpy.
""",
    },
    {
        "title": "Effective Communication and Presentation Skills",
        "id": "TRN006",
        "category": "Soft Skills",
        "level": "Beginner",
        "duration": "16 hours",
        "content": """
Effective Communication and Presentation Skills (TRN006) is a beginner-level course
designed to enhance verbal, written, and presentation abilities. This 16-hour training
teaches employees to communicate clearly, write professionally, deliver engaging presentations,
and handle difficult conversations with confidence.

Key topics include: Principles of effective communication, Active listening techniques,
Professional email writing, Presentation structure and design, Public speaking skills,
Body language and non-verbal communication, Handling Q&A sessions, Giving and receiving feedback.

Prerequisites: None. Suitable for all employees.
Delivery: In-person workshops with practice presentations.
Instructor: Emma Davis, Professional Communication Coach, 12+ years experience.
Benefits: Increased confidence, Better stakeholder relationships, Career advancement.
""",
    },
    {
        "title": "Agile and Scrum Methodology",
        "id": "TRN007",
        "category": "Project Management",
        "level": "Intermediate",
        "duration": "24 hours",
        "content": """
Agile and Scrum Methodology (TRN007) is an intermediate-level course teaching modern
project management practices. This 24-hour training covers Agile principles, Scrum framework,
sprint planning, daily standups, retrospectives, and tools for agile teams. Participants
learn to manage projects iteratively and deliver value faster.

Key topics include: Agile manifesto and principles, Scrum roles (Product Owner, Scrum Master, Team),
Sprint planning and execution, Daily standup meetings, Sprint review and retrospectives,
User stories and backlog management, Agile estimation techniques, Kanban vs Scrum.

Prerequisites: Basic project management experience helpful but not required.
Delivery: Hybrid with hands-on Scrum simulations.
Instructor: David Kumar, Certified Scrum Master (CSM), Agile Coach.
Certification: Prepares for Certified ScrumMaster (CSM) exam.
""",
    },
    {
        "title": "Financial Planning and Analysis",
        "id": "TRN008",
        "category": "Finance",
        "level": "Advanced",
        "duration": "35 hours",
        "content": """
Financial Planning and Analysis (TRN008) is an advanced-level course for finance professionals.
This comprehensive 35-hour training covers financial modeling, budgeting, forecasting,
variance analysis, and strategic financial planning. Participants learn to create complex
financial models and provide data-driven insights to senior management.

Key topics include: Financial statement analysis, Building financial models in Excel,
Budgeting processes and best practices, Financial forecasting techniques, Variance analysis,
Cash flow management, Investment appraisal, Financial ratios and KPIs, Scenario analysis.

Prerequisites: Accounting fundamentals, Advanced Excel skills.
Delivery: Online instructor-led with case studies from real companies.
Instructor: CFA John Smith, Director of FP&A with 15 years experience.
Target audience: Finance analysts, FP&A professionals, Controllers.
""",
    },
]

# -------------------------
# Load RAG pipeline (uses Gemini LLM + HuggingFace embeddings)
# -------------------------
@st.cache_resource
def load_rag_pipeline(gemini_api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Build RAG pipeline. Requires a Gemini API key (api passed directly to the LLM).
    Embeddings use a local Hugging Face model (sentence-transformers/all-MiniLM-L6-v2).
    Vector DB: Chroma (in-memory/default directory).
    Returns a dict with keys: qa_chain, vectorstore, embeddings
    """
    if not gemini_api_key:
        return None

    # Create embeddings (Hugging Face sentence-transformers)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma vectorstore
    # To persist to disk, provide persist_directory="/path/to/db"
    vectorstore = Chroma(embedding_function=embeddings, collection_name="my_rag_collection")

    retriever = vectorstore.as_retriever()

    # Create Gemini LLM wrapper (langchain_google_genai)
    # The ChatGoogleGenerativeAI here expects an api_key parameter in this wrapper
    try:
        llm = ChatGoogleGenerativeAI(api_key=gemini_api_key)
    except TypeError:
        # Some versions may expect 'credentials' or different init signature; try fallback
        llm = ChatGoogleGenerativeAI(gemini_api_key)

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
        "Local setup example:\n\n"
        "Create `.streamlit/secrets.toml` with:\n\n"
        'GEMINI_API_KEY = "your_gemini_key_here"\n\n'
        "Or set an env var and run:\n\n"
        "macOS / Linux:\nexport GEMINI_API_KEY='your_gemini_key_here'\nstreamlit run app.py\n\n"
        "Windows PowerShell:\n$env:GEMINI_API_KEY='your_gemini_key_here'\nstreamlit run app.py"
    )
    st.stop()

pipeline = load_rag_pipeline(gemini_key)
if pipeline is None or pipeline.get("qa_chain") is None:
    st.error("Failed to initialize the RAG pipeline. Check your Gemini API key and retry.")
    st.stop()

rag_chain = pipeline["qa_chain"]
vectorstore = pipeline["vectorstore"]
embeddings = pipeline["embeddings"]

# -------------------------
# Index training docs into Chroma (only if docs exist)
# -------------------------
def index_training_docs(vectorstore: Any, docs: List[Dict[str, Any]]):
    """
    Index the list of docs into the vectorstore. Handles different Chroma wrapper apis.
    """
    try:
        # Use langchain_core Document if available
        from langchain_core.documents import Document

        to_index = []
        for d in docs:
            meta = {
                "title": d.get("title"),
                "id": d.get("id"),
                "category": d.get("category"),
                "level": d.get("level"),
            }
            content = f"{d.get('title')}\n\n{d.get('content')}"
            to_index.append(Document(page_content=content, metadata=meta))
        # Try add_documents if available
        try:
            vectorstore.add_documents(to_index)
            return
        except Exception:
            # fallback to add_texts
            pass
    except Exception:
        # If langchain_core.Document not available, fallback to text + metadata lists
        to_index = []
        for d in docs:
            meta = {
                "title": d.get("title"),
                "id": d.get("id"),
                "category": d.get("category"),
                "level": d.get("level"),
            }
            content = f"{d.get('title')}\n\n{d.get('content')}"
            to_index.append((content, meta))

    # Generic fallback: attempt vectorstore.add_texts
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
        # Save temporarily and give user success message; ingestion pipeline can be added later
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
                # Some wrappers return objects; ensure string
                if isinstance(response, (list, dict)):
                    response_text = str(response)
                else:
                    response_text = response
            except Exception as e:
                response_text = f"Error calling Gemini LLM: {e}"

            full_response += response_text
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
