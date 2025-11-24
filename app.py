import streamlit as st
import os
import json

from backend.pdf_loader import load_pdf_files
from backend.text_splitter import split_text
from backend.embeddings import EmbeddingModel
from backend.vector_store import FAISSStore
from backend.rag_pipeline import RAGPipeline
import tempfile

# Writable directory for chat history (Streamlit Cloud allows writing here)
TEMP_HISTORY_DIR = os.path.join(tempfile.gettempdir(), "chat_history")
os.makedirs(TEMP_HISTORY_DIR, exist_ok=True)

# Path to temporary history file
HISTORY_FILE = os.path.join(TEMP_HISTORY_DIR, "history.json")

def load_history():
    """Safely load conversation history."""
    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_history(history):
    """Save chat history to JSON."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)



# ============================================
# Streamlit Page Settings
# ============================================

st.set_page_config(
    page_title="AI-Powered PDF Analyst",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
    }
    .subtitle {
        font-size: 15px;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)



# ============================================
# Page Header
# ============================================

st.markdown('<div class="main-title">🧠 AI-Powered PDF Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Chat with your PDFs • Summaries • Quizzes • Explanations</div>', unsafe_allow_html=True)



# ============================================
# Sidebar — Only Document Controls
# ============================================

with st.sidebar:
    st.header("📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    build_db = st.button("🔄 Build / Rebuild Knowledge Base", use_container_width=True)

    st.markdown("---")
    st.header("⚙ Actions")
    if st.button("🗑 Clear Chat History", use_container_width=True):
        save_history([])
        st.session_state.history = []
        st.success("History cleared. Refresh page.")



# ============================================
# Global State
# ============================================

if "history" not in st.session_state:
    st.session_state.history = load_history()

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

# Initialize embedding + vector store
embed_model = EmbeddingModel()
vector_store = FAISSStore()
rag = None



# ============================================
# BUILD KNOWLEDGE BASE
# ============================================

if build_db:

    if not uploaded_files:
        st.warning("Upload at least one PDF first.")
    else:
        with st.spinner("Processing PDFs..."):

            # Save PDFs to disk
            pdf_paths = []

            os.makedirs("data/uploaded_pdfs", exist_ok=True)

            for file in uploaded_files:
                save_path = f"data/uploaded_pdfs/{file.name}"
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
                pdf_paths.append(save_path)

            # Load → Split → Embed → Index
            pages = load_pdf_files(pdf_paths)
            chunks = split_text(pages)
            texts = [c["content"] for c in chunks]
            embeddings = embed_model.embed_text(texts)

            vector_store.create_new_index()
            vector_store.add_embeddings(embeddings, chunks)
            vector_store.save_index()

            st.session_state.vector_ready = True

        st.success("Knowledge base built successfully!")



# ============================================
# Main UI — Tabs
# ============================================

chat_tab, summary_tab, quiz_tab, explain_tab = st.tabs(
    ["💬 Chat", "📄 Summary", "📝 Quiz", "💡 Explain"]
)

# ============================================================
# 1️⃣ CHAT TAB — TRUE CHATGPT STYLE (input stays at bottom)
# ============================================================

with chat_tab:

    st.subheader("💬 Chat with your PDFs")

    # Load FAISS on app startup
    if vector_store.load_index():
        rag = RAGPipeline(vector_store, embed_model)
        st.session_state.vector_ready = True

    # ---- FIRST show all messages (old → new) ----
    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ---- THEN put chat_input at real bottom ----
    user_input = st.chat_input("Ask something from the PDFs...")

    # ---- AFTER user submits, show new messages ----
    if user_input:
        if not st.session_state.vector_ready:
            st.error("Build the knowledge base first.")
        else:
            # Display user question
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(user_input)

            # Save
            st.session_state.history.append({"role": "user", "content": user_input})
            save_history(st.session_state.history)

            # Generate answer
            with chat_box:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        context, _ = rag.retrieve_context(user_input)
                        answer = rag.generate_answer(user_input, context)
                        st.markdown(answer)

            # Save
            st.session_state.history.append({"role": "assistant", "content": answer})
            save_history(st.session_state.history)


# ============================================================
# 2️⃣ SUMMARY TAB
# ============================================================

with summary_tab:

    st.subheader("📄 Generate Summary")

    if st.button("Generate Summary"):
        if not st.session_state.vector_ready:
            st.error("Build knowledge base first.")
        else:
            with st.spinner("Summarizing..."):
                context, _ = rag.retrieve_context("summary", top_k=8)
                summary = rag.generate_summary(context)
                st.write(summary)



# ============================================================
# 3️⃣ QUIZ TAB
# ============================================================

with quiz_tab:

    st.subheader("📝 Create Quiz")

    q_type = st.selectbox("Quiz Type", ["mcq", "short answer"])

    if st.button("Generate Quiz"):
        if not st.session_state.vector_ready:
            st.error("Build knowledge base first.")
        else:
            with st.spinner("Generating quiz..."):
                context, _ = rag.retrieve_context("quiz", top_k=8)
                quiz = rag.generate_quiz(context, q_type)
                st.write(quiz)



# ============================================================
# 4️⃣ EXPLAIN TAB
# ============================================================

with explain_tab:

    st.subheader("💡 Explain in Different Styles")

    style = st.selectbox("Explanation Style", ["simple", "expert", "examples"])
    topic = st.text_input("Enter the topic you want explained:")

    if st.button("Explain"):
        if not st.session_state.vector_ready:
            st.error("Build knowledge base first.")
        else:
            with st.spinner("Explaining..."):
                context, _ = rag.retrieve_context(topic, top_k=8)
                explanation = rag.explain_topic(context, style)
                st.write(explanation)
