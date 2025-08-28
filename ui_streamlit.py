import streamlit as st
import os, json
from graph import create_rag_graph
from ingest import chunk_documents, build_faiss, upsert_pinecone
from embeddings_e5 import E5Embeddings
from utils import SETTINGS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader

# Paths
UPLOAD_DIR = "data/uploads"
META_FILE = "storage/files.json"
FAISS_DIR = "storage/faiss_index"
os.makedirs("storage", exist_ok=True)

# --- Page Config ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# --- Helper functions ---
def load_metadata():
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {}

def save_metadata(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

def delete_file_from_vector(filename, emb):
    """Delete file + chunks from vector DB"""
    meta = load_metadata()
    if filename not in meta:
        return

    ids_to_delete = meta[filename]

    if SETTINGS.vector_store == "faiss":
        from langchain_community.vectorstores import FAISS
        index = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)
        index.delete(ids=ids_to_delete)  # ✅ simple delete
        index.save_local(FAISS_DIR)

    else:  # Pinecone case
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(SETTINGS.pinecone_index)
        index.delete(ids=ids_to_delete, namespace="default")

    # remove local file + metadata
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        os.remove(path)

    del meta[filename]
    save_metadata(meta)


# --- Title ---
st.markdown('<h1 style="text-align:center; color:#4CAF50;">📚 RAG Chatbot with Groq + LangGraph</h1>', unsafe_allow_html=True)

# --- Upload section ---
st.subheader("📂 Upload Documents")
uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)

emb = E5Embeddings(SETTINGS.embed_model)
meta = load_metadata()

if uploaded_files:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Detect loader
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith((".doc", ".docx")):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        docs = loader.load()
        chunks = chunk_documents(docs)

        # Assign IDs to chunks
        chunk_ids = [f"{uploaded_file.name}_{i}" for i in range(len(chunks))]
        for cid, c in zip(chunk_ids, chunks):
            c.metadata["id"] = cid

        if SETTINGS.vector_store == "faiss":
            build_faiss(chunks, emb)
        else:
            upsert_pinecone(chunks, emb)

        # Update metadata
        meta[uploaded_file.name] = chunk_ids
        save_metadata(meta)

    st.success("✅ Files uploaded and stored in vector database!")


# --- Show Uploaded Files ---
st.subheader("📑 Uploaded Files")
if meta:
    for filename in list(meta.keys()):
        col1, col2 = st.columns([4, 1])
        col1.write(filename)
        if col2.button("🗑 Delete", key=filename):
            delete_file_from_vector(filename, emb)
            st.rerun()
else:
    st.info("No files uploaded yet.")


# --- Chat section ---
st.subheader("💬 Ask Questions")
question = st.text_input("Type your question:")

if st.button("🚀 Get Answer") and question.strip():
    with st.spinner("Thinking... 🤔"):
        graph = create_rag_graph()
        app = graph.compile()
        initial_state = {"question": question}
        final_state = app.invoke(initial_state)

    # Display results
    st.markdown(f"<div style='background:#e1f5fe;padding:12px;border-radius:10px;color:black;'><b>You:</b> {question}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:#f1f8e9;padding:12px;border-radius:10px;color:black;'><b>Bot:</b> {final_state['answer']}</div>", unsafe_allow_html=True)

    with st.expander("📄 Context Used"):
        st.write(final_state["context"])
