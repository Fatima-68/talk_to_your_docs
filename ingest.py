import os
from typing import List
from embeddings_e5 import E5Embeddings
from utils import SETTINGS

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

# Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

DATA_DIR = os.path.join("data", "uploads")
FAISS_DIR = os.path.join("storage", "faiss_index")

CHUNK_SIZE = 1000  # ~250 tokens
CHUNK_OVERLAP = 150


def load_documents() -> List:
    docs = []
    os.makedirs(DATA_DIR, exist_ok=True)
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            path = os.path.join(root, f)
            if f.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif f.lower().endswith((".txt", ".md")):
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
            elif f.lower().endswith((".doc", ".docx")):
                loader = UnstructuredWordDocumentLoader(path)
                docs.extend(loader.load())
    if not docs:
        raise RuntimeError(f"No documents found in {DATA_DIR}. Upload files first.")
    return docs


def chunk_documents(docs: List):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def build_faiss(chunks: List, embeddings: E5Embeddings):
    os.makedirs(FAISS_DIR, exist_ok=True)
    index = FAISS.from_documents(
        chunks,
        embeddings,
        ids=[c.metadata["id"] for c in chunks]  # ✅ assign your chunk_ids
    )
    index.save_local(FAISS_DIR)
    print(f"Saved FAISS index → {FAISS_DIR}")


def ensure_pinecone_index(embeddings: E5Embeddings):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    name = SETTINGS.pinecone_index
    if name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=name,
            dimension=embeddings.dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=SETTINGS.pinecone_cloud, region=SETTINGS.pinecone_region),
        )
        print(f"Created Pinecone index: {name}")


def upsert_pinecone(chunks: List, embeddings: E5Embeddings):
    ensure_pinecone_index(embeddings)
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=SETTINGS.pinecone_index,
        namespace="default",
        text_key="page_content",
    )
    print(f"Upserted {len(chunks)} chunks into Pinecone index '{SETTINGS.pinecone_index}'.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", choices=["faiss", "pinecone"], default="faiss")
    args = parser.parse_args()

    emb = E5Embeddings(SETTINGS.embed_model)
    documents = load_documents()
    chunks = chunk_documents(documents)

    if args.store == "faiss":
        build_faiss(chunks, emb)
    else:
        upsert_pinecone(chunks, emb)
