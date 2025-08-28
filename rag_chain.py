# file: rag_chain.py
import os
import datetime
from typing import List, Tuple
from embeddings_e5 import E5Embeddings
from utils import SETTINGS

from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA

from groq import Groq

# Path for FAISS index
FAISS_DIR = os.path.join("storage", "faiss_index")


def load_retriever(store: str, emb: E5Embeddings):
    """Load vector store retriever (FAISS or Pinecone)."""
    if store == "faiss":
        vs = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)
    elif store == "pinecone":
        vs = PineconeVectorStore.from_existing_index(
            index_name=SETTINGS.PINECONE_INDEX_NAME,
            embedding=emb
        )
    else:
        raise ValueError(f"Unknown store type: {store}")
    return vs.as_retriever(search_kwargs={"k": 5})


def build_rag_chain(store: str = "faiss"):
    """Builds the Retrieval-Augmented Generation (RAG) chain."""
    # Initialize embeddings
    embeddings = E5Embeddings(model_name=SETTINGS.EMBEDDING_MODEL)

    # Load retriever
    retriever = load_retriever(store, embeddings)

    # Initialize LLM client
    client = Groq(api_key=SETTINGS.GROQ_API_KEY)

    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
        Using the retrieved documents as primary context, answer the question. If the documents are insufficient, use general knowledge, but clearly indicate which info comes from the documents.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    # Define QA chain using LangChain's RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=client,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa_chain


