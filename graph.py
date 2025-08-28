import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from rag_chain import load_retriever
from embeddings_e5 import E5Embeddings
from utils import SETTINGS

from groq import Groq

# ----- STATE -----
class RAGState(Dict[str, Any]):
    question: str
    context: str
    answer: str

# ----- NODES -----
def retrieve_docs(state: RAGState) -> RAGState:
    emb = E5Embeddings()
    retriever = load_retriever(store=SETTINGS.vector_store, emb=emb)
    docs = retriever.get_relevant_documents(state["question"])
    context = "\n".join([doc.page_content for doc in docs])
    state["context"] = context
    return state

def generate_answer(state: RAGState) -> RAGState:
    client = Groq(api_key=SETTINGS.groq_api_key)
    prompt = f"Answer the question using the context:\n\nContext:\n{state['context']}\n\nQuestion:\n{state['question']}\n\nAnswer:"
    response = client.chat.completions.create(
        model = SETTINGS.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    state["answer"] = response.choices[0].message.content
    return state

# ----- GRAPH -----
def create_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("generate", generate_answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph
