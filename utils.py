import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY")
    vector_store: str = os.getenv("VECTOR_STORE", "faiss")
    embed_model: str = os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "talk-to-your-docs")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    top_k: int = int(os.getenv("TOP_K", 4))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", 0.2))

SETTINGS = Settings()