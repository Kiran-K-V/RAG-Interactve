"""
Custom embedding classes for the RAG Interactive application.
"""

from typing import List
from llama_index.core.base.embeddings.base import BaseEmbedding
from sentence_transformers import SentenceTransformer
from pydantic import PrivateAttr
import streamlit as st
from config import DEFAULT_EMBEDDING_MODEL


class LocalSentenceTransformerEmbedding(BaseEmbedding):
    """Local sentence transformer embedding model for LlamaIndex integration."""

    _model: SentenceTransformer = PrivateAttr()

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, **kwargs):
        super().__init__(**kwargs)
        self._model = SentenceTransformer(model_name)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts).tolist()

    def _get_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(queries)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)


@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model."""
    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
