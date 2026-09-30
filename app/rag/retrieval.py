"""Hybrid (BM25 + vector) retrieval with cross-encoder reranking."""

import re

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def normalize_for_lexical_search(text: str) -> str:
    """Normalized representation used by the lexical (BM25) retriever."""
    text = text.lower()
    text = re.sub(r"[.,!?;:\"'()\[\]{}<>/\\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_hybrid_retriever(
    store: FAISS,
    chunks: list[Document],
    reranker: BaseCrossEncoder,
    *,
    vector_k: int,
    lexical_k: int,
    final_k: int,
) -> BaseRetriever:
    vector = store.as_retriever(search_kwargs={"k": vector_k})
    lexical = BM25Retriever.from_documents(chunks, preprocess_func=normalize_for_lexical_search)
    lexical.k = lexical_k
    ensemble = EnsembleRetriever(retrievers=[lexical, vector])
    return ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=reranker, top_n=final_k),
        base_retriever=ensemble,
    )
