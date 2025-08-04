"""
RAG system classes and functionality for different retrieval strategies.
"""

import time
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
import streamlit as st
import numpy as np
import google.generativeai as genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import GEMINI_API_KEY


@dataclass
class RAGResult:
    """Structure to hold RAG query results"""
    method: str
    query: str
    retrieved_chunks: List[Dict]
    response: str
    retrieval_time: float
    generation_time: float
    total_time: float
    similarity_scores: List[float]
    sources: List[str]
    query_variations: Optional[List[str]] = None


class QdrantRAGSystem:
    """Advanced RAG system with multiple retrieval strategies"""

    def __init__(self, qdrant_url: str, qdrant_api_key: str, gemini_api_key: str = GEMINI_API_KEY):
        try:
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

            # Initialize embedding model with proper device handling
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Try different embedding models in order of preference
            embedding_models = [
                'all-MiniLM-L6-v2',
                'sentence-transformers/all-MiniLM-L6-v2',
                'paraphrase-MiniLM-L6-v2'
            ]

            self.embedding_model = None
            for model_name in embedding_models:
                try:
                    self.embedding_model = SentenceTransformer(model_name, device=device)
                    st.info(f"Successfully loaded embedding model: {model_name}")
                    break
                except Exception as e:
                    st.warning(f"Failed to load {model_name}: {str(e)}")
                    continue

            if self.embedding_model is None:
                raise Exception("Failed to load any embedding model")

            # Initialize other components
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            self.tfidf_fitted = False
            self.document_texts = []

        except Exception as e:
            raise Exception(f"Failed to initialize RAG system: {str(e)}")

    def fit_tfidf(self, collection_name: str):
        """Fit TF-IDF vectorizer on collection documents"""
        try:
            # Get all documents from collection for TF-IDF fitting
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True
            )

            self.document_texts = [point.payload.get('text', '') for point in scroll_result[0]]
            if self.document_texts:
                self.tfidf_vectorizer.fit(self.document_texts)
                self.tfidf_fitted = True
                st.success(f"TF-IDF fitted on {len(self.document_texts)} documents")
            else:
                st.warning("No documents found in collection for TF-IDF fitting")
        except Exception as e:
            st.error(f"Error fitting TF-IDF: {str(e)}")

    async def simple_rag(self, query: str, collection_name: str, top_k: int = 5) -> RAGResult:
        """Simple RAG: Direct vector similarity search"""
        start_time = time.time()

        try:
            # Generate query embedding with error handling
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=False).tolist()
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            raise

        # Vector search
        retrieval_start = time.time()
        try:
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
        except Exception as e:
            st.error(f"Error searching Qdrant: {str(e)}")
            raise

        retrieval_time = time.time() - retrieval_start

        # Prepare context
        retrieved_chunks = []
        sources = []
        similarity_scores = []

        for result in search_result:
            chunk_data = {
                'text': result.payload.get('text', ''),
                'source': result.payload.get('source', 'Unknown'),
                'score': result.score
            }
            retrieved_chunks.append(chunk_data)
            sources.append(chunk_data['source'])
            similarity_scores.append(result.score)

        # Generate response
        generation_start = time.time()
        context = "\n\n".join([f"Source: {chunk['source']}\n{chunk['text']}" for chunk in retrieved_chunks])
        response = await self.generate_response(query, context, "Simple RAG")
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        return RAGResult(
            method="Simple RAG",
            query=query,
            retrieved_chunks=retrieved_chunks,
            response=response,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            similarity_scores=similarity_scores,
            sources=list(set(sources))
        )

    async def hybrid_rag(self, query: str, collection_name: str, top_k: int = 5,
                         vector_weight: float = 0.7) -> RAGResult:
        """Hybrid RAG: Combines vector and keyword search"""
        start_time = time.time()

        if not self.tfidf_fitted:
            self.fit_tfidf(collection_name)

        retrieval_start = time.time()

        # Vector search
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False).tolist()
        vector_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k * 2,  # Get more candidates for reranking
            with_payload=True
        )

        # Keyword search using TF-IDF
        if self.tfidf_fitted and self.document_texts:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            doc_tfidf = self.tfidf_vectorizer.transform(self.document_texts)
            tfidf_scores = cosine_similarity(query_tfidf, doc_tfidf).flatten()
        else:
            tfidf_scores = np.zeros(len(vector_results))

        # Combine scores
        combined_results = []
        for i, result in enumerate(vector_results):
            vector_score = result.score
            keyword_score = tfidf_scores[min(i, len(tfidf_scores) - 1)]
            combined_score = vector_weight * vector_score + (1 - vector_weight) * keyword_score

            combined_results.append({
                'result': result,
                'combined_score': combined_score,
                'vector_score': vector_score,
                'keyword_score': keyword_score
            })

        # Sort by combined score and take top_k
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        top_results = combined_results[:top_k]

        retrieval_time = time.time() - retrieval_start

        # Prepare context
        retrieved_chunks = []
        sources = []
        similarity_scores = []

        for item in top_results:
            result = item['result']
            chunk_data = {
                'text': result.payload.get('text', ''),
                'source': result.payload.get('source', 'Unknown'),
                'score': item['combined_score'],
                'vector_score': item['vector_score'],
                'keyword_score': item['keyword_score']
            }
            retrieved_chunks.append(chunk_data)
            sources.append(chunk_data['source'])
            similarity_scores.append(item['combined_score'])

        # Generate response
        generation_start = time.time()
        context = "\n\n".join([f"Source: {chunk['source']}\n{chunk['text']}" for chunk in retrieved_chunks])
        response = await self.generate_response(query, context, "Hybrid RAG")
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        return RAGResult(
            method="Hybrid RAG",
            query=query,
            retrieved_chunks=retrieved_chunks,
            response=response,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            similarity_scores=similarity_scores,
            sources=list(set(sources))
        )

    async def multi_query_rag(self, query: str, collection_name: str, top_k: int = 5) -> RAGResult:
        """Multi-Query RAG: Generates multiple query variations for better retrieval"""
        start_time = time.time()

        # Generate query variations
        query_variations = await self.generate_query_variations(query)
        query_variations.append(query)  # Include original query

        retrieval_start = time.time()

        # Search with all query variations
        all_results = {}
        for variation in query_variations:
            variation_embedding = self.embedding_model.encode(variation, convert_to_tensor=False).tolist()
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=variation_embedding,
                limit=top_k,
                with_payload=True
            )

            for result in search_result:
                doc_id = result.id
                if doc_id not in all_results or result.score > all_results[doc_id]['score']:
                    all_results[doc_id] = {
                        'result': result,
                        'score': result.score,
                        'query_variation': variation
                    }

        # Sort by score and take top_k
        sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)[:top_k]
        retrieval_time = time.time() - retrieval_start

        # Prepare context
        retrieved_chunks = []
        sources = []
        similarity_scores = []

        for item in sorted_results:
            result = item['result']
            chunk_data = {
                'text': result.payload.get('text', ''),
                'source': result.payload.get('source', 'Unknown'),
                'score': result.score,
                'matched_query': item['query_variation']
            }
            retrieved_chunks.append(chunk_data)
            sources.append(chunk_data['source'])
            similarity_scores.append(result.score)

        # Generate response
        generation_start = time.time()
        context = "\n\n".join([f"Source: {chunk['source']}\n{chunk['text']}" for chunk in retrieved_chunks])
        response = await self.generate_response(query, context, "Multi-Query RAG")
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        return RAGResult(
            method="Multi-Query RAG",
            query=query,
            retrieved_chunks=retrieved_chunks,
            response=response,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            similarity_scores=similarity_scores,
            query_variations=query_variations,
            sources=list(set(sources))
        )

    async def generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations using Gemini"""
        try:
            prompt = f"""
            Generate 3 different variations of the following query that would help retrieve relevant information:

            Original query: {query}

            Variations should:
            1. Use different wording but maintain the same intent
            2. Include synonyms and related terms
            3. Vary in specificity (more general or more specific)

            Return only the 3 variations, one per line, without numbering or additional text.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None, self.gemini_model.generate_content, prompt
            )

            variations = response.text.strip().split('\n')
            return [v.strip() for v in variations if v.strip()]
        except Exception as e:
            st.warning(f"Could not generate query variations: {str(e)}")
            return []

    async def generate_response(self, query: str, context: str, method: str) -> str:
        """Generate response using retrieved context with Gemini"""
        try:
            prompt = f"""
            Based on the following context, answer the user's question. Be specific and cite sources when possible.

            Context:
            {context}

            Question: {query}

            Method used: {method}

            Answer:
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None, self.gemini_model.generate_content, prompt
            )

            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
