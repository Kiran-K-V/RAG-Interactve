import streamlit as st
import asyncio
import time
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Core imports for RAG functionality
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure page
st.set_page_config(
    page_title="RAG Systems Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


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


class QdrantRAGSystem:
    """Advanced RAG system with multiple retrieval strategies"""

    def __init__(self, qdrant_url: str, qdrant_api_key: str, gemini_api_key: str):
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


def create_performance_charts(results: List[RAGResult]):
    """Create performance comparison charts"""

    # Prepare data
    methods = [r.method for r in results]
    retrieval_times = [r.retrieval_time for r in results]
    generation_times = [r.generation_time for r in results]
    total_times = [r.total_time for r in results]
    avg_scores = [np.mean(r.similarity_scores) if r.similarity_scores else 0 for r in results]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Response Times', 'Average Similarity Scores',
                        'Time Breakdown', 'Retrieved Sources Count'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Response times comparison
    fig.add_trace(
        go.Bar(name='Total Time', x=methods, y=total_times, marker_color='lightblue'),
        row=1, col=1
    )

    # Average similarity scores
    fig.add_trace(
        go.Bar(name='Avg Score', x=methods, y=avg_scores, marker_color='lightgreen'),
        row=1, col=2
    )

    # Time breakdown
    fig.add_trace(
        go.Bar(name='Retrieval', x=methods, y=retrieval_times, marker_color='orange'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(name='Generation', x=methods, y=generation_times, marker_color='red'),
        row=2, col=1
    )

    # Sources count
    sources_count = [len(r.sources) for r in results]
    fig.add_trace(
        go.Bar(name='Sources', x=methods, y=sources_count, marker_color='purple'),
        row=2, col=2
    )

    fig.update_layout(height=800, showlegend=True, title_text="RAG Methods Performance Comparison")
    return fig


def display_results(results: List[RAGResult]):
    """Display detailed results for each RAG method"""

    for result in results:
        with st.expander(f"📊 {result.method} Results", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Time", f"{result.total_time:.3f}s")
                st.metric("Retrieval Time", f"{result.retrieval_time:.3f}s")

            with col2:
                st.metric("Generation Time", f"{result.generation_time:.3f}s")
                st.metric("Chunks Retrieved", len(result.retrieved_chunks))

            with col3:
                avg_score = np.mean(result.similarity_scores) if result.similarity_scores else 0
                st.metric("Avg Similarity", f"{avg_score:.3f}")
                st.metric("Unique Sources", len(result.sources))

            st.subheader("Generated Response")
            st.write(result.response)

            st.subheader("Retrieved Chunks")
            for i, chunk in enumerate(result.retrieved_chunks):
                with st.container():
                    st.write(f"**Chunk {i + 1}** (Score: {chunk['score']:.3f})")
                    st.write(f"*Source: {chunk['source']}*")
                    st.write(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])

                    # Show additional info for hybrid RAG
                    if 'vector_score' in chunk:
                        st.caption(f"Vector: {chunk['vector_score']:.3f}, Keyword: {chunk['keyword_score']:.3f}")

                    # Show matched query for multi-query RAG
                    if 'matched_query' in chunk:
                        st.caption(f"Matched query: {chunk['matched_query']}")

                    st.divider()


async def main():
    st.title("🔍 RAG Systems Comparison Dashboard")
    st.markdown("Compare Simple RAG, Hybrid RAG, and Multi-Query RAG performance on your Qdrant collections")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Qdrant settings
        st.subheader("Qdrant Settings")
        qdrant_url = st.text_input("Qdrant URL", value="https://your-cluster.qdrant.io")
        qdrant_api_key = st.text_input("Qdrant API Key", type="password")

        # Gemini settings
        st.subheader("Gemini Settings")
        gemini_api_key = st.text_input("Gemini API Key", type="password")

        # Collection selection
        st.subheader("Collection Settings")
        collection_name = st.text_input("Collection Name", value="your_collection")

        # RAG parameters
        st.subheader("RAG Parameters")
        top_k = st.slider("Top K Results", 1, 10, 5)
        vector_weight = st.slider("Vector Weight (Hybrid RAG)", 0.0, 1.0, 0.7, 0.1)

        # Method selection
        st.subheader("Methods to Compare")
        run_simple = st.checkbox("Simple RAG", value=True)
        run_hybrid = st.checkbox("Hybrid RAG", value=True)
        run_multi_query = st.checkbox("Multi-Query RAG", value=True)

    # Main interface
    if not all([qdrant_url, qdrant_api_key, gemini_api_key, collection_name]):
        st.warning("Please provide all required configuration parameters in the sidebar.")
        return

    # Initialize RAG system
    try:
        rag_system = QdrantRAGSystem(qdrant_url, qdrant_api_key, gemini_api_key)
        st.success("RAG system initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return

    # Query input
    query = st.text_area("Enter your query:",
                         placeholder="What would you like to know about your documents?",
                         height=100)

    if st.button("🚀 Run Comparison", type="primary"):
        if not query:
            st.warning("Please enter a query.")
            return

        st.info("Running RAG comparison... This may take a moment.")

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        methods_to_run = []
        if run_simple:
            methods_to_run.append(("Simple RAG", rag_system.simple_rag))
        if run_hybrid:
            methods_to_run.append(("Hybrid RAG", rag_system.hybrid_rag))
        if run_multi_query:
            methods_to_run.append(("Multi-Query RAG", rag_system.multi_query_rag))

        for i, (method_name, method_func) in enumerate(methods_to_run):
            status_text.text(f"Running {method_name}...")

            try:
                if method_name == "Hybrid RAG":
                    result = await method_func(query, collection_name, top_k, vector_weight)
                else:
                    result = await method_func(query, collection_name, top_k)
                results.append(result)
                st.success(f"✅ {method_name} completed in {result.total_time:.3f}s")
            except Exception as e:
                st.error(f"❌ {method_name} failed: {str(e)}")

            progress_bar.progress((i + 1) / len(methods_to_run))

        status_text.text("Comparison complete!")

        if results:
            # Performance charts
            st.header("📈 Performance Analysis")
            fig = create_performance_charts(results)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed results
            st.header("📋 Detailed Results")
            display_results(results)

            # Summary table
            st.header("📊 Summary Table")
            summary_data = []
            for result in results:
                summary_data.append({
                    'Method': result.method,
                    'Total Time (s)': f"{result.total_time:.3f}",
                    'Retrieval Time (s)': f"{result.retrieval_time:.3f}",
                    'Generation Time (s)': f"{result.generation_time:.3f}",
                    'Avg Similarity': f"{np.mean(result.similarity_scores):.3f}" if result.similarity_scores else "0.000",
                    'Chunks Retrieved': len(result.retrieved_chunks),
                    'Unique Sources': len(result.sources)
                })

            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)

            # Export results
            if st.button("📥 Export Results as JSON"):
                results_dict = {
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'results': [
                        {
                            'method': r.method,
                            'total_time': r.total_time,
                            'retrieval_time': r.retrieval_time,
                            'generation_time': r.generation_time,
                            'response': r.response,
                            'similarity_scores': r.similarity_scores,
                            'sources': r.sources
                        } for r in results
                    ]
                }

                st.download_button(
                    label="Download JSON",
                    data=json.dumps(results_dict, indent=2),
                    file_name=f"rag_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    asyncio.run(main())