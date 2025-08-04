"""
UI components and visualization functions for the RAG Interactive application.
"""

import json
from typing import List
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rag_system import RAGResult


def create_performance_charts(results: List[RAGResult]):
    """Create performance comparison charts"""

    # Prepare data
    methods = [r.method for r in results]
    retrieval_times = [r.retrieval_time for r in results]
    generation_times = [r.generation_time for r in results]
    total_times = [r.total_time for r in results]
    avg_scores = [
        np.mean(r.similarity_scores) if r.similarity_scores else 0 for r in results
    ]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Response Times",
            "Average Similarity Scores",
            "Time Breakdown",
            "Retrieved Sources Count",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Response times comparison
    fig.add_trace(
        go.Bar(name="Total Time", x=methods, y=total_times, marker_color="lightblue"),
        row=1,
        col=1,
    )

    # Average similarity scores
    fig.add_trace(
        go.Bar(name="Avg Score", x=methods, y=avg_scores, marker_color="lightgreen"),
        row=1,
        col=2,
    )

    # Time breakdown
    fig.add_trace(
        go.Bar(name="Retrieval", x=methods, y=retrieval_times, marker_color="orange"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(name="Generation", x=methods, y=generation_times, marker_color="red"),
        row=2,
        col=1,
    )

    # Sources count
    sources_count = [len(r.sources) for r in results]
    fig.add_trace(
        go.Bar(name="Sources", x=methods, y=sources_count, marker_color="purple"),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800, showlegend=True, title_text="RAG Methods Performance Comparison"
    )
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
                avg_score = (
                    np.mean(result.similarity_scores) if result.similarity_scores else 0
                )
                st.metric("Avg Similarity", f"{avg_score:.3f}")
                st.metric("Unique Sources", len(result.sources))

            st.subheader("Generated Response")
            st.write(result.response)

            st.subheader("Retrieved Chunks")
            for i, chunk in enumerate(result.retrieved_chunks):
                with st.container():
                    st.write(f"**Chunk {i + 1}** (Score: {chunk['score']:.3f})")
                    st.write(f"*Source: {chunk['source']}*")
                    st.write(
                        chunk["text"][:300] + "..."
                        if len(chunk["text"]) > 300
                        else chunk["text"]
                    )

                    # Show additional info for hybrid RAG
                    if "vector_score" in chunk:
                        st.caption(
                            f"Vector: {chunk['vector_score']:.3f}, Keyword: {chunk['keyword_score']:.3f}"
                        )

                    # Show matched query for multi-query RAG
                    if "matched_query" in chunk:
                        st.caption(f"Matched query: {chunk['matched_query']}")

                    st.divider()


def display_single_result(result: RAGResult):
    """Display results for a single RAG method"""

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Time", f"{result.total_time:.3f}s")
    with col2:
        st.metric("Retrieval Time", f"{result.retrieval_time:.3f}s")
    with col3:
        st.metric("Generation Time", f"{result.generation_time:.3f}s")
    with col4:
        avg_score = np.mean(result.similarity_scores) if result.similarity_scores else 0
        st.metric("Avg Similarity", f"{avg_score:.3f}")

    # Second metrics row
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Chunks Retrieved", len(result.retrieved_chunks))
    with col6:
        st.metric("Unique Sources", len(result.sources))

    # Generated Response
    with st.expander("📝 Generated Response", expanded=True):
        st.write(result.response)

    # Retrieved Chunks
    with st.expander(
        f"📚 Retrieved Chunks ({len(result.retrieved_chunks)})", expanded=False
    ):
        for i, chunk in enumerate(result.retrieved_chunks):
            st.write(f"**Chunk {i + 1}** (Score: {chunk['score']:.3f})")
            st.write(f"*Source: {chunk['source']}*")
            st.write(
                chunk["text"][:300] + "..."
                if len(chunk["text"]) > 300
                else chunk["text"]
            )

            # Show additional info for hybrid RAG
            if "vector_score" in chunk:
                st.caption(
                    f"Vector: {chunk['vector_score']:.3f}, Keyword: {chunk['keyword_score']:.3f}"
                )

            # Show matched query for multi-query RAG
            if "matched_query" in chunk:
                st.caption(f"Matched query: {chunk['matched_query']}")

            if i < len(result.retrieved_chunks) - 1:
                st.divider()

    # Show query variations for Multi-Query RAG
    if result.method == "Multi-Query RAG" and result.query_variations:
        with st.expander("🔍 Generated Query Variations", expanded=False):
            st.write("**Original Query:**")
            st.code(result.query)

            st.write("**Generated Variations:**")
            for i, variation in enumerate(
                result.query_variations[:-1], 1
            ):  # Exclude original query (last item)
                st.write(f"**Variation {i}:**")
                st.code(variation)

            st.info(
                f"Total queries used: {len(result.query_variations)} (1 original + {len(result.query_variations) - 1} variations)"
            )

    # Export single result
    if st.button(f"📥 Export {result.method} Results", key=f"export_{result.method}"):
        result_dict = {
            "query": result.query,
            "method": result.method,
            "timestamp": datetime.now().isoformat(),
            "total_time": result.total_time,
            "retrieval_time": result.retrieval_time,
            "generation_time": result.generation_time,
            "response": result.response,
            "similarity_scores": result.similarity_scores,
            "sources": result.sources,
            "retrieved_chunks": result.retrieved_chunks,
            "query_variations": result.query_variations
            if result.query_variations
            else [],
        }

        st.download_button(
            label="Download JSON",
            data=json.dumps(result_dict, indent=2),
            file_name=f"{result.method.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key=f"download_{result.method}",
        )


def create_summary_table(results: List[RAGResult]) -> pd.DataFrame:
    """Create a summary table from RAG results"""
    summary_data = []
    for result in results:
        summary_data.append(
            {
                "Method": result.method,
                "Total Time (s)": f"{result.total_time:.3f}",
                "Retrieval Time (s)": f"{result.retrieval_time:.3f}",
                "Generation Time (s)": f"{result.generation_time:.3f}",
                "Avg Similarity": f"{np.mean(result.similarity_scores):.3f}"
                if result.similarity_scores
                else "0.000",
                "Chunks Retrieved": len(result.retrieved_chunks),
                "Unique Sources": len(result.sources),
            }
        )

    return pd.DataFrame(summary_data)


def export_results_json(results: List[RAGResult], query: str) -> str:
    """Export results as JSON string"""
    results_dict = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "method": r.method,
                "total_time": r.total_time,
                "retrieval_time": r.retrieval_time,
                "generation_time": r.generation_time,
                "response": r.response,
                "similarity_scores": r.similarity_scores,
                "sources": r.sources,
            }
            for r in results
        ],
    }

    return json.dumps(results_dict, indent=2)
