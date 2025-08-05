"""
Main Streamlit application for RAG Interactive - Chunking & RAG Systems Comparison.
"""

import streamlit as st
import asyncio
from datetime import datetime

# Import custom modules
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_MAPPING
from chunking import (
    load_document,
    fixed_size_chunking,
    semantic_chunking,
    hierarchical_chunking,
    propositional_chunking,
    recursive_chunking,
)
from rag_system import QdrantRAGSystem
from ui_components import (
    create_performance_charts,
    display_results,
    display_single_result,
    create_summary_table,
    export_results_json,
)
from embeddings import load_embedding_model

# Import dependencies with error handling
try:
    from qdrant_client import QdrantClient  # noqa: F401
    from qdrant_client.models import Filter, FieldCondition, MatchValue  # noqa: F401
    from sentence_transformers import SentenceTransformer  # noqa: F401
    import google.generativeai as genai  # noqa: F401
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
except ImportError:
    st.warning(
        "Some dependencies are missing. RAG functionality may not work without: "
        "qdrant-client, sentence-transformers, google-generativeai, scikit-learn"
    )


# Configure page
st.set_page_config(
    page_title="Chunking & RAG Systems Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Chunking Comparison"

# Sidebar Navigation
with st.sidebar:
    st.title("🔍 Navigation")
    if st.button("📦 Chunking Comparison"):
        st.session_state.page = "Chunking Comparison"
    if st.button("🤖 RAG Comparison"):
        st.session_state.page = "RAG Comparison"


def get_chunking_parameters(chunk_method):
    """Get method-specific parameters for chunking."""
    params = {}

    if chunk_method == "Fixed Size":
        col1, col2 = st.columns(2)
        with col1:
            params["chunk_size"] = st.number_input(
                "Chunk size", min_value=50, max_value=2000, value=500
            )
        with col2:
            params["chunk_overlap"] = st.number_input(
                "Chunk overlap",
                min_value=0,
                max_value=params["chunk_size"] - 1,
                value=50,
            )
    elif chunk_method == "Semantic":
        params["breakpoint_threshold"] = st.slider(
            "Breakpoint Threshold", min_value=50, max_value=100, value=80
        )

    return params


def perform_chunking(document_text, chunk_method, params):
    """Perform chunking based on selected method and parameters."""
    if chunk_method == "Fixed Size":
        return fixed_size_chunking(
            document_text, params["chunk_size"], params["chunk_overlap"]
        )
    elif chunk_method == "Semantic":
        return semantic_chunking(breakpoint_threshold=params["breakpoint_threshold"])
    elif chunk_method == "Hierarchical":
        return hierarchical_chunking(document_text)
    elif chunk_method == "Propositional":
        return propositional_chunking(document_text)
    elif chunk_method == "Recursive":
        return recursive_chunking(document_text)
    else:
        return []


def display_chunk_statistics(chunks):
    """Display statistics about the generated chunks."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", len(chunks))
    with col2:
        avg_length = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
        st.metric("Avg Chunk Length", f"{avg_length:.0f} chars")
    with col3:
        min_length = min(len(chunk) for chunk in chunks) if chunks else 0
        st.metric("Min Chunk Length", f"{min_length} chars")
    with col4:
        max_length = max(len(chunk) for chunk in chunks) if chunks else 0
        st.metric("Max Chunk Length", f"{max_length} chars")


def display_chunks_with_highlights(chunks):
    """Display chunks with keyword highlighting."""
    st.subheader("📄 Generated Chunks")
    keywords = ["leave", "policy", "employee", "vacation", "sick", "medical"]

    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"Chunk {i} (Length: {len(chunk)} chars)", expanded=False):
            st.markdown(f"```text\n{chunk}\n```")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Character Count", len(chunk))
            with col2:
                st.metric("Word Count", len(chunk.split()))

            # Keyword highlighting
            highlighted = chunk
            for word in keywords:
                highlighted = highlighted.replace(word, f"**{word}**")

            st.markdown("**Keyword highlights:**")
            st.markdown(highlighted)


def display_hierarchical_chunks(chunks, level=1, keywords=None):
    """Recursively display hierarchical chunks with keyword highlighting."""
    if keywords is None:
        keywords = ["leave", "policy", "employee", "vacation", "sick", "medical"]

    indent = "  " * (level - 1)  # Simple indentation for console or logs (optional)

    for i, chunk in enumerate(chunks, 1):
        # Create a title combining type and title, with indentation for visual hierarchy
        title = f"{indent}{chunk.get('type', '').upper()} - {chunk.get('title', 'No Title')}"

        with st.expander(title, expanded=False):
            # Combine content list into a single string with line breaks
            content_text = "\n".join(chunk.get("content", []))
            if not content_text.strip():
                st.markdown("No content available for this chunk.")
            else:
                st.markdown(f"```text\n{content_text}\n```")

                # Show metrics about the chunk content
                char_count = len(content_text)
                word_count = len(content_text.split())
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Character Count", char_count)
                with col2:
                    st.metric("Word Count", word_count)

                # Keyword highlighting in content display
                highlighted = content_text
                for word in keywords:
                    highlighted = highlighted.replace(word, f"**{word}**")

                st.markdown("**Keyword Highlights:**")
                st.markdown(highlighted)

            # Recursively display children chunks if any
            if chunk.get("children"):
                display_hierarchical_chunks(
                    chunk["children"], level=level + 1, keywords=keywords
                )


def render_document_preview():
    """Render the document preview section."""
    document_text = load_document()
    st.subheader("📌 Document Preview")
    with st.expander("View Document Content", expanded=False):
        st.code(document_text, language="markdown")
    return document_text


def render_chunking_method_selection():
    """Render the chunking method selection UI."""
    chunk_method = st.selectbox(
        "Select chunking method",
        ["Fixed Size", "Semantic", "Hierarchical", "Propositional", "Recursive"],
    )
    params = get_chunking_parameters(chunk_method)
    return chunk_method, params


def process_and_display_chunks(document_text, chunk_method, params):
    """Process the document and display the resulting chunks."""
    try:
        chunks = perform_chunking(document_text, chunk_method, params)

        st.success(f"✅ Generated {len(chunks)} chunks using `{chunk_method}` method.")

        # Display chunk statistics
        display_chunk_statistics(chunks)

        if chunk_method == "Hierarchical":
            display_hierarchical_chunks(chunks)
        else:
            display_chunks_with_highlights(chunks)

    except Exception as e:
        st.error(f"Error during chunking: {str(e)}")


def render_chunking_page():
    """Render the chunking comparison page."""
    st.title("📦 Chunking Method Comparison")

    # Load and display document
    document_text = render_document_preview()
    _ = load_embedding_model()

    # Get chunking method and parameters
    chunk_method, params = render_chunking_method_selection()

    # Process chunks when button is clicked
    if st.button("🔄 Chunk Document", type="primary"):
        with st.spinner(f"Processing document with {chunk_method} chunking..."):
            process_and_display_chunks(document_text, chunk_method, params)


# ================================
# CHUNKING SECTION
# ================================

if st.session_state.page == "Chunking Comparison":
    render_chunking_page()


# ================================
# RAG SECTION
# ================================

elif st.session_state.page == "RAG Comparison":

    async def rag_main():
        st.title("🤖 RAG Systems Comparison Dashboard")
        st.markdown(
            "Compare Simple RAG, Hybrid RAG, and Multi-Query RAG performance on your Qdrant collections"
        )

        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")

            st.subheader("API Configuration")
            GEMINI_API_KEY= st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Enter your Gemini API key...",
                help="Get your API key from Google AI Studio"
            )

            # Collection selection
            st.subheader("Collection Settings")
            mapped_name = st.selectbox(
                "🔍 Select Chunking Strategy (Leave Policy)",
                options=list(COLLECTION_MAPPING.keys()),
            )

            collection_name = COLLECTION_MAPPING[mapped_name]

            # RAG parameters
            st.subheader("RAG Parameters")
            top_k = st.slider("Top K Results", 1, 10, 5)
            vector_weight = st.slider("Vector Weight (Hybrid RAG)", 0.0, 1.0, 0.7, 0.1)

        # Main interface
        if not collection_name or GEMINI_API_KEY is None:
            st.warning(
                "Please provide all required configuration parameters in the sidebar."
            )
            return

        # Initialize RAG system
        try:
            with st.spinner("Initializing RAG system..."):
                rag_system = QdrantRAGSystem(QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY)
            st.success("✅ RAG system initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {str(e)}")
            return

        # Query input
        query = st.text_area(
            "💬 Enter your query:",
            placeholder="What would you like to know about your documents?",
            height=100,
        )

        if not query:
            st.info("👆 Please enter a query to proceed.")
            return

        # Create columns for separate method buttons
        st.subheader("🚀 Run Individual Methods")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔍 Run Simple RAG", type="primary", use_container_width=True):
                with st.spinner("Running Simple RAG..."):
                    try:
                        result = await rag_system.simple_rag(
                            query, collection_name, top_k
                        )
                        st.success(
                            f"✅ Simple RAG completed in {result.total_time:.3f}s"
                        )

                        # Display results
                        st.subheader("📊 Simple RAG Results")
                        display_single_result(result)

                    except Exception as e:
                        st.error(f"❌ Simple RAG failed: {str(e)}")

        with col2:
            if st.button("🔄 Run Hybrid RAG", type="primary", use_container_width=True):
                with st.spinner("Running Hybrid RAG..."):
                    try:
                        result = await rag_system.hybrid_rag(
                            query, collection_name, top_k, vector_weight
                        )
                        st.success(
                            f"✅ Hybrid RAG completed in {result.total_time:.3f}s"
                        )

                        # Display results
                        st.subheader("📊 Hybrid RAG Results")
                        display_single_result(result)

                    except Exception as e:
                        st.error(f"❌ Hybrid RAG failed: {str(e)}")

        with col3:
            if st.button(
                "🎯 Run Multi-Query RAG", type="primary", use_container_width=True
            ):
                with st.spinner("Running Multi-Query RAG..."):
                    try:
                        result = await rag_system.multi_query_rag(
                            query, collection_name, top_k
                        )
                        st.success(
                            f"✅ Multi-Query RAG completed in {result.total_time:.3f}s"
                        )

                        # Display results
                        st.subheader("📊 Multi-Query RAG Results")
                        display_single_result(result)

                    except Exception as e:
                        st.error(f"❌ Multi-Query RAG failed: {str(e)}")

        # Add separator
        st.divider()

        # Compare all methods
        st.subheader("📈 Compare All Methods")
        if st.button(
            "🚀 Compare All Methods", help="Run all three methods and compare results"
        ):
            st.info(
                "🔄 Running all RAG methods for comparison... This may take a moment."
            )

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            methods_to_run = [
                ("Simple RAG", rag_system.simple_rag),
                (
                    "Hybrid RAG",
                    lambda q, c, k: rag_system.hybrid_rag(q, c, k, vector_weight),
                ),
                ("Multi-Query RAG", rag_system.multi_query_rag),
            ]

            for i, (method_name, method_func) in enumerate(methods_to_run):
                status_text.text(f"🔄 Running {method_name}...")

                try:
                    result = await method_func(query, collection_name, top_k)
                    results.append(result)
                    st.success(
                        f"✅ {method_name} completed in {result.total_time:.3f}s"
                    )
                except Exception as e:
                    st.error(f"❌ {method_name} failed: {str(e)}")

                progress_bar.progress((i + 1) / len(methods_to_run))

            status_text.text("✅ Comparison complete!")

            if results:
                # Performance charts
                st.header("📈 Performance Analysis")
                fig = create_performance_charts(results)
                st.plotly_chart(fig, use_container_width=True)

                # Summary table
                st.header("📊 Summary Table")
                df = create_summary_table(results)
                st.dataframe(df, use_container_width=True)

                # Detailed results
                st.header("📋 Detailed Results")
                display_results(results)

                # Export results
                st.header("📥 Export Results")
                if st.button("📥 Export Results as JSON"):
                    json_data = export_results_json(results, query)

                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"rag_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

    # Run the async RAG main function
    asyncio.run(rag_main())


# Footer
st.markdown("---")
st.markdown("🔍 **RAG Interactive** - Built with Streamlit, LlamaIndex, and Qdrant")
