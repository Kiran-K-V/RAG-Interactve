"""
Chunking functions for different text splitting strategies.
"""

import re
import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import SimpleDirectoryReader
from embeddings import LocalSentenceTransformerEmbedding
from config import GEMINI_API_KEY, DOCUMENT_PATH


# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


@st.cache_data
def load_document():
    """Load the document from file."""
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def fixed_size_chunking(text: str, chunk_size: int, chunk_overlap: int):
    """
    Split text into fixed-size chunks with specified overlap.

    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def semantic_chunking(breakpoint_threshold):
    """
    Split text into semantic chunks using LlamaIndex SemanticSplitterNodeParser.

    Args:
        breakpoint_threshold: Threshold for semantic breakpoints

    Returns:
        List of semantic chunks
    """
    documents = SimpleDirectoryReader(input_files=[DOCUMENT_PATH]).load_data()
    embed_model = LocalSentenceTransformerEmbedding()

    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=breakpoint_threshold,
        embed_model=embed_model,
    )

    nodes = splitter.get_nodes_from_documents(documents)
    return [node.text for node in nodes]


def hierarchical_chunking(text: str):
    """
    Uses Gemini LLM to split text into hierarchical chunks such as:
    - Paragraph-level units
    - Sentences within paragraphs
    - Bullet lists as individual chunks

    You must initialize the `model` (e.g., genai.GenerativeModel) before calling.
    """
    # prompt = f"""
    # Please split the following text into hierarchical chunks. Follow these rules:
    # 1. Each paragraph should be a separate chunk
    # 2. Long paragraphs (>300 words) should be split into smaller logical units
    # 3. Bullet points or numbered lists should be individual chunks
    # 4. Maintain context and coherence within each chunk
    # 5. Return only the chunks, separated by "---CHUNK---"

    # Text:
    # {text}
    # """

    # try:
    #     response = model.generate_content(prompt)
    #     chunks = response.text.split("---CHUNK---")
    #     return [chunk.strip() for chunk in chunks if chunk.strip()]
    # except Exception as e:
    #     st.error(f"Error in hierarchical chunking: {e}")
    #     # Fallback to simple paragraph splitting
    #     return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    lines = text.split("\n")
    # Initialize structure
    hierarchy = []
    parent_stack = []

    for line in lines:
        # Check for different header levels
        header = re.match(r"^(#+)\s*(.*)", line)
        bullet = re.match(r"^[-*]\s*(.*)", line.strip())

        if header:
            level = len(header.group(1))
            title = header.group(2).strip()
            node = {"type": f"h{level}", "title": title, "children": [], "content": []}
            # Ascend/descend hierarchy as per header level
            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()
            if parent_stack:
                parent_stack[-1][1]["children"].append(node)
            else:
                hierarchy.append(node)
            parent_stack.append((level, node))
        elif bullet:
            # Attach bullets to the correct parent
            if parent_stack:
                parent_stack[-1][1]["content"].append(bullet.group(1).strip())
        elif line.strip() != "":
            # Other content (not a header or bullet)
            if parent_stack:
                parent_stack[-1][1]["content"].append(line.strip())
    return hierarchy


def propositional_chunking(text: str):
    """
    Uses Gemini LLM to split text into propositional chunks.
    You must initialize the `model` (e.g., genai.GenerativeModel) before calling.
    """
    prompt = f"""
    Please split the following text into propositional chunks. Each chunk should contain:
    1. A single main proposition or claim
    2. Supporting evidence or context for that proposition
    3. Complete, self-contained information
    4. Return only the chunks, separated by "---CHUNK---"

    Text:
    {text}
    """

    try:
        response = model.generate_content(prompt)
        chunks = response.text.split("---CHUNK---")
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    except Exception as e:
        st.error(f"Error in propositional chunking: {e}")
        # Fallback to sentence-based splitting
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) > 200:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks


def recursive_chunking(text):
    """
    Split text using recursive character text splitter.

    Args:
        text: Input text to chunk

    Returns:
        List of recursively split chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)
