"""
Chunking functions for different text splitting strategies.
"""

import re
import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import DOCUMENT_PATH


@st.cache_data
def load_document():
    """Load the document from file."""
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def fixed_size_chunking(text: str, chunk_size: int, chunk_overlap: int):
    """
    Split text into fixed-size chunks with specified overlap.

    This implementation creates chunks of exactly chunk_size characters
    with the specified overlap between consecutive chunks.

    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk (in characters)
        chunk_overlap: Overlap between chunks (in characters)

    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end == text_length:
            break

        start = start + chunk_size - chunk_overlap

    return chunks


def semantic_chunking(breakpoint_threshold, gemini_api_key):
    """
    Split text into semantic chunks using LlamaIndex SemanticSplitterNodeParser.

    Args:
        breakpoint_threshold: Threshold for semantic breakpoints

    Returns:
        List of semantic chunks
    """
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    prompt = f"""
    Role: You are a highly proficient text analysis and document processing AI. Your task is to break down a provided document into semantically meaningful chunks.

Objective: The goal is not to create chunks of a fixed size (e.g., 500 characters) but to identify and group together sentences and paragraphs that convey a single, cohesive idea or topic. Each chunk should be self-contained and represent a distinct logical unit of the document.

Instructions:

* Analyze the Document: Read the entire document to understand its overall structure, main topics, and subtopics.
* Identify Semantic Boundaries: Look for natural transitions in the text. These are points where the author shifts from discussing one idea to another. These boundaries often occur at:
    * The end of a section or subsection.
    * The introduction of a new topic, concept, or argument.
    * The conclusion of a specific point.
    * Changes in the narrative or logical flow.
* Group Cohesive Content: Group together all sentences and paragraphs that contribute to a single, unified idea. A chunk should be a complete thought.
* Preserve Context: Each chunk should be able to be understood on its own, or with minimal context from the rest of the document. Ensure that the core meaning of the chunk is not lost.
* Output Format: Return only the chunks, separated by "---CHUNK---".

Example of an ideal chunk:
* A paragraph describing the causes of a historical event.
* A list of steps for a procedure, including its introduction and summary.
* A section defining a key term and providing a few examples.
* An argument presented in a paragraph, followed by a counter-argument in the next. (This could be two separate chunks, or one, depending on how tightly they are linked).

Text:
{text}

Expected Output:

The full text of the first semantic unit.
---CHUNK---
The full text of the second semantic unit.
---CHUNK---
The full text of the third semantic unit.
... and so on
    """

    chunks = model.generate_content(prompt)
    return chunks.text.split("---CHUNK---")


def hierarchical_chunking(text: str):
    """
    Uses Gemini LLM to split text into hierarchical chunks such as:
    - Paragraph-level units
    - Sentences within paragraphs
    - Bullet lists as individual chunks
    """

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


def propositional_chunking(text: str, gemini_api_key: str):
    """
    Uses Gemini LLM to split text into propositional chunks.
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

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

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
