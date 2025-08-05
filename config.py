"""
Configuration settings and constants for the RAG Interactive application.
"""

# Qdrant Configuration
QDRANT_URL: str = "https://eecc4d29-21eb-4d25-8a1b-eb23e1ef755f.us-east4-0.gcp.cloud.qdrant.io:6333/"
QDRANT_API_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.GD9lmgww5vC-oZ70qukwmd-vn3PaXIopyP5SpTldTdA"

# Collection name mapping for different chunking methods
COLLECTION_MAPPING = {
    "Semantic Chunking": "semantic_chunks_leave_policy_new",
    "Fixed Size Chunking": "fixed_size_chunks_leave_policy_new",
    "Recursive Chunking": "recursive_chunks_leave_policy",
    "Hierarchical Chunking": "hierarchical_chunks_leave_policy_new",
    "Propositional Chunking": "propositional_chunks_leave_policy_new"
}

# Document path
DOCUMENT_PATH = "leave_policy.txt"

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
