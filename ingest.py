"""
ingest.py
---------
Loads stories.csv, embeds each story description using a
free HuggingFace model, and saves the FAISS index to disk.

Run once before starting the API:
    python ingest.py
"""

import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


STORIES_PATH = "data/stories.csv"
INDEX_PATH   = "faiss_index"
EMBED_MODEL  = "all-MiniLM-L6-v2"


def load_and_embed():
    """Embed all stories and persist FAISS index."""
    df = pd.read_csv(STORIES_PATH)
    print(f"Loaded {len(df)} stories from {STORIES_PATH}")

    docs = []
    for _, row in df.iterrows():
        content = f"{row['title']}: {row['description']}"
        metadata = {
            "id":           int(row["id"]),
            "title":        row["title"],
            "genre":        row["genre"],
            "language":     row["language"],
            "duration_mins": int(row["duration_mins"]),
        }
        docs.append(Document(page_content=content, metadata=metadata))

    print("Loading embedding model (first run downloads ~90 MB) ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Embedding and indexing ...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_PATH)

    print(f"FAISS index saved to '{INDEX_PATH}/'")
    print("Done. You can now start the API with: uvicorn main:app --reload")


def search_stories(
    query: str,
    genre_filter: str   = None,
    language_filter: str = None,
    max_duration: int   = None,
    k: int              = 5,
) -> list[dict]:
    """
    Semantic search over the FAISS index with optional metadata filters.
    Returns a list of story metadata dicts.
    """
    embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore  = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )

    # Fetch extra results so filters don't leave us empty-handed
    raw_results = vectorstore.similarity_search(query, k=k * 3)

    filtered = []
    for doc in raw_results:
        m = doc.metadata
        if genre_filter    and m["genre"].lower()    != genre_filter.lower():
            continue
        if language_filter and m["language"].lower() != language_filter.lower():
            continue
        if max_duration    and m["duration_mins"]    >  max_duration:
            continue
        filtered.append(m)
        if len(filtered) == k:
            break

    # Fall back to unfiltered top-k if filters wiped everything out
    return filtered if filtered else [doc.metadata for doc in raw_results[:k]]


if __name__ == "__main__":
    load_and_embed()
