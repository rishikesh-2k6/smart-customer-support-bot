import os
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import DATA_DIR, CHROMA_DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, COMPANIES

_model = None
_client = None
_collections = {}

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def _infer_product_area(filepath: Path) -> str:
    """Infer product_area from the file path (e.g. data/hackerrank/screen/ → 'screen')."""
    parts = filepath.parts
    # parts looks like: ('data', 'hackerrank', 'screen', 'something.md')
    if len(parts) >= 3:
        return parts[2]  # subfolder name
    return "general"

def build_index(force_rebuild: bool = False):
    """
    Index all markdown files from data/ into ChromaDB.
    Creates one collection per company: 'hackerrank', 'claude', 'visa'.
    Skips if already indexed (unless force_rebuild=True).
    """
    client = _get_client()
    model = _get_model()

    for company in COMPANIES:
        company_dir = DATA_DIR / company
        if not company_dir.exists():
            print(f"[retriever] WARNING: {company_dir} not found, skipping.")
            continue

        # Check if already indexed
        existing = [c.name for c in client.list_collections()]
        if company in existing and not force_rebuild:
            print(f"[retriever] Collection '{company}' already exists, skipping indexing.")
            _collections[company] = client.get_collection(company)
            continue

        # Create or reset collection
        if company in existing:
            client.delete_collection(company)
        collection = client.create_collection(
            name=company,
            metadata={"hnsw:space": "cosine"}
        )
        _collections[company] = collection

        # Walk all .md files in this company's directory
        md_files = list(company_dir.rglob("*.md"))
        print(f"[retriever] Indexing {len(md_files)} files for '{company}'...")

        all_ids = []
        all_docs = []
        all_metas = []
        all_embeddings = []

        for filepath in md_files:
            try:
                text = filepath.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"[retriever] Could not read {filepath}: {e}")
                continue

            chunks = _chunk_text(text)
            product_area = _infer_product_area(filepath)

            for i, chunk in enumerate(chunks):
                # Use relative path for unique ID (multiple dirs may have same filename)
                rel_path = filepath.relative_to(company_dir).as_posix().replace("/", "_").replace(".md", "")
                doc_id = f"{company}::{rel_path}::{i}"
                all_ids.append(doc_id)
                all_docs.append(chunk)
                all_metas.append({
                    "company": company,
                    "product_area": product_area,
                    "source": str(filepath),
                    "chunk_index": i
                })

        if not all_docs:
            print(f"[retriever] No documents found for '{company}'.")
            continue

        # Batch embed and insert (ChromaDB handles batching internally, but we batch manually for safety)
        BATCH = 256
        for b in range(0, len(all_docs), BATCH):
            batch_docs = all_docs[b:b+BATCH]
            batch_embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
            collection.add(
                ids=all_ids[b:b+BATCH],
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=all_metas[b:b+BATCH]
            )

        print(f"[retriever] Indexed {len(all_docs)} chunks for '{company}'.")


def search(query: str, company: str | None = None, top_k: int = 5) -> list[dict]:
    """
    Search the corpus for the most relevant chunks.
    
    Args:
        query: The issue text to search with.
        company: 'hackerrank', 'claude', 'visa', or None (search all).
        top_k: Number of results to return.

    Returns:
        List of dicts: [{text, company, product_area, source, score}, ...]
        Sorted by score descending (highest similarity first).
    """
    client = _get_client()
    model = _get_model()

    query_embedding = model.encode([query])[0].tolist()

    # Determine which collections to search
    if company and company.lower() in COMPANIES:
        search_companies = [company.lower()]
    else:
        search_companies = COMPANIES

    results = []
    for comp in search_companies:
        if comp not in _collections:
            try:
                _collections[comp] = client.get_collection(comp)
            except Exception:
                continue

        coll = _collections[comp]
        count = coll.count()
        if count == 0:
            continue

        k = min(top_k, count)
        try:
            res = coll.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[retriever] Query error for {comp}: {e}")
            continue

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        distances = res["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (dist / 2) → range [0, 1]
            similarity = 1.0 - (dist / 2.0)
            results.append({
                "text": doc,
                "company": meta.get("company", comp),
                "product_area": meta.get("product_area", "general"),
                "source": meta.get("source", ""),
                "score": round(similarity, 4)
            })

    # Sort all results by similarity score descending, return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
