# app/retrieval.py
import numpy as np
import json
from typing import List, Optional
from sqlalchemy.orm import Session
from .models import Chunk, Document
from .embeddings import embed_texts

def _to_vector(arr):
    try:
        v = np.array(arr, dtype=float)
        if v.ndim == 1:
            norm = np.linalg.norm(v) + 1e-12
            return v / norm
        return v
    except Exception:
        return None

def search_chunks(db: Session, query: str, k: int = 8, org_id: Optional[int] = None) -> List[dict]:
    """
    Simple retrieval:
    - loads all Chunk rows joined with their Document metadata
    - filters by org_id when provided
    - computes cosine similarity between query embedding and chunk embeddings in Python
    - returns top-k ordered results with fields: text, title, source, page, rank, score
    """
    # fetch chunks joined with document
    q = db.query(Chunk, Document).join(Document, Chunk.document_id == Document.id)
    if org_id is not None:
        q = q.filter(Document.org_id == org_id)

    rows = q.all()
    if not rows:
        return []

    texts = []
    metas = []
    embeddings = []
    for (c, d) in rows:
        texts.append(c.text)
        metas.append({"title": d.title or "doc", "source": d.source, "page": c.page})
        try:
            emb = json.loads(c.embedding_json)
        except Exception:
            # if embedding is stored as plain text of a list, try eval-like parse fallback
            try:
                emb = json.loads(str(c.embedding_json))
            except Exception:
                emb = None
        if emb is None:
            # if missing, fallback to zeros
            embeddings.append(np.zeros(32, dtype=float))
        else:
            embeddings.append(np.array(emb, dtype=float))

    if len(embeddings) == 0:
        return []

    # normalize embeddings
    embeddings_np = np.vstack(
        [(e / (np.linalg.norm(e) + 1e-12)) if np.linalg.norm(e) > 0 else e for e in embeddings]
    )

    # compute query embedding using same embed_texts helper
    q_emb_list = embed_texts([query])
    if not q_emb_list:
        return []
    q_emb = np.array(q_emb_list[0], dtype=float)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    # cosine similarities
    sims = embeddings_np @ q_emb

    # top-k indices
    idx = np.argsort(sims)[::-1][:k]

    out = []
    for rank, i in enumerate(idx, start=1):
        i = int(i)
        out.append({
            "text": texts[i],
            "title": metas[i]["title"],
            "source": metas[i]["source"],
            "page": metas[i]["page"],
            "rank": rank,
            "score": float(sims[i]),
        })
    return out
