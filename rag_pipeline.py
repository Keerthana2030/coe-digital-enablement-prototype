import os
import re
import glob
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DOCS_DIR = "coe_docs"
OUT_DIR = "rag_store"
INDEX_PATH = os.path.join(OUT_DIR, "faiss.index")
META_PATH = os.path.join(OUT_DIR, "chunks.json")
MODEL_NAME = "all-MiniLM-L6-v2"

def read_docs(docs_dir: str):
    paths = sorted(glob.glob(os.path.join(docs_dir, "*.md")))
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(p), "text": f.read()})
    return docs

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def main():
    ensure_outdir()

    docs = read_docs(DOCS_DIR)
    if not docs:
        print("No docs found in coe_docs/. Add .md files and try again.")
        return

    chunks = []
    for d in docs:
        chs = chunk_text(d["text"])
        for i, ch in enumerate(chs):
            chunks.append({
                "source": d["source"],
                "chunk_index": i,
                "text": ch
            })

    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True)  # cosine similarity friendly
    emb = np.array(emb).astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(emb)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"FAISS index built: {len(chunks)} chunks from {len(docs)} documents")
    print(f"Saved: {INDEX_PATH}")
    print(f"Saved: {META_PATH}")

if __name__ == "__main__":
    main()


