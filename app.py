from datetime import datetime
import os
import json

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="COE Digital Enablement Prototype", layout="wide")

DATA_PATH = "data/coe_initiatives.csv"

RAG_DIR = "rag_store"
FAISS_INDEX_PATH = os.path.join(RAG_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(RAG_DIR, "chunks.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# -----------------------------
# Data Utils (Dashboard)
# -----------------------------
def to_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip().replace("%", "")
    try:
        return float(x)
    except Exception:
        return np.nan


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Dates
    df["Start_Date"] = pd.to_datetime(df["Start_Date"], errors="coerce")
    df["End_Date"] = pd.to_datetime(df["End_Date"], errors="coerce")

    # KPI cleanup
    df["KPI_Target"] = df["KPI_Target"].apply(to_num)
    df["KPI_Achieved"] = df["KPI_Achieved"].apply(to_num)

    # Owner normalization
    df["Owner"] = df["Owner"].astype(str).str.strip()
    df["Owner"] = df["Owner"].apply(lambda s: " ".join([w.capitalize() for w in s.split()]))

    # Derived metrics
    df["KPI_Achievement_Pct"] = np.where(
        df["KPI_Target"] > 0, (df["KPI_Achieved"] / df["KPI_Target"]) * 100, np.nan
    )

    today = pd.Timestamp(datetime.today().date())
    df["Missing_End_Date"] = df["End_Date"].isna()

    # Delay heuristic
    df["Delay_Flag"] = np.where(
        (df["End_Date"].notna()) & (today > df["End_Date"]) & (df["Status"] != "Completed"),
        True,
        False
    )

    return df


def format_benefit(row) -> str:
    return f"{row['Business_Benefit_Value']} {row['Business_Benefit_Unit']}"


# -----------------------------
# RAG Utils (FAISS)
# -----------------------------
@st.cache_resource
def load_rag_assets():
    """
    Loads FAISS index, chunk metadata, and embedding model once.
    Requires:
      - rag_store/faiss.index
      - rag_store/chunks.json
    """
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None, None

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    return index, chunks, model


def retrieve_top_k(index, chunks, model, query: str, k: int = 3):
    """
    Returns top-k chunks using cosine similarity:
      - embeddings were normalized at build time
      - index uses inner product => cosine similarity for normalized vectors
    """
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        meta = chunks[int(idx)]
        results.append(
            {
                "score": float(score),
                "source": meta.get("source"),
                "chunk_index": meta.get("chunk_index"),
                "text": meta.get("text"),
            }
        )

    overall = results[0]["score"] if results else 0.0
    return results, overall


def confidence_label(score: float) -> str:
    # cosine similarity typically 0..1 (for relevant matches)
    if score >= 0.55:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def grounded_answer(query: str, retrieved: list) -> str:
    """
    No external LLM to keep it stable + grounded.
    Produces an evidence-based answer by extracting key points from retrieved text.
    """
    if not retrieved:
        return "I couldn't find enough relevant content in the COE knowledge base to answer that."

    bullets = []
    for r in retrieved[:3]:
        txt = " ".join(r["text"].split())
        parts = txt.split(". ")
        snippet = ". ".join(parts[:2]).strip()
        if snippet and not snippet.endswith("."):
            snippet += "."
        bullets.append(snippet[:320])

    answer = (
        "Based on the COE knowledge base, here’s a grounded response:\n\n"
        + "\n".join([f"- {b}" for b in bullets])
        + "\n\nIf needed, refer to the retrieved sources below for exact context."
    )
    return answer


# -----------------------------
# UI
# -----------------------------
st.title("Mini COE Digital Enablement Prototype")
st.caption("Part 1: COE Analytics Dashboard • Part 2: COE AI Assistant (RAG)")

tab1, tab2 = st.tabs(["COE Dashboard", "COE AI Assistant (RAG)"])

# -----------------------------
# Tab 1: Dashboard
# -----------------------------
with tab1:
    st.subheader("COE Analytics Mini Dashboard")

    with st.spinner("Loading initiatives dataset..."):
        df = load_and_clean_data(DATA_PATH)

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        status_filter = st.multiselect(
            "Status",
            sorted(df["Status"].dropna().unique().tolist()),
            default=sorted(df["Status"].dropna().unique().tolist()),
        )
    with f2:
        owner_filter = st.multiselect(
            "Owner",
            sorted(df["Owner"].dropna().unique().tolist()),
            default=sorted(df["Owner"].dropna().unique().tolist()),
        )
    with f3:
        dept_filter = st.multiselect(
            "Department",
            sorted(df["Department"].dropna().unique().tolist()),
            default=sorted(df["Department"].dropna().unique().tolist()),
        )

    dff = df[df["Status"].isin(status_filter) & df["Owner"].isin(owner_filter) & df["Department"].isin(dept_filter)]

    # KPI cards (assignment requirements)
    total = len(dff)
    pct_on_track = (dff["Status"].eq("On Track").mean() * 100) if total else 0
    pct_delayed = (dff["Status"].eq("Delayed").mean() * 100) if total else 0
    avg_kpi = float(dff["KPI_Achievement_Pct"].mean()) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Initiatives", total)
    c2.metric("% On Track", f"{pct_on_track:.1f}%")
    c3.metric("% Delayed", f"{pct_delayed:.1f}%")
    c4.metric("Avg KPI Achievement %", f"{avg_kpi:.1f}%")

    left, right = st.columns(2)

    with left:
        st.subheader("Status Distribution")
        status_counts = dff["Status"].value_counts()
        fig = plt.figure()
        plt.bar(status_counts.index, status_counts.values)
        plt.xlabel("Status")
        plt.ylabel("Count")
        st.pyplot(fig)

    with right:
        st.subheader("Top 3 Initiatives by Business Benefit (Value)")
        top3 = dff.sort_values("Business_Benefit_Value", ascending=False).head(3)

        fig2 = plt.figure()
        plt.bar(top3["Initiative_ID"], top3["Business_Benefit_Value"])
        plt.xlabel("Initiative")
        plt.ylabel("Benefit Value")
        st.pyplot(fig2)

        st.dataframe(
            top3[["Initiative_ID", "Initiative_Name", "Owner", "Status", "Business_Benefit_Value", "Business_Benefit_Unit"]],
            width="stretch",
        )

    st.subheader("Data Quality Checks (COE Lens)")
    dq1, dq2, dq3 = st.columns(3)
    dq1.metric("Missing End Dates", int(dff["Missing_End_Date"].sum()))
    dq2.metric("Delay Flag (Heuristic)", int(dff["Delay_Flag"].sum()))
    dq3.metric("Missing KPI Values", int(dff["KPI_Target"].isna().sum() + dff["KPI_Achieved"].isna().sum()))

    st.subheader("Initiatives Table")
    dff2 = dff.copy()
    dff2["Business_Benefit"] = dff2.apply(format_benefit, axis=1)

    st.dataframe(
        dff2[
            [
                "Initiative_ID",
                "Initiative_Name",
                "Owner",
                "Department",
                "Start_Date",
                "End_Date",
                "Status",
                "KPI_Target",
                "KPI_Achieved",
                "KPI_Achievement_Pct",
                "Business_Benefit",
            ]
        ],
        width="stretch",
    )


# -----------------------------
# Tab 2: RAG Assistant
# -----------------------------
with tab2:
    st.subheader("COE AI Assistant (RAG)")
    st.caption("Retrieves top-k sections from COE docs and answers with sources + confidence.")

    index, chunks, model = load_rag_assets()

    if index is None:
        st.error("RAG assets not found. Please run: python rag_pipeline.py (FAISS builder) first.")
        st.code("python rag_pipeline.py")
    else:
        q = st.text_input(
            "Ask a COE question",
            placeholder="e.g., Explain Six Sigma DMAIC. What are KPI tracking best practices?",
        )

        cA, cB = st.columns([1, 1])
        with cA:
            top_k = st.selectbox("Top-k retrieval", [3, 5, 7], index=0)
        with cB:
            show_sources = st.checkbox("Show retrieved sources", value=True)

        if st.button("Ask"):
            if not q.strip():
                st.warning("Please enter a question.")
            else:
                retrieved, overall = retrieve_top_k(index, chunks, model, q, k=int(top_k))
                label = confidence_label(overall)

                st.markdown(f"**Confidence:** {label} ({overall:.2f})")

                st.subheader("Answer")
                st.write(grounded_answer(q, retrieved))

                if show_sources:
                    st.subheader("Retrieved Sources")
                    for i, r in enumerate(retrieved, start=1):
                        st.markdown(
                            f"**{i}. {r['source']}**  | chunk {r['chunk_index']}  | similarity: **{r['score']:.2f}**"
                        )
                        st.code(r["text"][:1200] + ("..." if len(r["text"]) > 1200 else ""))


