# app.py — FINAL (Replace entire file)
# COE Dashboard + Chatbot-style RAG (FAISS + SentenceTransformers) + OpenAI answer generation
# Matches assignment requirements:
# ✅ Top-3 retrieval (always)
# ✅ Show retrieved text alongside final answer (Evidence expander)
# ✅ LLM-based answer generation (OpenAI; falls back safely if key missing)
# ✅ Basic confidence score (similarity score + label)

from datetime import datetime
import os
import json
import re

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import faiss
from sentence_transformers import SentenceTransformer

# OpenAI (official SDK)
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="COE Digital Enablement Prototype", layout="wide")

DATA_PATH = "data/coe_initiatives.csv"
RAG_DIR = "rag_store"
FAISS_INDEX_PATH = os.path.join(RAG_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(RAG_DIR, "chunks.json")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# --- RAG minimum requirement: Top-3 retrieval ---
TOP_K = 3

# Governance thresholds (tune as needed)
SIM_HIGH = 0.55
SIM_MED = 0.40  # below this => REFUSE

REFUSAL_LINE = "Not enough information in the provided documents."

# OpenAI model (you can change this)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# -----------------------------
# Heavy CSS Theme
# -----------------------------
st.markdown(
    """
<style>
:root{
  --bg: rgba(255,255,255,0.035);
  --bg2: rgba(255,255,255,0.05);
  --stroke: rgba(255,255,255,0.10);
  --stroke2: rgba(255,255,255,0.14);
  --muted: rgba(255,255,255,0.70);
  --muted2: rgba(255,255,255,0.55);
  --shadow: 0 16px 40px rgba(0,0,0,0.32);
  --radius: 18px;
  --radius2: 24px;
}

/* Layout */
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1320px; }
section[data-testid="stSidebar"] { border-right: 1px solid var(--stroke); }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* Hero */
.hero{
  padding: 20px 20px;
  border-radius: var(--radius2);
  background: radial-gradient(1200px 600px at 10% 0%, rgba(125,125,255,0.22), transparent 50%),
              radial-gradient(900px 550px at 80% 10%, rgba(46,204,113,0.12), transparent 55%),
              radial-gradient(1000px 650px at 60% 90%, rgba(241,196,15,0.08), transparent 60%),
              rgba(255,255,255,0.03);
  border: 1px solid var(--stroke);
  box-shadow: var(--shadow);
  margin-bottom: 16px;
}
.hero-title { font-size: 42px; font-weight: 900; margin: 0; letter-spacing: -0.03em; }
.hero-sub { margin-top: 6px; color: var(--muted); font-size: 0.98rem; }

/* Panels / Cards */
.panel{
  border-radius: var(--radius2);
  border: 1px solid var(--stroke);
  background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.02));
  box-shadow: var(--shadow);
  padding: 16px;
}
.card{
  border-radius: var(--radius);
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.04);
  padding: 14px 14px;
}
.card-title{ font-size: 0.95rem; color: var(--muted2); margin-bottom: 6px; font-weight: 650; }
.card-value{ font-size: 1.9rem; font-weight: 900; }
.section-title{
  font-size: 1.18rem; font-weight: 900; margin: 2px 0 10px 0; letter-spacing: -0.01em;
}
.hr { height: 1px; background: var(--stroke); margin: 14px 0; }

/* Badges */
.badge{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 900;
  border: 1px solid var(--stroke2);
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.92);
}
.badge::before{
  content: "";
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.35);
}
.badge-high { background: rgba(46,204,113,0.18); border-color: rgba(46,204,113,0.35); }
.badge-high::before{ background: rgba(46,204,113,0.9); }
.badge-med  { background: rgba(241,196,15,0.18); border-color: rgba(241,196,15,0.35); }
.badge-med::before{ background: rgba(241,196,15,0.9); }
.badge-low  { background: rgba(231,76,60,0.18); border-color: rgba(231,76,60,0.35); }
.badge-low::before{ background: rgba(231,76,60,0.9); }

.chat-wrap{
  border-radius: var(--radius2);
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.03);
  padding: 14px;
}
.chat-row{
  display:flex;
  gap: 12px;
  margin: 10px 0;
}
.chat-row.user{ justify-content: flex-end; }
.chat-row.bot{ justify-content: flex-start; }

.bubble{
  max-width: 78%;
  padding: 14px 14px;
  border-radius: 18px;
  border: 1px solid var(--stroke);
  box-shadow: 0 10px 26px rgba(0,0,0,0.22);
  line-height: 1.35;
  white-space: pre-wrap;
}
.bubble.user{
  background: rgba(255,255,255,0.06);
  border-color: rgba(255,255,255,0.14);
}
.bubble.bot{
  background: rgba(95,90,255,0.16);
  border-color: rgba(130,125,255,0.28);
}
.meta{
  font-size: 0.85rem;
  color: var(--muted2);
  margin-left: 6px;
}
.source-card{
  border-radius: 14px;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.03);
  padding: 10px;
  margin: 10px 0;
}

/* Inputs rounding */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div{
  border-radius: 14px !important;
}

/* Streamlit chat input full width */
div[data-testid="stChatInput"] textarea{
  border-radius: 18px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Dashboard Helpers
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


@st.cache_data
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["Start_Date"] = pd.to_datetime(df["Start_Date"], errors="coerce")
    df["End_Date"] = pd.to_datetime(df["End_Date"], errors="coerce")

    df["KPI_Target"] = df["KPI_Target"].apply(to_num)
    df["KPI_Achieved"] = df["KPI_Achieved"].apply(to_num)

    df["Owner"] = df["Owner"].astype(str).str.strip()
    df["Owner"] = df["Owner"].apply(lambda s: " ".join([w.capitalize() for w in s.split()]))

    df["KPI_Achievement_Pct"] = np.where(
        df["KPI_Target"] > 0, (df["KPI_Achieved"] / df["KPI_Target"]) * 100, np.nan
    )

    today = pd.Timestamp(datetime.today().date())
    df["Missing_End_Date"] = df["End_Date"].isna()
    df["Delay_Flag"] = np.where(
        (df["End_Date"].notna()) & (today > df["End_Date"]) & (df["Status"] != "Completed"),
        True,
        False,
    )
    return df


def format_benefit(row) -> str:
    return f"{row['Business_Benefit_Value']} {row['Business_Benefit_Unit']}"


# -----------------------------
# RAG Helpers
# -----------------------------
@st.cache_resource
def load_rag_assets():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None, None
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    return index, chunks, model


def retrieve_top_k(index, chunks, model, query: str, k: int = TOP_K):
    """
    FAISS index assumed to be built for cosine similarity using normalized embeddings + IndexFlatIP.
    score is then cosine similarity in [-1, 1] typically.
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
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", "?"),
                "text": meta.get("text", ""),
            }
        )

    overall = results[0]["score"] if results else 0.0
    return results, overall


def confidence_label(score: float) -> str:
    if score >= SIM_HIGH:
        return "High"
    if score >= SIM_MED:
        return "Medium"
    return "Low"


def badge_html(label: str) -> str:
    if label == "High":
        return '<span class="badge badge-high">High</span>'
    if label == "Medium":
        return '<span class="badge badge-med">Medium</span>'
    return '<span class="badge badge-low">Low</span>'


def pack_context(retrieved, max_chars=2600):
    """
    Packs ONLY top-3 evidence blocks (requirement) into a compact context.
    """
    retrieved = (retrieved or [])[:TOP_K]
    packed = []
    used = 0

    for r in retrieved:
        src = r.get("source", "unknown")
        ck = r.get("chunk_index", "?")
        txt = (r.get("text") or "").strip()
        excerpt = re.sub(r"\s+", " ", txt)[:700].strip()
        block = f"[Source: {src} | chunk {ck}] {excerpt}"
        if used + len(block) > max_chars:
            break
        packed.append(block)
        used += len(block)

    return "\n".join(packed)


def extractive_fallback(retrieved):
    """
    Safe extractive fallback: produces evidence bullets + citations.
    """
    if not retrieved:
        return REFUSAL_LINE

    text = (retrieved[0].get("text") or "").strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    bullets = []
    seen = set()
    for ln in lines:
        ln = ln.lstrip("-*• ").strip()
        if not ln or ln.lower() in seen:
            continue
        seen.add(ln.lower())
        bullets.append(f"• {ln}")
        if len(bullets) >= 6:
            break

    sources = ", ".join(sorted({r.get("source", "unknown") for r in (retrieved or [])[:TOP_K]}))
    if not bullets:
        return REFUSAL_LINE
    return "\n".join(bullets) + f"\n\nCitations: {sources}"


def _clean_llm_answer(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    # remove common preambles
    for bad in ("Answer:", "Response:", "assistant:", "Assistant:", "###", "Context:", "Evidence:"):
        text = text.replace(bad, "")
    return text.strip()


@st.cache_resource
def get_openai_client():
    """
    Uses OPENAI_API_KEY from env. Do NOT hardcode keys in app.py.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def generate_answer(question: str, retrieved: list) -> str:
    """
    Professional RAG answer generator:
    - Uses ONLY top-3 retrieved context
    - Enforces refusal if context insufficient (via prompt + governance outside)
    - 4–6 bullets + citations line
    """
    retrieved = (retrieved or [])[:TOP_K]
    if not retrieved:
        return REFUSAL_LINE

    evidence = pack_context(retrieved)
    sources = ", ".join(sorted({r.get("source", "unknown") for r in retrieved}))

    client = get_openai_client()
    if client is None:
        # If OpenAI key missing, remain strict and safe.
        return extractive_fallback(retrieved)

    system_msg = (
        "You are a COE Enterprise Excellence assistant. "
        "You must answer ONLY using the Evidence provided. "
        f"If the Evidence does not contain the answer, reply exactly: {REFUSAL_LINE} "
        "Keep it crisp and professional."
    )

    user_msg = f"""
Question:
{question}

Evidence (Top 3 retrieved chunks):
{evidence}

Output format rules:
- Provide 4–6 bullet points MAX.
- Each bullet must be grounded in the Evidence.
- After bullets, add one final line exactly like:
  Citations: <comma-separated source filenames>
""".strip()

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        out = completion.choices[0].message.content or ""
    except Exception:
        return extractive_fallback(retrieved)

    out = _clean_llm_answer(out)

    # Normalize bullets and enforce citations
    lines = [ln.strip() for ln in out.split("\n") if ln.strip()]
    bullets = []
    citations_line = None

    for ln in lines:
        low = ln.lower()
        if low.startswith("citations:"):
            citations_line = ln
            continue

        # normalize bullets
        if ln.startswith(("-", "*")):
            ln = "• " + ln[1:].strip()
        elif not ln.startswith("•"):
            ln = "• " + ln

        bullets.append(ln)

    bullets = bullets[:6]

    # If the model refused, return exact refusal line (no extras)
    if out.strip() == REFUSAL_LINE:
        return REFUSAL_LINE

    if not bullets:
        return extractive_fallback(retrieved)

    if not citations_line:
        citations_line = f"Citations: {sources}"
    else:
        # keep citations anchored to top-3 sources
        citations_line = f"Citations: {sources}"

    return "\n".join(bullets).strip() + "\n\n" + citations_line


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="hero">
  <div class="hero-title">Mini COE Digital Enablement Prototype</div>
  <div class="hero-sub">Portfolio dashboard + chatbot-style RAG • strict governance • evidence grounded answers</div>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["📊 COE Dashboard", "🤖 COE AI Assistant (RAG)"])

# -----------------------------
# TAB 1: Dashboard
# -----------------------------
with tab1:
    df = load_and_clean_data(DATA_PATH)

    st.sidebar.markdown("## Filters")
    status_filter = st.sidebar.multiselect(
        "Status",
        sorted(df["Status"].dropna().unique().tolist()),
        default=sorted(df["Status"].dropna().unique().tolist()),
    )
    owner_filter = st.sidebar.multiselect(
        "Owner",
        sorted(df["Owner"].dropna().unique().tolist()),
        default=sorted(df["Owner"].dropna().unique().tolist()),
    )
    dept_filter = st.sidebar.multiselect(
        "Department",
        sorted(df["Department"].dropna().unique().tolist()),
        default=sorted(df["Department"].dropna().unique().tolist()),
    )

    dff = df[df["Status"].isin(status_filter) & df["Owner"].isin(owner_filter) & df["Department"].isin(dept_filter)]

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">COE Portfolio Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-muted">KPI tracking • portfolio health • business benefit • data quality checks</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    total = len(dff)
    pct_on_track = (dff["Status"].eq("On Track").mean() * 100) if total else 0
    pct_delayed = (dff["Status"].eq("Delayed").mean() * 100) if total else 0
    avg_kpi = float(dff["KPI_Achievement_Pct"].mean()) if total else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="card"><div class="card-title">Total initiatives</div><div class="card-value">{total}</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="card"><div class="card-title">% On track</div><div class="card-value">{pct_on_track:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="card"><div class="card-title">% Delayed</div><div class="card-value">{pct_delayed:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="card"><div class="card-title">Avg KPI achievement</div><div class="card-value">{avg_kpi:.1f}%</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Status distribution</div>', unsafe_allow_html=True)
        status_counts = dff["Status"].value_counts()
        fig = plt.figure()
        plt.bar(status_counts.index, status_counts.values)
        plt.xlabel("Status")
        plt.ylabel("Count")
        st.pyplot(fig)

    with right:
        st.markdown('<div class="section-title">Top initiatives by business benefit</div>', unsafe_allow_html=True)
        top3 = dff.sort_values("Business_Benefit_Value", ascending=False).head(3)
        fig2 = plt.figure()
        plt.bar(top3["Initiative_ID"], top3["Business_Benefit_Value"])
        plt.xlabel("Initiative")
        plt.ylabel("Benefit value")
        st.pyplot(fig2)

        st.dataframe(
            top3[
                [
                    "Initiative_ID",
                    "Initiative_Name",
                    "Owner",
                    "Status",
                    "Business_Benefit_Value",
                    "Business_Benefit_Unit",
                ]
            ],
            width="stretch",
        )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Data quality checks</div>', unsafe_allow_html=True)
    dq1, dq2, dq3 = st.columns(3)
    with dq1:
        st.markdown(
            f'<div class="card"><div class="card-title">Missing end dates</div><div class="card-value">{int(dff["Missing_End_Date"].sum())}</div></div>',
            unsafe_allow_html=True,
        )
    with dq2:
        st.markdown(
            f'<div class="card"><div class="card-title">Delay flag (heuristic)</div><div class="card-value">{int(dff["Delay_Flag"].sum())}</div></div>',
            unsafe_allow_html=True,
        )
    with dq3:
        miss_kpi = int(dff["KPI_Target"].isna().sum() + dff["KPI_Achieved"].isna().sum())
        st.markdown(
            f'<div class="card"><div class="card-title">Missing KPI values</div><div class="card-value">{miss_kpi}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Initiatives table</div>', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TAB 2: RAG Chatbot
# -----------------------------
with tab2:
    index, chunks, model = load_rag_assets()

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">COE AI Assistant (RAG)</div>', unsafe_allow_html=True)

    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    llm_label = f"OpenAI: {OPENAI_MODEL}" if api_key_present else "OpenAI: (missing key → safe fallback)"

    st.markdown(
        f'<div class="small-muted">LLM: <b>{llm_label}</b> • Retrieval: FAISS (Top-{TOP_K}) • Embeddings: {EMBED_MODEL_NAME}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    if index is None:
        st.error("RAG assets not found. Run: python rag_pipeline.py")
        st.code("python rag_pipeline.py")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        c1, c2 = st.columns([1.2, 1.8])
        with c1:
            show_sources = st.checkbox("Show retrieved text (Evidence)", value=True)
        with c2:
            strict_mode = st.checkbox("Strict governance (refuse if low similarity)", value=True)

        st.markdown('<div class="small-muted">Try these:</div>', unsafe_allow_html=True)
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        suggestions = [
            "What is Lean methodology?",
            "Explain DMAIC in Six Sigma",
            "How should COE track KPIs?",
            "What are common Lean wastes?",
        ]
        clicked = None
        for col, q in zip([qcol1, qcol2, qcol3, qcol4], suggestions):
            with col:
                if st.button(q, width="stretch"):
                    clicked = q

        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        # Chat transcript UI
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for m in st.session_state.rag_messages:
            if m["role"] == "user":
                st.markdown(
                    f'<div class="chat-row user"><div class="bubble user">{m["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                badge = badge_html(m.get("conf_label", "Low"))
                score = m.get("conf_score", 0.0)
                header = f'{badge}<span class="meta"> similarity {score:.2f}</span>'
                st.markdown(
                    f'<div class="chat-row bot"><div class="bubble bot">{header}\n\n{m["content"]}</div></div>',
                    unsafe_allow_html=True,
                )

                # Requirement: show retrieved text alongside the final answer
                if show_sources and m.get("sources"):
                    with st.expander("Evidence (Top 3 Retrieved Chunks)"):
                        for i, r in enumerate(m["sources"][:TOP_K], start=1):
                            st.markdown(
                                f'<div class="source-card"><b>{i}. {r["source"]}</b> '
                                f'<span class="meta">chunk {r["chunk_index"]} • sim {r["score"]:.2f}</span></div>',
                                unsafe_allow_html=True,
                            )
                            st.code(
                                (r["text"] or "")[:1600] + ("..." if len((r["text"] or "")) > 1600 else ""),
                                language="markdown",
                            )
        st.markdown("</div>", unsafe_allow_html=True)

        prompt = st.chat_input("Ask about Lean, Six Sigma, KPI tracking, COE governance…")
        if clicked and not prompt:
            prompt = clicked

        if prompt:
            st.session_state.rag_messages.append({"role": "user", "content": prompt})

            retrieved, overall = retrieve_top_k(index, chunks, model, prompt, k=TOP_K)
            label = confidence_label(overall)

            # Governance: REFUSE if low similarity
            if strict_mode and overall < SIM_MED:
                answer = REFUSAL_LINE
            else:
                answer = generate_answer(prompt, retrieved)

                # If answer isn't bulleted (or looks messy) and similarity isn't high, keep it safe
                if strict_mode and overall < SIM_HIGH and ("•" not in answer) and (answer != REFUSAL_LINE):
                    answer = extractive_fallback(retrieved)

            st.session_state.rag_messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "conf_label": label,
                    "conf_score": float(overall),  # basic confidence score
                    "sources": retrieved[:TOP_K],
                }
            )
            st.rerun()

        b1, b2 = st.columns([1, 2])
        with b1:
            if st.button("Clear chat", width="stretch"):
                st.session_state.rag_messages = []
                st.rerun()
        with b2:
            st.markdown(
                f'<div class="small-muted">Governance test: ask something unrelated (e.g., “Who is the CEO of Microsoft?”) — it should refuse below similarity {SIM_MED:.2f}.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

