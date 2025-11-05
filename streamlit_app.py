import streamlit as st
import pandas as pd
import numpy as np
import os, pickle, re, faiss
from datetime import datetime
from dateutil import parser as dateparser
from sentence_transformers import SentenceTransformer
import openai

# ==================================================
# CONFIG
# ==================================================
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index.index"
META_PATH = "metadata.pkl"
EMBED_DIM = 384
TOP_K = 6

# ==================================================
# SESSION STATE INIT
# ==================================================
if "index_obj" not in st.session_state:
    st.session_state["index_obj"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None
if "df_clean" not in st.session_state:
    st.session_state["df_clean"] = None

# ==================================================
# HELPERS: CLEANING & NORMALIZATION
# ==================================================
def clean_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return ""
    s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)
    if s.strip().lower() in {"none", "null", "nan", ""}: return ""
    return re.sub(r"\s+", " ", s).strip()

def normalize_price(x):
    try:
        if isinstance(x, str):
            x = re.sub(r"[^\d\.-]", "", x)
        return float(x)
    except: return np.nan

def parse_date_safe(x):
    try: return dateparser.parse(str(x))
    except: return pd.NaT

def normalize_type(t):
    if pd.isna(t): return "unknown"
    s = str(t).lower().strip()
    mapping = {
        "apt":"apartment", "apartment":"apartment", "flat":"apartment",
        "terraced":"terraced", "terrace":"terraced", "terraced house":"terraced"
    }
    return mapping.get(s, s.replace(" ", "_"))

def preprocess_dataframe(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.replace({"NULL":np.nan,"None":np.nan,"":"nan"})
    df["price"] = df["price"].apply(normalize_price)
    if "listing_update_date" in df.columns:
        df["listing_update_date"] = df["listing_update_date"].apply(parse_date_safe)
    df["type_norm"] = df["type"].apply(normalize_type)

    if "property_type_full_description" in df.columns:
        df["property_type_full_description"] = df["property_type_full_description"].apply(clean_text)
    else:
        df["property_type_full_description"] = ""

    def synth_desc(r):
        parts=[]
        if r["bedrooms"] and not pd.isna(r["bedrooms"]): parts.append(f"{int(r['bedrooms'])} bed")
        if r["bathrooms"] and not pd.isna(r["bathrooms"]): parts.append(f"{int(r['bathrooms'])} bath")
        if r["type_norm"]: parts.append(r["type_norm"])
        if r["price"]: parts.append(f"price £{int(r['price'])}")
        if "laua" in r and r["laua"]: parts.append(f"laua {r['laua']}")
        return ", ".join(parts)

    df["synth_description"] = df.apply(synth_desc, axis=1)
    def make_text(r):
        d = r["property_type_full_description"] or r["synth_description"]
        return f"{d} | {r['type_norm']} | {r['bedrooms']} beds | £{r['price']} | {r['address']} | crime:{r.get('crime_score_weight')} | flood:{r.get('flood_risk')}"
    df["index_text"] = df.apply(make_text, axis=1)
    return df

# ==================================================
# CACHE MODEL / INDEX / METADATA
# ==================================================
@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_index_cached(path=INDEX_PATH):
    return faiss.read_index(path)

@st.cache_resource(show_spinner=False)
def load_metadata_cached(path=META_PATH):
    with open(path, "rb") as fh:
        return pickle.load(fh)

def make_embeddings(model, texts):
    embs = model.encode(texts, batch_size=512, convert_to_numpy=True, show_progress_bar=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-9, None)
    return embs.astype("float32")

def build_faiss_index(dim):
    return faiss.IndexFlatIP(dim)

# ==================================================
# OPENAI RESPONSE
# ==================================================
def query_to_openai_prompt(query, docs):
    ctx = []
    for i, d in enumerate(docs[:5], start=1):
        ctx.append(f"{i}. {d.get('address')} — £{d.get('price')}, {d.get('bedrooms')} bed, {d.get('bathrooms')} bath, {d.get('type_norm')}.")
    context = "\n".join(ctx)
    return f"""User query: "{query}"
Context properties:
{context}
Provide:
1. 2–3 line summary answering the user's intent.
2. A ranked top 5 list with brief justifications.
3. Suggestions for filters or next actions.
"""

def generate_openai_response(prompt):
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return "⚠️ OpenAI API key missing. Add in Streamlit Secrets."
    openai.api_key = api_key
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=400
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from OpenAI API: {e}"

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(page_title="🏡 Estate Genie — Property RAG", layout="wide")
st.title("🏡 Estate Genie — AI Property Search")

st.sidebar.header("⚙️ Setup")
uploaded_file = st.sidebar.file_uploader("Upload Property CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use local Property_data.csv", value=False)
build_index_btn = st.sidebar.button("🧠 Build / Rebuild Index")
load_index_btn = st.sidebar.button("📦 Load Saved Index")

index_path_input = st.sidebar.text_input("Index file", INDEX_PATH)
meta_path_input = st.sidebar.text_input("Metadata file", META_PATH)

# ==================================================
# LOAD DATASET
# ==================================================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully.")
elif use_sample and os.path.exists("Property_data.csv"):
    df = pd.read_csv("Property_data.csv")
    st.sidebar.success("✅ Loaded local Property_data.csv.")
else:
    df = None
    st.sidebar.info("Upload your dataset to start.")

if df is not None:
    st.write(f"📊 Dataset rows: {len(df)}")
    if st.checkbox("Show raw data preview"):
        st.dataframe(df.head(10))

    if st.session_state["df_clean"] is None:
        with st.spinner("Cleaning and preprocessing data..."):
            st.session_state["df_clean"] = preprocess_dataframe(df)
        st.success("✅ Preprocessing complete.")

# ==================================================
# AUTO-LOAD INDEX IF AVAILABLE
# ==================================================
if (st.session_state["index_obj"] is None and
    os.path.exists(index_path_input) and os.path.exists(meta_path_input)):
    with st.spinner("Auto-loading saved FAISS index..."):
        st.session_state["index_obj"] = load_index_cached(index_path_input)
        st.session_state["metadata"] = load_metadata_cached(meta_path_input)
    st.success("✅ Index auto-loaded.")

# ==================================================
# BUILD INDEX BUTTON
# ==================================================
if build_index_btn and st.session_state["df_clean"] is not None:
    model = load_sentence_transformer()
    texts = st.session_state["df_clean"]["index_text"].tolist()
    with st.spinner("🔎 Encoding properties..."):
        embs = make_embeddings(model, texts)
    with st.spinner("📦 Building FAISS index..."):
        index = build_faiss_index(embs.shape[1])
        index.add(embs)
        faiss.write_index(index, index_path_input)
        metadata = st.session_state["df_clean"][[
            "address","price","bedrooms","bathrooms","type_norm",
            "property_type_full_description","synth_description","laua",
            "flood_risk","crime_score_weight","listing_update_date"
        ]].to_dict(orient="records")
        with open(meta_path_input, "wb") as f: pickle.dump(metadata, f)
    st.session_state["index_obj"] = index
    st.session_state["metadata"] = metadata
    st.success("✅ Index built and saved successfully.")

# ==================================================
# LOAD INDEX BUTTON
# ==================================================
if load_index_btn:
    try:
        st.session_state["index_obj"] = load_index_cached(index_path_input)
        st.session_state["metadata"] = load_metadata_cached(meta_path_input)
        st.success("✅ Index loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load index/meta: {e}")

# ==================================================
# QUERY SECTION
# ==================================================
index_obj = st.session_state["index_obj"]
metadata = st.session_state["metadata"]

if index_obj and metadata:
    st.markdown("---")
    st.header("🔍 Search Properties")

    user_query = st.text_input("Ask a property query:", "")
    top_k = st.slider("Number of results", 3, 20, TOP_K)
    run_q = st.button("🚀 Search")

    if run_q and user_query.strip():
        with st.spinner("Running semantic search..."):
            model = load_sentence_transformer()
            q_emb = make_embeddings(model, [user_query])
            D, I = index_obj.search(q_emb, top_k)
            hits = []
            for score, idx in zip(D[0], I[0]):
                if idx < len(metadata):
                    d = metadata[idx].copy()
                    d["score"] = float(score)
                    hits.append(d)

        if hits:
            df_hits = pd.DataFrame(hits)
            st.dataframe(df_hits[["address","price","bedrooms","bathrooms","type_norm","score"]])

            with st.spinner("🧠 Generating AI summary..."):
                prompt = query_to_openai_prompt(user_query, hits)
                summary = generate_openai_response(prompt)
            st.subheader("✨ AI Summary")
            st.write(summary)
        else:
            st.warning("No results found for your query.")
else:
    st.info("Please build or load an index first to start searching.")

st.markdown("---")
st.caption("💡 Estate Genie — Cached FAISS + Streamlit state persistence. Upload once, build index once, and query freely without resets.")
