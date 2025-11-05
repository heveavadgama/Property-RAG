# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from dateutil import parser as dateparser
import re
import faiss
import pickle
from io import BytesIO

# Embedding (sentence-transformers)
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# Optional OpenAI for generation
import openai

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good semantic embeddings
EMBED_DIM = 384  # all-MiniLM-L6-v2 dimension
TOP_K = 6
INDEX_PATH = "faiss_index.index"
META_PATH = "metadata.pkl"

# ---------------------------
# Helpers: cleaning & preprocessing
# ---------------------------
@st.cache_resource
def load_sentence_transformer(name=MODEL_NAME):
    return SentenceTransformer(name)

def clean_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    # remove html tags
    s = re.sub(r"<[^>]+>", " ", s)
    # common null tokens
    if s.strip().lower() in {"none", "null", "nan", ""}:
        return ""
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_price(x):
    if pd.isna(x):
        return np.nan
    try:
        if isinstance(x, str):
            # remove currency and commas
            x2 = re.sub(r"[^\d\.-]", "", x)
            return float(x2) if x2 != "" else np.nan
        return float(x)
    except Exception:
        return np.nan

def parse_date_safe(x):
    if pd.isna(x):
        return pd.NaT
    try:
        return dateparser.parse(str(x))
    except Exception:
        return pd.NaT

def normalize_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return False

def normalize_type(t):
    if pd.isna(t):
        return "unknown"
    s = str(t).strip().lower()
    # synonyms
    if s in {"apt", "apartment", "flat"}:
        return "apartment"
    if s in {"terraced", "terrace", "terraced house", "terraced_house", "terraced-house", "terrace house"}:
        return "terraced"
    if s in {"detached bungalow", "detachedbungalow", "detached bungalow"}:
        return "detached_bungalow"
    # fallback
    return re.sub(r"\s+", "_", s)

def normalize_flood_risk(x):
    if pd.isna(x):
        return "unknown"
    s = str(x).strip().lower()
    if s in {"high", "1", "h"}:
        return "high"
    if s in {"medium", "2", "m"}:
        return "medium"
    if s in {"low", "0", "l"}:
        return "low"
    return "unknown"

def synthesize_description(row):
    # make a fallback description to index if description missing
    parts = []
    typ = row.get("type_norm", "")
    beds = row.get("bedrooms")
    baths = row.get("bathrooms")
    price = row.get("price")
    laua = row.get("laua")
    desc = clean_text(row.get("property_type_full_description") or "")
    if desc:
        parts.append(desc)
    else:
        if beds is not None and not pd.isna(beds):
            parts.append(f"{int(beds)} bedroom")
        if baths is not None and not pd.isna(baths):
            parts.append(f"{int(baths)} bathroom")
        if typ:
            parts.append(typ.replace("_", " "))
        if price is not None and not pd.isna(price):
            parts.append(f"priced at {int(price)}")
    if laua:
        parts.append(f"local authority {laua}")
    return ", ".join(parts)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # standardize columns lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # common null tokens
    df = df.replace({"NULL": np.nan, "None": np.nan, "null": np.nan, "": np.nan})

    # normalize price
    if "price" in df.columns:
        df["price"] = df["price"].apply(normalize_price)

    # parse dates
    if "listing_update_date" in df.columns:
        df["listing_update_date"] = df["listing_update_date"].apply(parse_date_safe)

    # normalize booleans
    for col in ["is_new_home", "flood_risk"]:
        if col in df.columns:
            # flood_risk handled separately
            pass

    # normalize type
    if "type" in df.columns:
        df["type_norm"] = df["type"].apply(normalize_type)
    else:
        df["type_norm"] = "unknown"

    # flood risk normalization
    if "flood_risk" in df.columns:
        df["flood_risk_norm"] = df["flood_risk"].apply(normalize_flood_risk)
    else:
        df["flood_risk_norm"] = "unknown"

    # crime score: coerce to numeric and clip 0-10
    if "crime_score_weight" in df.columns:
        df["crime_score_weight"] = pd.to_numeric(df["crime_score_weight"], errors="coerce")
        # clip
        df["crime_score_weight"] = df["crime_score_weight"].clip(lower=0, upper=10)
    else:
        df["crime_score_weight"] = np.nan

    # laua to str
    if "laua" in df.columns:
        df["laua"] = df["laua"].astype(str)

    # clean description fields
    for col in ["property_type_full_description", "description"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # fill missing property_type_full_description with synthesized
    df["synth_description"] = df.apply(synthesize_description, axis=1)

    # create final text to embed
    def make_index_text(r):
        parts = []
        # put the human description first if present
        desc = r.get("property_type_full_description") or ""
        if desc and desc.strip():
            parts.append(desc)
        else:
            parts.append(r.get("synth_description", ""))

        parts.append(f"type: {r.get('type_norm','unknown')}")
        if r.get("bedrooms") is not None and not pd.isna(r.get("bedrooms")):
            parts.append(f"{int(r.get('bedrooms'))} bed")
        if r.get("bathrooms") is not None and not pd.isna(r.get("bathrooms")):
            parts.append(f"{int(r.get('bathrooms'))} bath")
        if r.get("price") is not None and not pd.isna(r.get("price")):
            parts.append(f"price: £{int(r.get('price'))}")
        parts.append(f"flood_risk: {r.get('flood_risk_norm','unknown')}")
        parts.append(f"crime_score: {r.get('crime_score_weight')}")
        # include address and laua for geosemantic cues
        if r.get("address"):
            parts.append(f"address: {r.get('address')}")
        if r.get("laua"):
            parts.append(f"laua: {r.get('laua')}")
        return " | ".join([p for p in parts if p])

    df["index_text"] = df.apply(make_index_text, axis=1)

    # small token-length field for diagnostics
    df["index_text_len"] = df["index_text"].apply(lambda s: len(str(s).split()))

    return df

# ---------------------------
# Embedding + FAISS index utilities
# ---------------------------
def build_faiss_index(embeddings: np.ndarray, dim: int = EMBED_DIM):
    # simple flat L2 index (works well up to a few hundred thousand points)
    index = faiss.IndexFlatIP(dim)  # use inner-product on normalized vectors for cosine
    # if not normalized we will normalize before adding
    return index

def make_embeddings(model, texts: list, batch_size=256):
    # returns normalized vectors (unit length) for cosine similarity with inner product index
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    # normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embs = embs / norms
    return embs

def save_index(index, path=INDEX_PATH):
    faiss.write_index(index, path)

def load_index(path=INDEX_PATH):
    return faiss.read_index(path)

# ---------------------------
# RAG prompt templates & generation
# ---------------------------
def query_to_openai_prompt(query:str, docs:list):
    """
    docs: list of dicts with keys ['address','price','bedrooms','bathrooms','type_norm','property_type_full_description','laua','flood_risk_norm','crime_score_weight']
    """
    ctx = []
    for i, d in enumerate(docs, start=1):
        ctx.append(f"### Property {i}\nAddress: {d.get('address')}\nType: {d.get('type_norm')}\nBeds: {d.get('bedrooms')}\nBaths: {d.get('bathrooms')}\nPrice: £{d.get('price')}\nFlood risk: {d.get('flood_risk_norm')}\nCrime score: {d.get('crime_score_weight')}\nDescription: {clean_text(d.get('property_type_full_description') or d.get('synth_description') or '')}\n")
    context_str = "\n\n".join(ctx)
    prompt = f"""
You are Estate Genie, an expert assistant for property discovery.
User query: "{query}"
Use the following candidate properties (most relevant first) and:
1) Provide a concise 2-3 line summary answering the user's intent.
2) Return a short ranked list (top 5) of the retrieved properties with one-line reasons for ranking.
3) Provide a small suggestions section: e.g., filters to narrow results (price/beds/type), or an action (contact agent/view map).
Do not hallucinate prices or addresses beyond the provided data.
Context:
{context_str}
Answer clearly and with bullet points or short paragraphs."""

    return prompt

def generate_openai_response(prompt, model="gpt-4o-mini", max_tokens=400):
    openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not openai_api_key:
        return "OpenAI API key not configured. Set OPENAI_API_KEY in environment or Streamlit secrets to enable AI summaries."
    openai.api_key = openai_api_key
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API error: {e}"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Estate Genie — RAG Property Search", layout="wide")
st.title("Estate Genie — RAG Property Search (Streamlit)")

# Sidebar: upload / load dataset
st.sidebar.header("Dataset & Index")
uploaded_file = st.sidebar.file_uploader("Upload Property CSV (Property_data.csv)", type=["csv"])
use_sample = st.sidebar.checkbox("Use local sample (if available)", value=False)
build_index_btn = st.sidebar.button("Build / Rebuild Index")
load_index_btn = st.sidebar.button("Load saved index (faiss & metadata)")

# Optionally specify index path
index_path_input = st.sidebar.text_input("Index path (faiss)", INDEX_PATH)
meta_path_input = st.sidebar.text_input("Metadata pickle path", META_PATH)

# Load dataset
@st.cache_data(show_spinner=False)
def load_df_from_buffer(buffer):
    df = pd.read_csv(buffer, low_memory=False)
    return df

df = None
if uploaded_file is not None:
    df = load_df_from_buffer(uploaded_file)
    st.sidebar.success("CSV uploaded.")
elif use_sample:
    # try to load Property_data.csv from current dir
    sample_path = "Property_data.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path, low_memory=False)
        st.sidebar.success(f"Loaded sample {sample_path}.")
    else:
        st.sidebar.warning("No local Property_data.csv found.")
else:
    st.sidebar.info("Upload the CSV to start. Use sample if you uploaded it to repo as Property_data.csv")

if df is not None:
    st.write(f"Dataset rows: {len(df)}")
    if st.checkbox("Show raw preview"):
        st.dataframe(df.head(10))

    # Preprocess
    with st.spinner("Preprocessing data..."):
        df_clean = preprocess_dataframe(df)
    st.success("Preprocessing complete.")

    # Index building
    model = load_sentence_transformer()
    index_obj = None
    metadata = None

    if os.path.exists(index_path_input) and os.path.exists(meta_path_input) and not build_index_btn and not load_index_btn:
        st.info("Found existing saved index and metadata. Click 'Load saved index' in sidebar to load.")
    if load_index_btn:
        try:
            index_obj = load_index(index_path_input)
            with open(meta_path_input, "rb") as fh:
                metadata = pickle.load(fh)
            st.success("Index and metadata loaded.")
        except Exception as e:
            st.error(f"Failed to load index/meta: {e}")

    if build_index_btn:
        # create index_texts
        texts = df_clean["index_text"].astype(str).tolist()
        st.info(f"Computing embeddings for {len(texts)} properties (this may take time)...")
        embs = make_embeddings(model, texts)
        # build faiss index
        st.info("Building FAISS index...")
        index = build_faiss_index(embs.shape[1])
        # Using inner-product requires normalized vectors
        index.add(embs.astype("float32"))
        # save metadata: keep relevant fields per doc in same order as embeddings
        metadata = df_clean[["address", "price", "bedrooms", "bathrooms", "type_norm",
                             "property_type_full_description","synth_description","laua",
                             "flood_risk_norm","crime_score_weight","listing_update_date"]].to_dict(orient="records")
        # persist
        save_index(index, index_path_input)
        with open(meta_path_input, "wb") as fh:
            pickle.dump(metadata, fh)
        st.success("Index built and saved.")
        index_obj = index

    # live querying
    if index_obj is not None and metadata is not None:
        st.markdown("---")
        st.header("Search")
        col1, col2 = st.columns([3,1])
        with col1:
            user_query = st.text_input("Ask in natural language (e.g. 'affordable 3-bedroom houses with garden near schools')", "")
            top_k = st.number_input("Top K results to retrieve", min_value=1, max_value=20, value=TOP_K)
            run_q = st.button("Search")
        with col2:
            # filters
            st.write("Quick Filters")
            min_price, max_price = st.slider("Price range (£)", 0, 100000, (0, 5000))
            bedrooms_filter = st.multiselect("Bedrooms", options=sorted(df_clean["bedrooms"].dropna().unique().astype(int).tolist()), default=[])
            prop_types = sorted(df_clean["type_norm"].dropna().unique().tolist())
            prop_type_filter = st.multiselect("Property type", options=prop_types, default=[])

        if run_q and user_query.strip():
            with st.spinner("Embedding query and retrieving..."):
                q_emb = make_embeddings(model, [user_query])
                # search
                D, I = index_obj.search(q_emb.astype("float32"), top_k)
                hits = []
                for score, idx in zip(D[0], I[0]):
                    if idx < 0 or idx >= len(metadata):
                        continue
                    md = metadata[idx].copy()
                    md["score"] = float(score)
                    hits.append(md)
                # apply quick filters in-app (post-retrieval)
                def pass_filters(h):
                    if h.get("price") is None:
                        p_ok = True
                    else:
                        p_ok = (min_price <= float(h.get("price") or 0) <= max_price)
                    if bedrooms_filter:
                        try:
                            b = int(h.get("bedrooms")) if h.get("bedrooms") is not None else None
                            if b not in bedrooms_filter:
                                return False
                        except Exception:
                            pass
                    if prop_type_filter:
                        if h.get("type_norm") not in prop_type_filter:
                            return False
                    return p_ok

                hits_filtered = [h for h in hits if pass_filters(h)]
                st.write(f"Retrieved {len(hits)} results, {len(hits_filtered)} after applying filters.")

                # show table
                if len(hits_filtered) > 0:
                    df_hits = pd.DataFrame(hits_filtered)
                    df_hits_display = df_hits[["address","price","bedrooms","bathrooms","type_norm","score","flood_risk_norm","crime_score_weight"]]
                    st.dataframe(df_hits_display)

                    # Call OpenAI to generate a short summary using the hits as context
                    prompt = query_to_openai_prompt(user_query, hits_filtered[:5])
                    generated = generate_openai_response(prompt)
                    st.subheader("AI Summary")
                    st.write(generated)
                else:
                    st.warning("No results after filters. Try widening price range or removing type/bedroom filters.")

        st.markdown("---")
        st.write("Index & metadata quick operations")
        colA, colB, colC = st.columns(3)
        with colA:
            if st.button("Download FAISS index (binary)"):
                try:
                    # write index to memory and download
                    tmp_path = index_path_input
                    faiss.write_index(index_obj, tmp_path)
                    with open(tmp_path, "rb") as fh:
                        st.download_button("Download index file", data=fh, file_name=os.path.basename(tmp_path))
                except Exception as e:
                    st.error(f"Failed to export index: {e}")
        with colB:
            if st.button("Download Metadata (pickle)"):
                try:
                    bio = BytesIO()
                    pickle.dump(metadata, bio)
                    bio.seek(0)
                    st.download_button("Download metadata", data=bio, file_name="metadata.pkl")
                except Exception as e:
                    st.error(f"Failed to export metadata: {e}")
        with colC:
            if st.button("Show index stats"):
                try:
                    st.write(index_obj.ntotal)
                except Exception as e:
                    st.error(f"Index error: {e}")

else:
    st.info("Please upload your dataset to start.")

st.markdown("---")
st.caption("Estate Genie — RAG example. Preprocessing handles Null/None/NULL tokens, normalizes types, parses dates, synthesizes missing descriptions, normalizes flood risk and crime scores, builds FAISS index, and generates LLM summaries via OpenAI.")
