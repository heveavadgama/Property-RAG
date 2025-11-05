# streamlit_app.py
"""
Estate Genie — Improved RAG Streamlit App (hybrid retrieval + robust NLP)
- Handles many language anomalies for property types and numeric price hints.
- Uses SentenceTransformers + FAISS for semantic retrieval.
- Applies structured filters after semantic retrieval to ensure strict constraints (price, beds, baths, types).
- Caches heavy objects and persists index/metadata across reruns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, re, pickle, faiss
from datetime import datetime
from dateutil import parser as dateparser
from sentence_transformers import SentenceTransformer

# Optional OpenAI (used for summaries if API key present)
import openai

# Try to import spaCy; fallback if not installed
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    _nlp = None

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index.index"
META_PATH = "metadata.pkl"
DEFAULT_TOP_K = 10
EMBED_DIM = 384

# ----------------------------
# Initialize session_state slots
# ----------------------------
if "index_obj" not in st.session_state:
    st.session_state["index_obj"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None
if "df_clean" not in st.session_state:
    st.session_state["df_clean"] = None

# ----------------------------
# Property synonyms mapping (regex patterns -> normalized token)
# Add more patterns here if you encounter other variants.
# ----------------------------
PROPERTY_PATTERNS = [
    (r"\bend[-_\s]?terrace(d)?\b", "end_of_terrace"),
    (r"\bend[-_\s]?of[-_\s]?terrace(d)?\b", "end_of_terrace"),
    (r"\bmid[-_\s]?terrace(d)?\b", "mid_terrace"),
    (r"\bterrace(d)?\b", "terraced"),
    (r"\bterraced[-_\s]?house\b", "terraced"),
    (r"\bflat\b|\bapt\b|\bapartment\b", "apartment"),
    (r"\bsemi[-_\s]?detached\b", "semi_detached"),
    (r"\bdetached\b", "detached"),
    (r"\bbungalow\b", "detached_bungalow"),
    (r"\bcottage\b", "cottage"),
    (r"\bstudio\b", "studio"),
    (r"\btown[-_\s]?house\b", "townhouse"),
    (r"\bmaisonette\b", "maisonette"),
    (r"\bhouse\b", "house"),
]

# ----------------------------
# Utilities: cleaning & normalization
# ----------------------------
def clean_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)  # remove HTML-like
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = s.strip()
    if s.lower() in {"none", "null", "nan", ""}:
        return ""
    return re.sub(r"\s+", " ", s)

def normalize_price_value(x):
    """Convert string/number to float price in same currency units (assumes GBP numeric)."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x).lower().strip()
    # handle "1.5k", "1k", "1500", "£1,500"
    s = s.replace("£", "").replace(",", "").replace(" ", "")
    # convert 'k' to thousands
    m = re.match(r"^([0-9]*\.?[0-9]+)k$", s)
    if m:
        try:
            return float(m.group(1)) * 1000.0
        except:
            return np.nan
    # remove non-numeric except dot and minus
    s2 = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s2) if s2 != "" else np.nan
    except:
        return np.nan

def parse_date_safe(x):
    try:
        return dateparser.parse(str(x))
    except Exception:
        return pd.NaT

def normalize_type_value(t):
    if pd.isna(t):
        return "unknown"
    s = str(t).lower().strip()
    # apply pattern mapping
    for pat, repl in PROPERTY_PATTERNS:
        if re.search(pat, s):
            return repl
    # fallback replace spaces with underscore
    return re.sub(r"\s+", "_", s)

# ----------------------------
# Preprocessing DataFrame
# ----------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # unify null tokens
    df = df.replace({"NULL": np.nan, "None": np.nan, "null": np.nan, "": np.nan})
    if "price" in df.columns:
        df["price"] = df["price"].apply(normalize_price_value)
    if "listing_update_date" in df.columns:
        df["listing_update_date"] = df["listing_update_date"].apply(parse_date_safe)
    # normalize type
    if "type" in df.columns:
        df["type_norm"] = df["type"].apply(normalize_type_value)
    else:
        df["type_norm"] = "unknown"
    # clean description fields
    if "property_type_full_description" in df.columns:
        df["property_type_full_description"] = df["property_type_full_description"].apply(clean_text)
    else:
        df["property_type_full_description"] = ""
    # synth desc
    def synth_desc(r):
        parts = []
        try:
            if not pd.isna(r.get("bedrooms")):
                parts.append(f"{int(r.get('bedrooms'))} bed")
        except:
            pass
        try:
            if not pd.isna(r.get("bathrooms")):
                parts.append(f"{int(r.get('bathrooms'))} bath")
        except:
            pass
        if r.get("type_norm"):
            parts.append(r["type_norm"].replace("_", " "))
        if not pd.isna(r.get("price")):
            try:
                parts.append(f"price £{int(r.get('price'))}")
            except:
                pass
        if r.get("laua"):
            parts.append(f"laua {r.get('laua')}")
        return ", ".join(parts)
    df["synth_description"] = df.apply(synth_desc, axis=1)
    # create index text for embeddings
    def make_index_text(r):
        desc = r.get("property_type_full_description") or r.get("synth_description") or ""
        parts = [desc,
                 f"type: {r.get('type_norm')}",
                 f"beds: {r.get('bedrooms')}",
                 f"baths: {r.get('bathrooms')}",
                 f"price: £{r.get('price')}" if not pd.isna(r.get('price')) else "",
                 f"laua: {r.get('laua')}" if r.get("laua") else ""]
        return " | ".join([p for p in parts if p])
    df["index_text"] = df.apply(make_index_text, axis=1)
    return df

# ----------------------------
# Load / cache heavy objects
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_sentence_transformer(name=MODEL_NAME):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_faiss_index(path=INDEX_PATH):
    return faiss.read_index(path)

@st.cache_resource(show_spinner=False)
def load_metadata(path=META_PATH):
    with open(path, "rb") as fh:
        return pickle.load(fh)

def save_index(index, path=INDEX_PATH):
    faiss.write_index(index, path)

# ----------------------------
# Embedding utility
# ----------------------------
def make_embeddings(model, texts, batch_size=512):
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embs = embs / norms
    return embs.astype("float32")

# ----------------------------
# Query normalization & filter extraction
# ----------------------------
def normalize_query(query: str) -> str:
    """Normalize query: expand k -> thousands, map synonyms with PROPERTY_PATTERNS, clean punctuation,
       and lemmatize if spaCy available."""
    q = query.lower().strip()
    # Expand K shorthand: 1.5k -> 1500
    q = re.sub(r"(\d+(?:\.\d+)?)\s*k\b", lambda m: str(float(m.group(1)) * 1000), q)
    # replace currency symbols and commas for number parsing later (but keep for text)
    q = q.replace("£", "").replace(",", "")
    # map property synonyms using patterns
    for pat, repl in PROPERTY_PATTERNS:
        q = re.sub(pat, repl, q)
    # lemmatize if spaCy available
    if SPACY_AVAILABLE and _nlp:
        doc = _nlp(q)
        q = " ".join([tok.lemma_ for tok in doc])
    # collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()
    return q

def extract_filters_from_query(q: str):
    """Return dict with price_limit, beds, baths, types (list). Works with common language forms."""
    price_limit = None
    beds = None
    baths = None
    types = []

    # price expressions: under 1500, <1500, less than 1500, below 1500, upto 1500, <=1500
    m = re.search(r"(?:under|below|less than|<|<=|up to|upto)\s*£?(\d+(?:\.\d+)?)", q)
    if m:
        price_limit = float(m.group(1))

    # also catch "between x and y" and prefer upper bound if present
    m2 = re.search(r"between\s*£?(\d+(?:\.\d+)?)\s*(?:and|-)\s*£?(\d+(?:\.\d+)?)", q)
    if m2:
        try:
            price_limit = float(m2.group(2))
        except:
            pass

    # extract forms like "<1500"
    m3 = re.search(r"<\s*£?(\d+(?:\.\d+)?)", q)
    if m3:
        price_limit = float(m3.group(1))

    # bedroom / bath extraction
    m_bed = re.search(r"(\d+)\s*(?:bed|bedroom|bedrooms)\b", q)
    if m_bed:
        beds = int(m_bed.group(1))
    m_bath = re.search(r"(\d+)\s*(?:bath|bathroom|bathrooms)\b", q)
    if m_bath:
        baths = int(m_bath.group(1))

    # property types — match normalized tokens directly (from PROPERTY_PATTERNS replacement)
    for _, repl in PROPERTY_PATTERNS:
        if repl in q:
            types.append(repl)
    # dedupe
    types = list(dict.fromkeys(types))
    return {"price": price_limit, "beds": beds, "baths": baths, "types": types}

# ----------------------------
# Structured filter application
# ----------------------------
def apply_structured_filters(hits, filters, sidebar_filters):
    """
    hits: list of metadata dicts (from FAISS)
    filters: parsed from query (price/beds/baths/types)
    sidebar_filters: dict from UI (max_price, bedrooms_sel, types_sel)
    """
    filtered = []
    for h in hits:
        # price check
        price_val = h.get("price")
        if filters.get("price") is not None:
            if price_val is None or price_val > filters["price"]:
                continue
        if sidebar_filters.get("max_price") is not None:
            if price_val is None or price_val > sidebar_filters["max_price"]:
                continue
        # beds
        if filters.get("beds") is not None:
            try:
                if int(h.get("bedrooms") or 0) != filters["beds"]:
                    continue
            except:
                continue
        if sidebar_filters.get("bedrooms") and len(sidebar_filters["bedrooms"]) > 0:
            try:
                if int(h.get("bedrooms") or -1) not in sidebar_filters["bedrooms"]:
                    continue
            except:
                continue
        # baths
        if filters.get("baths") is not None:
            try:
                if int(h.get("bathrooms") or 0) != filters["baths"]:
                    continue
            except:
                continue
        if sidebar_filters.get("bathrooms") and len(sidebar_filters["bathrooms"]) > 0:
            try:
                if int(h.get("bathrooms") or -1) not in sidebar_filters["bathrooms"]:
                    continue
            except:
                continue
        # types (either from query or sidebar)
        type_norm = str(h.get("type_norm") or "").lower()
        q_types = filters.get("types") or []
        sidebar_types = sidebar_filters.get("types") or []
        # if query demanded some type, ensure match
        if q_types:
            if not any(t in type_norm for t in q_types):
                continue
        # if user selected types in sidebar, ensure match
        if sidebar_types:
            if type_norm not in sidebar_types:
                continue
        filtered.append(h)
    return filtered

# ----------------------------
# OpenAI prompt and generation (optional)
# ----------------------------
def build_openai_prompt(query, hits):
    ctx = []
    for i, d in enumerate(hits[:5], 1):
        ctx.append(f"{i}. {d.get('address')} — £{d.get('price')}, {d.get('bedrooms')} bed / {d.get('bathrooms')} bath — {d.get('type_norm')}")
    ctx_str = "\n".join(ctx)
    prompt = f"""
You are Estate Genie, an expert property assistant.
User query: "{query}"
Candidate properties (most relevant first):
{ctx_str}

Please:
1) Give a concise 2-line summary answering user's intent.
2) Provide a ranked top-5 list with one-line reasons.
3) Suggest filters or next actions.
Be factual and don't invent details beyond what's provided.
"""
    return prompt

def generate_openai_summary(prompt):
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return "OpenAI key not configured. Add OPENAI_API_KEY to environment or Streamlit secrets to enable AI summaries."
    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API error: {e}"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Estate Genie 2.0", layout="wide")
st.title("🏡 Estate Genie — Improved RAG Search")

# SIDEBAR: dataset, index controls, and filters
with st.sidebar:
    st.header("Data & Index")
    uploaded_file = st.file_uploader("Upload Property CSV (Property_data.csv)", type=["csv"])
    use_local = st.checkbox("Use local Property_data.csv if available", value=True)
    build_index_btn = st.button("🔁 Build / Rebuild Index")
    load_index_btn = st.button("📦 Load Index")
    st.text_input("Index path", INDEX_PATH, key="index_path")
    st.text_input("Metadata path", META_PATH, key="meta_path")
    st.markdown("---")
    st.header("Filters (Sidebar)")
    max_price = st.slider("Max price (£)", 0, 10000, 3000, step=50)
    bed_options = []  # will be filled below after loading df
    bathroom_options = []
    type_options = []
    st.markdown("---")
    st.write("Advanced")
    use_openai = st.checkbox("Enable AI Summaries (OpenAI)", value=False)
    st.caption("If enabled, set OPENAI_API_KEY in Streamlit secrets.")

# LOAD dataset into df_clean
df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
        st.sidebar.success("CSV uploaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
elif use_local and os.path.exists("Property_data.csv"):
    df = pd.read_csv("Property_data.csv", low_memory=False)
    st.sidebar.success("Loaded local Property_data.csv.")
else:
    st.sidebar.info("Upload CSV or add Property_data.csv to the app folder.")

if df is not None and st.session_state["df_clean"] is None:
    with st.spinner("Preprocessing dataset..."):
        st.session_state["df_clean"] = preprocess_dataframe(df)
    st.success("Preprocessing complete.")

# populate sidebar select options after preprocess
if st.session_state["df_clean"] is not None:
    df_clean = st.session_state["df_clean"]
    # bedroom options
    try:
        bed_options = sorted([int(x) for x in df_clean["bedrooms"].dropna().unique()])
    except:
        bed_options = sorted([int(x) for x in df_clean["bedrooms"].dropna().astype(int).unique()]) if "bedrooms" in df_clean.columns else []
    try:
        bathroom_options = sorted([int(x) for x in df_clean["bathrooms"].dropna().unique()])
    except:
        bathroom_options = sorted([int(x) for x in df_clean["bathrooms"].dropna().astype(int).unique()]) if "bathrooms" in df_clean.columns else []
    type_options = sorted(df_clean["type_norm"].dropna().unique().tolist())

    # overwrite the sidebar placeholders by using session_state keys
    st.sidebar.multiselect("Bedrooms", options=bed_options, key="sidebar_bedrooms")
    st.sidebar.multiselect("Bathrooms", options=bathroom_options, key="sidebar_bathrooms")
    st.sidebar.multiselect("Property types", options=type_options, key="sidebar_types")

# Auto-load index if files exist and not loaded
index_path = st.session_state.get("index_path", INDEX_PATH)
meta_path = st.session_state.get("meta_path", META_PATH)
if st.session_state["index_obj"] is None and os.path.exists(index_path) and os.path.exists(meta_path):
    try:
        with st.spinner("Auto-loading saved index & metadata..."):
            st.session_state["index_obj"] = load_faiss_index(index_path)
            st.session_state["metadata"] = load_metadata(meta_path)
        st.success("Index auto-loaded from disk.")
    except Exception as e:
        st.error(f"Auto-load failed: {e}")

# Build index flow
if build_index_btn:
    if st.session_state["df_clean"] is None:
        st.error("No dataset loaded to build index. Upload CSV first.")
    else:
        model = load_sentence_transformer()
        texts = st.session_state["df_clean"]["index_text"].astype(str).tolist()
        with st.spinner("Computing embeddings... (this may take a while for large datasets)"):
            embs = make_embeddings(model, texts)
        with st.spinner("Building FAISS index..."):
            idx = faiss.IndexFlatIP(embs.shape[1])
            idx.add(embs)
            try:
                save_index(idx, index_path)
                metadata = st.session_state["df_clean"][[
                    "address", "price", "bedrooms", "bathrooms", "type_norm",
                    "property_type_full_description", "synth_description", "laua",
                    "flood_risk", "crime_score_weight", "listing_update_date"
                ]].to_dict(orient="records")
                with open(meta_path, "wb") as fh:
                    pickle.dump(metadata, fh)
                st.session_state["index_obj"] = idx
                st.session_state["metadata"] = metadata
                st.success("Index built and saved.")
            except Exception as e:
                st.error(f"Failed to save index/metadata: {e}")

# Load index button
if load_index_btn:
    try:
        st.session_state["index_obj"] = load_faiss_index(index_path)
        st.session_state["metadata"] = load_metadata(meta_path)
        st.success("Index loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load index/meta: {e}")

# Main search UI
st.markdown("---")
st.header("Search properties")
query_input = st.text_input("Ask a natural language query (e.g., '2 bed terraced under 1500 near schools')", value="")
num_results = st.slider("Number of results to display", 3, 30, DEFAULT_TOP_K)
search_btn = st.button("🔎 Search")

index_obj = st.session_state.get("index_obj")
metadata = st.session_state.get("metadata")
sidebar_filters = {
    "max_price": max_price,
    "bedrooms": st.session_state.get("sidebar_bedrooms", []),
    "bathrooms": st.session_state.get("sidebar_bathrooms", []),
    "types": st.session_state.get("sidebar_types", []),
}

if search_btn:
    if index_obj is None or metadata is None:
        st.error("No index loaded. Build or load index first.")
    else:
        q_norm = normalize_query(query_input)
        filters = extract_filters_from_query(q_norm)
        model = load_sentence_transformer()
        with st.spinner("Embedding query and retrieving candidates..."):
            q_emb = make_embeddings(model, [q_norm])
            # retrieve a larger set to allow structured filtering
            retrieve_k = max(200, num_results * 5)
            D, I = index_obj.search(q_emb, retrieve_k)
            hits = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(metadata):
                    continue
                rec = dict(metadata[idx])
                rec["score"] = float(score)
                hits.append(rec)
        # apply structured + sidebar filters
        hits_after = apply_structured_filters(hits, filters, sidebar_filters)
        # sort by score desc then price asc as secondary heuristic
        hits_after = sorted(hits_after, key=lambda h: (-h.get("score", 0), h.get("price") or 1e12))
        hits_display = hits_after[:num_results]

        st.markdown(f"### Results — {len(hits_after)} match(es) after filters (showing top {len(hits_display)})")
        if len(hits_display) == 0:
            st.warning("No matches. Try widening your filters or removing strict numeric constraints.")
        else:
            # display cards in grid
            cols = st.columns(2)
            for i, rec in enumerate(hits_display):
                col = cols[i % 2]
                with col:
                    # determine if rec fully matches query filters (for highlight)
                    full_ok = True
                    if filters.get("price") is not None and (rec.get("price") is None or rec["price"] > filters["price"]):
                        full_ok = False
                    if filters.get("beds") is not None and int(rec.get("bedrooms") or 0) != filters["beds"]:
                        full_ok = False
                    if filters.get("baths") is not None and int(rec.get("bathrooms") or 0) != filters["baths"]:
                        full_ok = False
                    if filters.get("types"):
                        if not any(t in (rec.get("type_norm") or "") for t in filters["types"]):
                            full_ok = False
                    # color
                    card_bg = "#0b6623" if full_ok else "#222"
                    text_color = "white"
                    st.markdown(
                        f"""
                        <div style="background:{card_bg}; padding:12px; border-radius:8px; margin-bottom:8px;">
                          <div style="font-size:16px; font-weight:700; color:{text_color}">{rec.get('address') or '—'}</div>
                          <div style="color:{text_color}">£{int(rec.get('price')) if rec.get('price') is not None else 'N/A'} • {int(rec.get('bedrooms') or 0)} bed • {int(rec.get('bathrooms') or 0)} bath • {rec.get('type_norm')}</div>
                          <div style="margin-top:6px; color:{text_color}; font-size:13px;">Score: {rec.get('score'):.4f} • Flood: {rec.get('flood_risk') or 'unknown'} • Crime: {rec.get('crime_score_weight') or 'N/A'}</div>
                          <div style="margin-top:6px; color:{text_color}; font-size:13px;">{rec.get('property_type_full_description') or rec.get('synth_description')}</div>
                        </div>
                        """, unsafe_allow_html=True)

        # AI summary if enabled
        if use_openai:
            with st.spinner("Generating AI summary..."):
                prompt = build_openai_prompt(query_input, hits_display)
                summary = generate_openai_summary(prompt)
                st.markdown("#### AI Summary")
                st.write(summary)

# Footer: glossary & tips
with st.expander("🏠 Property Type Glossary (UK)"):
    st.markdown("""
    **end_of_terrace** — house at the end of a row of terraced houses (shares only one side wall).  
    **mid_terrace / terraced** — houses in the middle of a terrace (share two side walls).  
    **semi_detached** — two houses joined by a single wall (pair).  
    **detached** — standalone house, no shared walls.  
    **detached_bungalow** — single-storey detached property.  
    **apartment / flat** — single residential unit inside a larger building.  
    **studio** — one-room apartment with combined living/sleeping/kitchen space.
    """)

st.caption("Estate Genie 2.0 — hybrid semantic + structured retrieval. Preprocesses text, normalizes synonyms (terrace/terraced), parses numeric constraints, and filters results strictly. For large datasets, build the FAISS index offline and store faiss_index.index & metadata.pkl in the app folder or cloud storage.")
