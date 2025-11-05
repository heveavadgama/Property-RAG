# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
from typing import List, Tuple
import re
from dotenv import load_dotenv
from openai import OpenAI

# --- Page Config ---
st.set_page_config(page_title="Estate Genie", layout="wide", page_icon="🧞‍♂️")

# --- OpenAI key ---
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key not found. Add it to Streamlit secrets or .env.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Helper Functions
# -------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embed_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(name)

@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns, create a reliable lowercase 'description' field,
    coerce numeric columns, and trim addresses.
    """
    df = df.copy()

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # helper that returns a string series or empty strings
    def series_or_empty(col: str) -> pd.Series:
        if col in df.columns:
            return df[col].astype(str).fillna("")
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    # pick best available description-like column
    if "property_type_full_description" in df.columns:
        base_desc = df["property_type_full_description"].astype(str).fillna("")
    elif "description" in df.columns:
        base_desc = df["description"].astype(str).fillna("")
    else:
        base_desc = series_or_empty("")

    type_part = series_or_empty("type")

    # create normalized description and lower-case it for reliable matching
    desc = (base_desc.str.strip() + " | " + type_part.str.strip()).str.strip(" |")
    # remove literal 'None'/'NULL' strings and collapse whitespace
    desc = desc.replace(r"\b(None|NULL|null)\b", "", regex=True, case=False).str.strip().str.lower()
    df["description"] = desc

    # Numeric coercion
    for c in ["price", "bedrooms", "bathrooms", "crime_score_weight"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Address cleanup
    if "address" in df.columns:
        df["address"] = df["address"].astype(str).str.strip()

    df.reset_index(drop=True, inplace=True)
    return df

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    # faiss expects raw vectors; we already normalized with sklearn.normalize where used
    index.add(embeddings)
    return index

def save_index_and_meta(index: faiss.IndexFlatIP, meta_list: list,
                        filepath_index: str = "faiss.index",
                        filepath_meta: str = "meta.pkl") -> None:
    faiss.write_index(index, filepath_index)
    with open(filepath_meta, "wb") as f:
        pickle.dump(meta_list, f)

def load_index_and_meta(filepath_index: str = "faiss.index",
                        filepath_meta: str = "meta.pkl"):
    if not (os.path.exists(filepath_index) and os.path.exists(filepath_meta)):
        return None, None
    index = faiss.read_index(filepath_index)
    with open(filepath_meta, "rb") as f:
        meta_list = pickle.load(f)
    return index, meta_list

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    # model.encode returns numpy array; normalize and convert to float32
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    emb = normalize(emb, norm="l2")
    return emb.astype("float32")

# Analytic query handling
def handle_analytic_query(df: pd.DataFrame, q: str) -> Tuple[str, pd.DataFrame]:
    q_l = q.lower()

    m = re.search(r"average price of\s+(\d+)[ -]?(?:bed|bedroom|bedrooms)\b", q_l)
    if m:
        beds = int(m.group(1))
        sub = df[df["bedrooms"] == beds] if "bedrooms" in df.columns else pd.DataFrame()
        if len(sub):
            avg = sub["price"].dropna().mean()
            if pd.notna(avg):
                return (f"The average price of {beds}-bedroom homes, based on {len(sub)} listings, is **${avg:,.2f}**.",
                        sub)
        return (f"I couldn't find any listings for {beds}-bedroom homes to calculate the average price.",
                pd.DataFrame())

    if "most crime" in q_l or "highest crime" in q_l:
        if "crime_score_weight" in df.columns and "address" in df.columns:
            tmp = (df.dropna(subset=["address"])
                     .groupby("address")["crime_score_weight"]
                     .mean()
                     .sort_values(ascending=False))
            if len(tmp):
                top_address = tmp.index[0]
                top_score = tmp.iloc[0]
                return (f"The area with the highest average crime score is **{top_address}** with a score of {top_score:.2f}.",
                        df[df["address"] == top_address])
        return ("The dataset does not contain a usable 'crime_score_weight' column to answer this question.",
                pd.DataFrame())

    return None, None

# Query parsing and keyword normalization
def normalize_query_keywords(text: str) -> List[str]:
    """
    Broaden patterns to include surface forms like 'terraced' and 'balconies'.
    Returns list of normalized keys found in the query.
    """
    text_l = text.lower()
    patterns = {
        "terrace": r"\b(terrace|terraced|roof terrace|private terrace)\b",
        "balcony": r"\b(balcony|balconies)\b",
        "garden": r"\b(garden|backyard|yard|lawn)\b",
        "parking": r"\b(parking|garage|carport)\b",
        "studio": r"\bstudio\b",
    }
    found = []
    for key, pat in patterns.items():
        if re.search(pat, text_l):
            found.append(key)
    return found

def parse_query_filters(query: str) -> Tuple[int, int, List[str]]:
    """
    Extract bedroom and bathroom counts (support 'studio' -> 0 beds)
    and normalized amenity keywords.
    Returns: (beds or None, baths or None, [keywords])
    """
    q = query.lower()
    beds = None
    baths = None

    # detect explicit "studio" -> bedrooms = 0
    if re.search(r"\bstudio\b", q):
        beds = 0

    mb = re.search(r"(\d+)[ -]?(?:bed|bedroom|bedrooms)\b", q)
    if mb:
        beds = int(mb.group(1))

    mba = re.search(r"(\d+)[ -]?(?:bath|bathroom|bathrooms)\b", q)
    if mba:
        baths = int(mba.group(1))

    keywords = normalize_query_keywords(q)
    return beds, baths, keywords

# AI / Summary
def synthesize_answer_with_context(query: str,
                                   retrieved_records: pd.DataFrame,
                                   use_openai: bool = False,
                                   top_n: int = 5) -> str:
    if retrieved_records.empty:
        return "I couldn't find any properties matching your criteria."

    intro_text = "Based on your query, here are some properties that match your criteria:"
    top_properties = retrieved_records.head(top_n)
    summary_lines = []
    for _, r in top_properties.iterrows():
        price_val = r.get("price")
        price_str = f"${int(price_val):,}" if pd.notna(price_val) else "N/A"
        beds = r.get("bedrooms", "N/A")
        baths = r.get("bathrooms", "N/A")
        address = r.get("address", "N/A")
        summary_lines.append(f"- {address} — {beds} bd / {baths} ba — {price_str}")

    basic_summary = intro_text + "\n" + "\n".join(summary_lines)

    if not use_openai:
        return basic_summary

    # OpenAI path with strict context use
    context_rows = []
    for _, r in retrieved_records.head(max(10, top_n)).iterrows():
        price_val = r.get("price")
        price_str = f"${int(price_val):,}" if pd.notna(price_val) else "N/A"
        context_rows.append(
            f"- Address: {r.get('address','N/A')}, Price: {price_str}, Beds: {r.get('bedrooms','N/A')}, Baths: {r.get('bathrooms','N/A')}"
        )
    prompt = (
        "You are Estate Genie, a real estate assistant. "
        "Answer the user's question based only on the given context. "
        "Be concise and list 3–5 best matches with addresses and key stats.\n\n"
        f"Context:\n{chr(10).join(context_rows)}\n\n"
        f"User Question: {query}\n\nAnswer:"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        return text or basic_summary
    except Exception as e:
        st.warning(f"OpenAI call failed: {e}. Showing a basic summary instead.")
        return basic_summary

# -------------------------
# Streamlit UI
# -------------------------
st.title("🧞‍♂️ Estate Genie")
st.markdown("Your personal real estate assistant. Ask about the property listings.")

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload Property CSV", type=["csv"])

    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    else:
        default_path = "properties_cleaned.csv"
        if os.path.exists(default_path):
            try:
                df = pd.read_csv(default_path)
                st.info(f"Loaded local file: `{default_path}`")
            except Exception as e:
                st.error(f"Failed to read local file `{default_path}`: {e}")

    if df is not None:
        if st.button("Build Property Index"):
            with st.spinner("Processing data and building vector index..."):
                df_proc = preprocess(df)
                model = get_embed_model()
                texts = df_proc["description"].fillna("").tolist()
                embeddings = embed_texts(model, texts)
                index = build_faiss_index(embeddings)
                meta_list = df_proc.to_dict(orient="records")
                save_index_and_meta(index, meta_list)
                st.success(f"Index built and saved with {len(meta_list)} properties.")

    st.markdown("---")
    st.header("🔍 Filters")
    _, meta = load_index_and_meta()

    min_p, max_p = 0, 5_000_000
    if meta:
        df_meta_filter = pd.DataFrame(meta)
        if "price" in df_meta_filter.columns and df_meta_filter["price"].notna().any():
            # safe min/max extraction
            try:
                min_p = int(np.nanmin(df_meta_filter["price"].to_numpy()))
                max_p = int(np.nanmax(df_meta_filter["price"].to_numpy()))
            except Exception:
                min_p, max_p = 0, 5_000_000

    price_range = st.slider("Price Range", min_p, max_p, (min_p, max_p))
    bedrooms_filter = st.selectbox("Bedrooms (Sidebar filter)", ["Any"] + list(range(0, 9)))
    bathrooms_filter = st.selectbox("Minimum Bathrooms (Sidebar filter)", ["Any"] + list(range(0, 7)))
    terrace_filter = st.selectbox("Terrace preference", ["Any", "With terrace", "Without terrace"])
    num_results = st.number_input("Number of results to display", min_value=5, max_value=50, value=10, step=5)

# --- Main Query ---
query = st.text_input("Ask a question...", placeholder="e.g., 2 baths 3 bedroom with terrace")
use_openai = st.checkbox("Use AI-powered answers", value=True)

if st.button("Ask Genie", type="primary"):
    index, meta = load_index_and_meta()
    if index is None or meta is None:
        st.error("The property index has not been built. Click 'Build Property Index' in the sidebar.")
    elif not query:
        st.warning("Please ask a question.")
    else:
        with st.spinner("🧞‍♂️ The Genie is processing your request..."):
            df_meta = pd.DataFrame(meta)

            # Parse query
            q_beds, q_baths, q_keywords = parse_query_filters(query)

            # Quick analytic answers
            analytic_ans, analytic_df = handle_analytic_query(df_meta, query)
            if analytic_ans:
                st.subheader("💡 Quick Answer")
                st.markdown(analytic_ans)
                if analytic_df is not None and not analytic_df.empty:
                    cols = ["address", "price", "bedrooms", "bathrooms", "description"]
                    st.dataframe(analytic_df.head(num_results)[[c for c in cols if c in analytic_df.columns]].fillna("N/A"))
            else:
                # Vector search
                model = get_embed_model()
                q_emb = embed_texts(model, [query])  # shape (1, dim)
                # Search top-k
                try:
                    distances, indices = index.search(q_emb, 100)  # inner product on normalized vectors ~ cosine sim
                except Exception:
                    # If index wasn't trained with normalized vectors, normalize here for similarity
                    distances, indices = index.search(q_emb, 100)

                retrieved_items = [meta[i] for i in indices[0] if (isinstance(i, (int, np.integer)) and i < len(meta))]
                retrieved_df = pd.DataFrame(retrieved_items)

                # Inject similarity safely
                sims = distances[0][:len(retrieved_df)] if len(distances) and len(distances[0]) else []
                retrieved_df["similarity"] = np.array(sims, dtype="float32") if len(sims) == len(retrieved_df) else np.nan

                # ----------------------------
                # APPLY FILTERS: query priority first
                # ----------------------------
                filtered_df = retrieved_df.copy()

                # Query filters with safe numeric handling
                if q_beds is not None and "bedrooms" in filtered_df.columns:
                    # handle float/NaN by filling then casting
                    try:
                        filtered_df = filtered_df[filtered_df["bedrooms"].fillna(-1).astype(int) == int(q_beds)]
                    except Exception:
                        # fallback strict comparison for string-like data
                        filtered_df = filtered_df[filtered_df["bedrooms"] == q_beds]

                if q_baths is not None and "bathrooms" in filtered_df.columns:
                    try:
                        filtered_df = filtered_df[filtered_df["bathrooms"].fillna(0).astype(float) >= float(q_baths)]
                    except Exception:
                        filtered_df = filtered_df[filtered_df["bathrooms"].notna() & (filtered_df["bathrooms"].astype(float) >= float(q_baths))]

                # Keyword (amenity) filters using normalized, lowercase description
                if q_keywords and "description" in filtered_df.columns:
                    def has_all(desc: str) -> bool:
                        text = str(desc).lower()
                        intent_map = {
                            "terrace": r"\b(terrace|terraced|roof terrace|private terrace)\b",
                            "balcony": r"\b(balcony|balconies)\b",
                            "garden": r"\b(garden|backyard|yard|lawn)\b",
                            "parking": r"\b(parking|garage|carport)\b",
                            "studio": r"\bstudio\b",
                        }
                        return all(re.search(intent_map[k], text) for k in q_keywords if k in intent_map)
                    filtered_df = filtered_df[filtered_df["description"].apply(has_all)]

                # Sidebar filters
                if "price" in filtered_df.columns:
                    try:
                        # pandas >= 1.3 uses inclusive arg; safe default provided
                        filtered_df = filtered_df[filtered_df["price"].between(price_range[0], price_range[1], inclusive="both")]
                    except TypeError:
                        # older pandas versions
                        filtered_df = filtered_df[(filtered_df["price"] >= price_range[0]) & (filtered_df["price"] <= price_range[1])]

                if bedrooms_filter != "Any" and "bedrooms" in filtered_df.columns:
                    try:
                        filtered_df = filtered_df[filtered_df["bedrooms"].fillna(-1).astype(int) == int(bedrooms_filter)]
                    except Exception:
                        filtered_df = filtered_df[filtered_df["bedrooms"] == bedrooms_filter]

                if bathrooms_filter != "Any" and "bathrooms" in filtered_df.columns:
                    try:
                        filtered_df = filtered_df[filtered_df["bathrooms"].fillna(0).astype(float) >= int(bathrooms_filter)]
                    except Exception:
                        filtered_df = filtered_df[filtered_df["bathrooms"].notna() & (filtered_df["bathrooms"].astype(float) >= int(bathrooms_filter))]

                if terrace_filter != "Any" and "description" in filtered_df.columns:
                    terr_pat = r"\b(terrace|terraced|roof terrace|private terrace|balcony)\b"
                    if terrace_filter == "With terrace":
                        filtered_df = filtered_df[filtered_df["description"].str.contains(terr_pat, case=False, na=False, regex=True)]
                    else:
                        filtered_df = filtered_df[~filtered_df["description"].str.contains(terr_pat, case=False, na=False, regex=True)]

                # Sorting
                price_keywords = ['cheap', 'cheapest', 'under', 'less than', 'lowest price', 'by price']
                sort_by_price = any(keyword in query.lower() for keyword in price_keywords)

                if filtered_df.empty:
                    st.warning("No properties found that match your search and filter criteria.")
                else:
                    if sort_by_price and "price" in filtered_df.columns:
                        filtered_df = filtered_df.sort_values(by="price", ascending=True, na_position="last")
                    elif "similarity" in filtered_df.columns:
                        filtered_df = filtered_df.sort_values(by="similarity", ascending=False, na_position="last")

                    # Output
                    st.subheader("💬 Genie's Summary")
                    answer = synthesize_answer_with_context(query, filtered_df, use_openai, top_n=num_results)
                    st.markdown(answer)

                    st.subheader("🏡 Relevant Properties Found")
                    if sort_by_price:
                        st.info("ℹ️ Results sorted by price (lowest to highest).")
                    else:
                        st.info("ℹ️ Results sorted by relevance to your query.")

                    base_cols = ["address", "price", "bedrooms", "bathrooms", "description"]
                    if "similarity" in filtered_df.columns:
                        # put similarity before description for quick inspection
                        base_cols.insert(4, "similarity")
                    show_cols = [c for c in base_cols if c in filtered_df.columns]

                    st.dataframe(filtered_df.head(num_results)[show_cols].fillna("N/A"))
