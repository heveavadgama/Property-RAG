# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
from typing import List
import re
import openai
from dotenv import load_dotenv
from openai import OpenAI
import nltk

# --- Page Config ---
st.set_page_config(page_title="Estate Genie", layout="wide", page_icon="🧞‍♂️")

# Load OpenAI key
load_dotenv()
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("❌ OpenAI API key not found. Add it to Streamlit secrets or .env.")
    st.stop()

client = OpenAI(api_key=openai.api_key)

# -------------------------
# Helper Functions
# -------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "property_type_full_description" in df.columns:
        df["description"] = df["property_type_full_description"].fillna("") + " | " + df.get("type","").fillna("").astype(str)
    else:
        df["description"] = df.get("description", "").fillna("") + " | " + df.get("type","").fillna("").astype(str)
    for c in ["price","bedrooms","bathrooms","crime_score_weight"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.reset_index(drop=True, inplace=True)
    return df

def build_faiss_index(embeddings: np.ndarray, dim: int):
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def save_index_and_meta(index, meta_list, filepath_index="faiss.index", filepath_meta="meta.pkl"):
    faiss.write_index(index, filepath_index)
    with open(filepath_meta, "wb") as f:
        pickle.dump(meta_list, f)

def load_index_and_meta(filepath_index="faiss.index", filepath_meta="meta.pkl"):
    if not (os.path.exists(filepath_index) and os.path.exists(filepath_meta)):
        return None, None
    index = faiss.read_index(filepath_index)
    with open(filepath_meta, "rb") as f:
        meta_list = pickle.load(f)
    return index, meta_list

def embed_texts(model, texts: List[str]) -> np.ndarray:
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    emb = normalize(emb, norm='l2')
    return emb.astype("float32")

# Analytic query handling
def handle_analytic_query(df: pd.DataFrame, q: str):
    q_l = q.lower()
    m = re.search(r"average price of (\d+)[ -]?bed", q_l)
    if m:
        beds = int(m.group(1))
        sub = df[df["bedrooms"] == beds]
        if len(sub):
            return f"The average price of {beds}-bedroom homes, based on {len(sub)} listings, is **${sub['price'].mean():,.2f}**.", sub
        else:
            return f"I couldn't find any listings for {beds}-bedroom homes to calculate the average price.", pd.DataFrame()

    if "most crime" in q_l:
        if "crime_score_weight" in df.columns:
            tmp = df.groupby("address", dropna=True)["crime_score_weight"].mean().sort_values(ascending=False)
            if len(tmp):
                top_address = tmp.index[0]
                top_score = tmp.iloc[0]
                return f"The area with the highest average crime score is **{top_address}** with a score of {top_score:.2f}.", df[df["address"] == top_address]
        return "The dataset does not contain a 'crime_score_weight' column to answer this question.", pd.DataFrame()
    return None, None

# NLP parsing for query
def normalize_query_keywords(text):
    synonym_map = {
        "terrace": "terraced",
        "terraced": "terraced",
        "balcony": "balcony",
        "garden": "garden",
    }
    normalized = []
    for word in text.lower().split():
        if word in synonym_map:
            normalized.append(synonym_map[word])
    return normalized

def parse_query_filters(query: str):
    q = query.lower()
    beds = None
    baths = None
    match_beds = re.search(r"(\d+)[ -]?bed", q)
    if match_beds:
        beds = int(match_beds.group(1))
    match_baths = re.search(r"(\d+)[ -]?bath", q)
    if match_baths:
        baths = int(match_baths.group(1))
    keywords = normalize_query_keywords(q)
    return beds, baths, keywords

# AI / Summary
def synthesize_answer_with_context(query: str, retrieved_records: pd.DataFrame, use_openai: bool = False, top_n: int = 3):
    if len(retrieved_records) == 0:
        return "I couldn't find any properties matching your criteria."
    top_properties = retrieved_records.head(top_n)
    summary_lines = []
    for _, r in top_properties.iterrows():
        price_str = f"${r.get('price', 0):,}" if pd.notna(r.get('price')) else "N/A"
        beds = r.get('bedrooms', 'N/A')
        baths = r.get('bathrooms', 'N/A')
        summary_lines.append(f"- {r.get('address','N/A')} — {beds} bd / {baths} ba — {price_str}")
    text_summary = f"I found {len(retrieved_records)} properties matching your query. Top {top_n} results:\n" + "\n".join(summary_lines)

    if use_openai and openai.api_key:
        context_rows = [
            f"- Address: {r.get('address', 'N/A')}, Price: ${r.get('price', 0):,}, Beds: {r.get('bedrooms', 'N/A')}, Baths: {r.get('bathrooms', 'N/A')}"
            for _, r in retrieved_records.head(max(10, top_n)).iterrows()
        ]
        context_text = "\n".join(context_rows)
        prompt = (
            f"You are Estate Genie, a helpful real estate assistant. "
            f"Answer the user's question based *only* on the context below.\n\n"
            f"**Context:**\n{context_text}\n\n"
            f"**User Question:** {query}\n\n"
            f"**Answer:**"
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            ai_answer = resp.choices[0].message.content.strip()
            return ai_answer
        except Exception as e:
            st.warning(f"OpenAI call failed: {e}. Showing a basic summary instead.")
    return text_summary

# -------------------------
# Streamlit UI
# -------------------------
st.title("🧞‍♂️ Estate Genie")
st.markdown("Your personal real estate assistant. Ask me anything about the property listings!")

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload Property CSV", type=["csv"])

    df = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        default_path = "properties_cleaned.csv"
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
            st.info(f"Loaded local file: `{default_path}`")

    if df is not None:
        if st.button("Build Property Index"):
            with st.spinner("Processing data and building vector index... Please wait."):
                df_proc = preprocess(df)
                model = get_embed_model()
                embeddings = embed_texts(model, df_proc["description"].fillna("").tolist())
                index = build_faiss_index(embeddings, embeddings.shape[1])
                meta_list = df_proc.to_dict(orient="records")
                save_index_and_meta(index, meta_list)
                st.success(f"Index built and saved with {len(meta_list)} properties.")

    st.markdown("---")
    st.header("🔍 Filters")
    _, meta = load_index_and_meta()
    min_p, max_p = 0, 5000000
    if meta:
        df_meta_filter = pd.DataFrame(meta)
        if 'price' in df_meta_filter.columns and df_meta_filter['price'].notna().any():
            min_p = int(df_meta_filter['price'].min())
            max_p = int(df_meta_filter['price'].max())  

    price_range = st.slider("Price Range", min_p, max_p, (min_p, max_p))
    bedrooms_filter = st.selectbox("Bedrooms (Sidebar filter)", ["Any"] + sorted(list(range(1, 9))))
    bathrooms_filter = st.selectbox("Minimum Bathrooms (Sidebar filter)", ["Any"] + sorted(list(range(1, 7))))
    terrace_filter = st.selectbox("Terrace preference", ["Any", "With terrace", "Without terrace"])
    num_results = st.number_input("Number of results to display", min_value=5, max_value=50, value=10, step=5)

# --- Main Query ---
query = st.text_input("Ask a question...", placeholder="e.g., 2 baths 3 bedroom with terrace")
use_openai = st.checkbox("Use AI-powered answers", value=True)

if st.button("Ask Genie", type="primary"):
    index, meta = load_index_and_meta()
    if index is None or meta is None:
        st.error("The property index has not been built. Please click 'Build Property Index' in the sidebar.")
    elif not query:
        st.warning("Please ask a question.")
    else:
        with st.spinner("🧞‍♂️ The Genie is thinking..."):
            df_meta = pd.DataFrame(meta)

            # Parse query
            q_beds, q_baths, q_keywords = parse_query_filters(query)

            # Quick analytic answers
            analytic_ans, analytic_df = handle_analytic_query(df_meta, query)
            if analytic_ans:
                st.subheader("💡 Quick Answer")
                st.markdown(analytic_ans)
                if not analytic_df.empty:
                    df_display = analytic_df.head(num_results)[["address", "price", "bedrooms", "bathrooms", "description"]].fillna("N/A")
                    st.dataframe(df_display)
            else:
                model = get_embed_model()
                q_emb = embed_texts(model, [query])
                distances, indices = index.search(q_emb, 100)

                retrieved_items = [meta[i] for i in indices[0] if i < len(meta)]
                retrieved_df = pd.DataFrame(retrieved_items)
                retrieved_df["similarity"] = distances[0][:len(retrieved_df)]

                # ----------------------------
                # APPLY FILTERS: query priority first
                # ----------------------------
                filtered_df = retrieved_df.copy()
                if q_beds:
                    filtered_df = filtered_df[filtered_df["bedrooms"] == q_beds]
                if q_baths:
                    filtered_df = filtered_df[filtered_df["bathrooms"].fillna(0) >= q_baths]
                if q_keywords:
                    filtered_df = filtered_df[filtered_df["description"].str.lower().apply(
                        lambda desc: all(k in desc for k in q_keywords)
                    )]

                # Sidebar filters applied additionally if user selected
                if bedrooms_filter != "Any":
                    filtered_df = filtered_df[filtered_df["bedrooms"] == int(bedrooms_filter)]
                if bathrooms_filter != "Any":
                    filtered_df = filtered_df[filtered_df["bathrooms"].fillna(0) >= int(bathrooms_filter)]
                if terrace_filter != "Any":
                    if terrace_filter == "With terrace":
                        filtered_df = filtered_df[filtered_df["description"].str.lower().str.contains("terraced")]
                    else:
                        filtered_df = filtered_df[~filtered_df["description"].str.lower().str.contains("terraced")]

                # Sort by price if query mentions
                price_keywords = ['cheap', 'cheapest', 'under', 'less than', 'lowest price', 'by price']
                sort_by_price = any(keyword in query.lower() for keyword in price_keywords)
                if sort_by_price:
                    filtered_df = filtered_df.sort_values(by="price", ascending=True)
                else:
                    filtered_df = filtered_df.sort_values(by="similarity", ascending=False)

                if filtered_df.empty:
                    st.warning("No properties found that match your search and filter criteria.")
                else:
                    st.subheader("💬 Genie's Summary")
                    answer = synthesize_answer_with_context(query, filtered_df, use_openai, top_n=num_results)
                    st.markdown(answer)

                    st.subheader("🏡 Relevant Properties Found")
                    if sort_by_price:
                        st.info("ℹ️ Results sorted by price (lowest to highest).")
                    else:
                        st.info("ℹ️ Results sorted by relevance to your query.")
                    display_cols = ["address","price","bedrooms","bathrooms","similarity","description"]
                    st.dataframe(filtered_df.head(num_results)[display_cols].fillna("N/A"))
