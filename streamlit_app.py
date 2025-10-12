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

# --- Page Config and Secrets ---
st.set_page_config(page_title="Estate Genie", layout="wide", page_icon="🧞‍♂️")

# Load from Streamlit secrets if available, otherwise from .env
load_dotenv()

if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("❌ OpenAI API key not found. Please add it to your Streamlit Secrets or a .env file.")
    st.stop()

# Create the OpenAI client instance
client = OpenAI(api_key=openai.api_key)


# -------------------------
# Caching and Helper Functions
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

def synthesize_answer_with_context(query: str, retrieved_records: pd.DataFrame, use_openai: bool = False):
    if len(retrieved_records) == 0:
        return "I couldn't find any properties matching your criteria."

    context_rows = []
    for _, r in retrieved_records.head(10).iterrows():
        context_rows.append(f"- Address: {r.get('address', 'N/A')}, Price: ${r.get('price', 0):,}, Beds: {r.get('bedrooms', 'N/A')}, Baths: {r.get('bathrooms', 'N/A')}")
    context_text = "\n".join(context_rows)

    if use_openai:
        prompt = (
            f"You are Estate Genie, a helpful real estate assistant. Answer the user's question based *only* on the context provided below. "
            f"If multiple properties in the context match the user's request, list all of them. Be concise and friendly.\n\n"
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
            answer = resp.choices[0].message.content.strip()
            return answer
        except Exception as e:
            st.warning(f"OpenAI call failed: {e}. Showing a basic summary instead.")

    top_record = retrieved_records.iloc[0]
    price_str = f"${top_record.get('price', 0):,}" if pd.notna(top_record.get('price')) else "N/A"
    return f"Based on your query, the top match is at **{top_record.get('address', 'N/A')}** with a price of **{price_str}**. I found a total of {len(retrieved_records)} relevant properties."


# -------------------------
# Streamlit UI
# -------------------------
st.title("🧞‍♂️ Estate Genie")
st.markdown("Your personal real estate assistant. Ask me anything about the property listings!")

# --- Sidebar for Data Loading and Filtering ---
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
    bedrooms_filter = st.selectbox("Bedrooms", ["Any"] + sorted(list(range(1, 9))))
    bathrooms_filter = st.selectbox("Minimum Bathrooms", ["Any"] + sorted(list(range(1, 7))))


# --- Main Page for Query and Results ---
query = st.text_input("Ask a question...", placeholder="e.g., Show me modern 2-bedroom houses under $1000")
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
            
            analytic_ans, analytic_df = handle_analytic_query(df_meta, query)
            if analytic_ans:
                st.subheader("💡 Quick Answer")
                st.markdown(analytic_ans)
                if not analytic_df.empty:
                    st.dataframe(analytic_df.head(10)[["address", "price", "bedrooms", "bathrooms", "description"]])
            else:
                model = get_embed_model()
                q_emb = embed_texts(model, [query])
                distances, indices = index.search(q_emb, 100) # Retrieve more candidates for filtering
                
                retrieved_items = [meta[i] for i in indices[0] if i < len(meta)]
                retrieved_df = pd.DataFrame(retrieved_items)
                retrieved_df["similarity"] = distances[0][:len(retrieved_df)]

                # Apply sidebar filters
                final_df = retrieved_df[
                    (retrieved_df["price"] >= price_range[0]) & 
                    (retrieved_df["price"] <= price_range[1])
                ].copy()
                if bedrooms_filter != "Any":
                    final_df = final_df[final_df["bedrooms"] == int(bedrooms_filter)]
                if bathrooms_filter != "Any":
                    final_df = final_df[final_df["bathrooms"].fillna(0) >= int(bathrooms_filter)]
                
                # *** NEW: CONDITIONAL SORTING LOGIC ***
                price_keywords = ['cheap', 'cheapest', 'under', 'less than', 'lowest price', 'by price']
                sort_by_price = any(keyword in query.lower() for keyword in price_keywords)

                if sort_by_price:
                    # If the query is about price, sort by price (ascending)
                    final_df = final_df.sort_values(by="price", ascending=True)
                else:
                    # Otherwise, sort by relevance (similarity descending)
                    final_df = final_df.sort_values(by="similarity", ascending=False)


                if final_df.empty:
                    st.warning("No properties found that match your search and filter criteria. Try broadening your search!")
                else:
                    st.subheader("💬 Genie's Answer")
                    answer = synthesize_answer_with_context(query, final_df, use_openai)
                    st.markdown(answer)
                    
                    st.subheader("🏡 Relevant Properties Found")
                    if sort_by_price:
                        st.info("ℹ️ Results sorted by price (lowest to highest).")
                    else:
                        st.info("ℹ️ Results sorted by relevance to your query.")
                        
                    display_cols = ["address", "price", "bedrooms", "bathrooms", "similarity", "description"]
                    st.dataframe(final_df.head(10)[display_cols])
