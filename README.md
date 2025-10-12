# 🧞‍♂️ Estate Genie

[![Streamlit App](https://img.shields.io/badge/Live-App-blue?style=for-the-badge\&logo=streamlit)](https://estategenie.streamlit.app/#estate-genie)
[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge\&logo=python)](https://www.python.org/)


**Estate Genie** is an intelligent real estate exploration platform that transforms how users discover properties through natural language conversations. Leveraging cutting-edge AI and semantic search technologies, it provides instant, human-like responses to property queries with both concise summaries and detailed listings.

---

## 1. Overview

Estate Genie bridges the gap between complex property data and user-friendly discovery. Instead of traditional filter-based searches, users can ask questions in plain English like:

* *"Show me affordable 3-bedroom houses with a garden near schools"*
* *"Top 5 luxury apartments with ocean views under $1M"*
* *"Most expensive properties in downtown with high crime scores"*

The system understands context, synonyms, and user intent to deliver precisely matched properties with AI-generated insights.

---

## 2. ✨ Key Features

### 🗣️ Natural Language Understanding

* **Advanced Query Parsing:** Extracts bedrooms, bathrooms, price ranges, and special features.
* **Synonym Handling:** Automatically maps `"terrace"` → `"terraced"`, `"apt"` → `"apartment"`.
* **Intent Recognition:** Distinguishes between search queries and analytical questions.

### 🤖 AI-Powered Intelligence

* **GPT-4o-mini Integration:** Generates human-like property summaries.
* **Semantic Search:** Finds properties based on meaning, not just keywords.
* **Analytical Queries:** Answers statistical questions about market trends.

### 🔍 Advanced Search Capabilities

* **FAISS Vector Search:** Lightning-fast similarity matching.
* **Query Priority System:** Natural language queries override sidebar filters.
* **Multi-modal Filtering:** Combine NLP with traditional filters seamlessly.

### 📊 Interactive Results

* **Dual Presentation:** Concise AI summary + detailed tabular data.
* **Smart Sorting:** Relevance-based or price-based sorting based on query context.
* **Real-time Filtering:** Dynamic updates with interactive sidebar controls.

---

## 3. 🛠 Tech Stack

| Component       | Technology           |
| --------------- | -------------------- |
| Frontend        | Streamlit            |
| Backend         | Python 3.13          |
| Language Model  | OpenAI GPT-4o-mini   |
| Embeddings      | SentenceTransformers |
| NLP             | SpaCy + NLTK         |
| Vector Database | FAISS                |
| Data Processing | Pandas + NumPy       |
| Caching         | Streamlit Cache      |
| Environment     | python-dotenv        |
| Secrets         | Streamlit Secrets    |

---

## 4. 🏗 Architecture & Data Flow
<img width="697" height="813" alt="image" src="https://github.com/user-attachments/assets/1b994964-6dfc-410d-86b9-e203938f6a45" />



1. **Ingestion:** CSV data is loaded and preprocessed.
2. **Embedding:** Property descriptions converted to vector representations.
3. **Indexing:** Vectors stored in FAISS for fast retrieval.
4. **Query Processing:** User input parsed and converted to query vector.
5. **Retrieval:** Similarity search returns candidate properties.
6. **Filtering:** Query and sidebar filters applied.
7. **Generation:** AI creates natural language summary.
8. **Presentation:** Dual-format results displayed to user.

---

## 5. 📥 Installation

### Prerequisites

* Python 3.13+
* OpenAI API key
* 2GB+ RAM recommended

### Steps

```bash
# Clone the repo
git clone https://github.com/yourusername/estategenie.git
cd estategenie

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

---

## 6. 🏃 How to Use the App

1. Upload your CSV property dataset or use the default file.
2. Click **Build Property Index** to preprocess data and build vector index.
3. Ask queries in natural language, e.g., `"Top 10 2-bedroom terraced houses under $1500"`.
4. Use sidebar toggles for price range, bedrooms, bathrooms, and terrace options.
5. View:

   * **AI Summary:** Quick, human-like property insights.
   * **Property Table:** Detailed listings with address, price, beds, baths, similarity score, and description.

---

## 7. 📂 Dataset Requirements

Your CSV should include these columns (case-insensitive):

| Column      | Description                               |
| ----------- | ----------------------------------------- |
| address     | Property address                          |
| price       | Numeric property price                    |
| bedrooms    | Number of bedrooms                        |
| bathrooms   | Number of bathrooms                       |
| type        | Property type (e.g., apartment, terraced) |
| description | Optional property description             |

> NLP preprocessing automatically handles variations like "terrace" → "terraced" for better matching.

---

## 8. 📌 Live Demo

[![Streamlit App](https://img.shields.io/badge/Launch-Live%20Demo-orange?style=for-the-badge\&logo=streamlit)](https://estategenie.streamlit.app/#estate-genie)

---

## 9. 👏 Outro

Estate Genie makes property discovery smarter, faster, and more intuitive. Whether you're a casual homebuyer, real estate investor, or analyst, Estate Genie provides relevant insights at your fingertips.

---


# 🧞‍♂️ Estate Genie

[![Streamlit App](https://img.shields.io/badge/Live-App-blue?style=for-the-badge\&logo=streamlit)](https://estategenie.streamlit.app/#estate-genie)
[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge\&logo=python)](https://www.python.org/)

**Estate Genie** is an AI-powered real estate assistant that lets users explore property listings using natural language queries. Get both a concise AI summary and a detailed table of relevant properties instantly!

---

## 🌟 Key Features

* **Natural Language Queries:** Ask questions in plain English like “Top 5 3-bedroom terraced houses under $2000.”
* **AI-Powered Summaries:** GPT-4o-mini generates quick, concise summaries of the top matching properties.
* **Query-Prioritized Results:** User query input is prioritized over toggle filters in the sidebar.
* **Interactive Filters:** Use toggles for price, bedrooms, bathrooms, and terrace availability.
* **Semantic Search:** Advanced sentence embeddings & FAISS vector search for relevant results.
* **CSV Upload Support:** Upload your own property dataset or use the default CSV.
* **Detailed Property Table:** Shows address, price, beds, baths, similarity score, and description.

---

## 🛠 Tech Stack

* **Frontend:** Streamlit
* **AI & NLP:** OpenAI GPT-4o-mini, NLTK, SpaCy
* **Vector Search:** Sentence Transformers, FAISS
* **Data Processing:** Pandas, NumPy
* **Environment:** Python 3.13, dotenv

---

## 🚀 How It Works

1. **Upload Data:** Upload a CSV file containing property listings.
2. **Build Vector Index:** Preprocess data, generate embeddings, and create a FAISS index.
3. **Ask Queries:** Enter a natural language question in the input box.
4. **View Results:**

   * **Summary:** AI-powered concise overview of top results.
   * **Table:** Interactive table with property details, sorted by relevance or price.

---

## 📂 Data Requirements

Your CSV file should include these columns (case-insensitive):

| Column      | Description                               |
| ----------- | ----------------------------------------- |
| address     | Property address                          |
| price       | Property price (numeric)                  |
| bedrooms    | Number of bedrooms                        |
| bathrooms   | Number of bathrooms                       |
| type        | Property type (e.g., apartment, terraced) |
| description | Optional property description             |

> Note: NLP preprocessing handles variations like “terrace” → “terraced” automatically.

---

## 🧠 AI & NLP Features

* **Query Parsing:** Extracts top N results, bedrooms, bathrooms, and special features from user queries.
* **Semantic Matching:** Handles synonyms, plurals, and common variations in property features.
* **Neutral Summaries:** Provides concise property summaries without misleading statements.

---

## 📸 Screenshots

| Home Page                                    | Property Table                               |
| -------------------------------------------- | -------------------------------------------- |
| ![Screenshot 1](screenshots/screenshot1.png) | ![Screenshot 2](screenshots/screenshot2.png) |

*(Replace with actual screenshots from your app folder)*

---

## 💻 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/estategenie.git
cd estategenie

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

---


## 🔗 Live Demo

[![Streamlit App](https://img.shields.io/badge/Launch-Live%20Demo-orange?style=for-the-badge\&logo=streamlit)](https://estategenie.streamlit.app/#estate-genie)


