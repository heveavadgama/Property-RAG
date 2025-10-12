# Property-RAG
Perfect! Here’s a **more polished, visually appealing GitHub README** for **Estate Genie**, complete with badges, sections, and a modern structure:

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


