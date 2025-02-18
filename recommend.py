import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import chardet
@st.cache_data
def load_data(filepath='data/movies.csv'):
    try:
        # Best practice: Detect encoding first
        with open(filepath, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
            
        return pd.read_csv(filepath, encoding=encoding)[['title', 'overview']].dropna()
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def preprocess_text(text):
    return text.str.lower().fillna('')

@st.cache_resource
def initialize_model(texts):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def get_recommendations(query, df, tfidf_matrix, vectorizer, top_n=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)
    similar_indices = np.argsort(similarities[0])[::-1][:top_n]
    return df.iloc[similar_indices].assign(similarity=similarities[0][similar_indices])
