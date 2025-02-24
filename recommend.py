# recommend.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import chardet

# @st.cache_data
# def load_data(filepath='data/movies.csv'):
#     try:
#         with open(filepath, 'rb') as f:
#             result = chardet.detect(f.read())
#             encoding = result['encoding']
            
#         df = pd.read_csv(filepath, encoding=encoding)
#         df.columns = df.columns.str.strip().str.lower()
        
#         # Column name mapping with fallbacks
#         column_aliases = {
#             'title': ['title', 'movie_title', 'name', 'movie_name'],
#             'overview': ['overview', 'description', 'plot', 'summary', 'synopsis'],
#             'language': ['language', 'original_language', 'lang'],
#             'release_date': ['release_date', 'date', 'year', 'release_year'],
#             'vote_average': ['vote_average', 'rating', 'score', 'imdb_rating']
#         }
        
#         # Rename columns based on aliases
#         for standard_name, aliases in column_aliases.items():
#             for alias in aliases:
#                 if alias in df.columns:
#                     if standard_name != alias:
#                         df.rename(columns={alias: standard_name}, inplace=True)
#                     break
        
#         # Validate required columns
#         required_cols = ['title', 'overview']
#         missing = [col for col in required_cols if col not in df.columns]
#         if missing:
#             raise ValueError(f"Missing required columns: {', '.join(missing)}")
        
#         # Handle optional columns
#         optional_cols = ['language', 'release_date', 'vote_average']
#         for col in optional_cols:
#             if col not in df.columns:
#                 df[col] = np.nan
                
#         return df[required_cols + optional_cols].dropna()
        
#     except Exception as e:
#         st.error(f"🚨 Data Loading Error: {str(e)}")
#         return pd.DataFrame()
@st.cache_data
def load_data(filepath='data/movies.csv'):
    try:
        with open(filepath, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
            
        df = pd.read_csv(filepath, encoding=encoding)
        df.columns = df.columns.str.strip().map(lambda x: x.lower())
        
        # Validate required columns
        required_cols = ['title', 'overview', 'genres']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        
        # Handle optional columns
        optional_cols = ['homepage', 'original_language', 'release_date', 'vote_average']
        for col in optional_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Extract genres into a list
        def parse_genres(genres_str):
            try:

                genres_list = eval(genres_str)  # Convert string to list of dictionaries
                # print("genres_list",genres_list)
                return [genre['name'] for genre in genres_list if 'name' in genre]
            except (SyntaxError, TypeError):
                return ['Unknown']
        # print(df['genres'])
        df['genres'] = df['genres'].apply(parse_genres)     
        # print(df['genres'])   
        # Set default homepage if missing
        df['homepage'] = df['homepage'].fillna('#')
        
        # Convert release_date to datetime
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        
        return df[required_cols + optional_cols].dropna(subset=required_cols)
        
    except Exception as e:
        st.error(f"🚨 Data Loading Error: {str(e)}")
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
