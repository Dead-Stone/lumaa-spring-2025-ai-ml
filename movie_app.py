# app.py
import pandas as pd
import streamlit as st
from recommend import load_data, preprocess_text, initialize_model, get_recommendations

def show_sidebar():
    with st.sidebar:
        st.header("🎬 Movie Recommender Guide")
        with st.expander("🔍 How It Works", expanded=True):
            st.markdown("""
            1. **Describe** your movie preferences
            2. **Analyze** plot summaries using AI
            3. **Match** with similar movies
            4. **Explore** recommendations with details
            """)
        st.markdown("---")
        st.markdown("Built with ❤️ by Mohana")

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Tell me about your ideal movie experience! 🍿"
        }]

def display_chat():
    for msg in st.session_state.messages:
        avatar = "🎥" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

def generate_recommendation_table(recommendations):
    
    LANGUAGE_MAP = {
        'en': 'English',
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'ja': 'Japanese',
        'cn': 'Chinese',
        'hi': 'Hindi',
        'ko': 'Korean'
    }
    css = """
    <style>
        .movie-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-family: Arial, sans-serif;
        }
        
        .movie-table th {
            background-color: #f4f4f4;
            padding: 10px;
            text-align: left;
            border-bottom: 2px solid #ddd;
        }
        
        .movie-table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        
        .movie-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .movie-description {
            font-size: 12px;
            color: #555;
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 250px;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 5px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s ease-in-out;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """
    
    html = f"""
    {css}
    <table class="movie-table">
        <tr>
            <th>🎬 Title</th>
            <th>Match Score</th>
        </tr>
    """
    print(recommendations)
    for _, row in recommendations.iterrows():
        # genres = row.get('genres', 'N/A')
        # print(genres)
        # if isinstance(genres, str):
        #     genres = genres.split('|')  # Assuming genres are pipe-separated
        # elif isinstance(genres, list):
        #     genres = [g['name'] for g in genres if isinstance(g, dict) and 'name' in g]
        # else:
        #     genres = ['N/A']
        
        # homepage = row.get('homepage', '#')
        # # Convert language code
        # lang_code = str(row.get('language', 'en')).lower()
        # lang_name = LANGUAGE_MAP.get(lang_code, 'Unknown')
        # year = 'N/A'
        print(row['genres'])
        genres = ', '.join(row['genres'])
        lang_code = row['original_language']
        lang_name = LANGUAGE_MAP.get(lang_code, 'Unknown')
        year = row['release_date'].year if pd.notnull(row['release_date']) else 'N/A'
        if pd.notnull(row.get('release_date')):
            try:
                year = pd.to_datetime(row['release_date']).year
            except:
                pass
        html += f"""
        <tr class="movie-row" onclick="this.nextElementSibling.classList.toggle('show')">
            <td>
                <div class="movie-title">{row['title']}<div class="tooltip" style="
            font-size: 9px;">
                     ℹ️
                    <span class="tooltiptext" style="font-size: 12px;">
                        <b>🗣 Language:</b> {lang_name}<br>
                        <b>⭐ Rating:</b> {row.get('vote_average', 'N/A')}/10<br>
                        <b>📅 Released:</b> {year}<br>
                        <b>🎭 Genres:</b> {genres}<br>
                    </span>
                </div> </div>
                <div class="movie-description">{row['overview']}</div>
                
            </td>
            <td style="text-align:center">{int(row['similarity'] * 100)}%</td>
        </tr>
        """
    
    html += "</table>"
    return html



def main():
    st.set_page_config(
        page_title="CineMatch AI",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_chat()
    show_sidebar()
    
    st.title("CineMatch AI: Smart Movie Recommendations")
    st.caption("Discover films tailored to your tastes using NLP technology")
    
    try:
        movies_df = load_data()
        if movies_df.empty:
            st.error("No data loaded. Please check your CSV file.")
            return
            
        movies_df['clean_text'] = preprocess_text(movies_df['overview'])
        tfidf_matrix, vectorizer = initialize_model(movies_df['clean_text'])
        
    except Exception as e:
        st.error(f"🔧 Initialization Error: {str(e)}")
        return

    display_chat()

    if prompt := st.chat_input("Describe your ideal movie..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.spinner("🔍 Analyzing preferences..."):
            try:
                recommendations = get_recommendations(
                    prompt,
                    movies_df,
                    tfidf_matrix,
                    vectorizer
                )
                
                if not recommendations.empty:
                    html_table = generate_recommendation_table(recommendations)
                    with st.chat_message("assistant", avatar="🎥"):
                        st.subheader(f"🎯 Recommendations for: _{prompt}_")
                        st.html(html_table)
                        st.caption("ℹ️ Hover over titles for detailed information")
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Found {len(recommendations)} matching movies!"
                    })
                else:
                    st.error("No recommendations found. Try a different description.")
                    
            except Exception as e:
                st.error(f"🚨 Recommendation Error: {str(e)}")

if __name__ == "__main__":
    main()
