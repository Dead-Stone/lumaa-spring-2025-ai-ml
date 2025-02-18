import streamlit as st
from recommend import load_data, preprocess_text, initialize_model, get_recommendations

def show_sidebar():
    with st.sidebar:
        st.header("About This App")
        with st.expander("⚙️ System Mechanics", expanded=True):
            st.markdown("""
            **Recommendation Process:**
            1. **Text Processing**: Clean and normalize input text
            2. **TF-IDF Vectorization**: Convert text to numerical features
            3. **Similarity Search**: Find closest matches using cosine similarity
            4. **Ranking**: Select top 5 most relevant movies
            """)
            st.markdown("""
            ```
            graph LR
                A[User Query] --> B(Text Preprocessing)
                B --> C(TF-IDF Vectorization)
                C --> D[Similarity Calculation]
                D --> E[Top 5 Recommendations]
            ```
            """)
        st.markdown("---\nBuilt with ❤️ using Streamlit")

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hi! I'm your movie expert. Describe your perfect movie and I'll recommend something! 🍿"
        }]

def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🎥" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

def show_recommendations(recommendations, prompt):
    with st.chat_message("assistant", avatar="🎥"):
        st.subheader(f"Top Recommendations for: _{prompt}_")
        for idx, row in recommendations.iterrows():
            with st.expander(f"🎞️ {row['title']} (Score: {row['similarity']:.2f})"):
                st.markdown(f"**Plot Summary**:\n{row['overview']}")
                st.progress(row['similarity'])
        st.caption("🔢 Recommendations ranked by cosine similarity score (0-1 scale)")

def main():
    st.set_page_config(
        page_title="Movie Recommender Chat",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize app components
    initialize_chat()
    show_sidebar()
    
    # App header
    st.title("Movie Recommendation ChatBot 🎥")
    st.caption("🍿 Let's find your next favorite movie!")
    # Load data and models
    try:
        movies_df = load_data()
        movies_df['clean_text'] = preprocess_text(movies_df['overview'])
        tfidf_matrix, vectorizer = initialize_model(movies_df['clean_text'])
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # Display chat history
    display_chat()

    # Handle user input
    if prompt := st.chat_input("Describe your movie preferences..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate and display recommendations
        with st.spinner("🔍 Searching our movie database..."):
            recommendations = get_recommendations(
                prompt,
                movies_df,
                tfidf_matrix,
                vectorizer
            )
        
        show_recommendations(recommendations, prompt)

        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Found {len(recommendations)} movies matching your preferences!"
        })

if __name__ == "__main__":
    main()
