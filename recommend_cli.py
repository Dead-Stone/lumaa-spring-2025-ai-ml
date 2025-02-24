# Add to recommend.py
from recommend import load_data, preprocess_text, initialize_model, get_recommendations


def cli_recommend(query, top_n=5):
    df = load_data()
    df['clean_text'] = preprocess_text(df['overview'])
    tfidf_matrix, vectorizer = initialize_model(df['clean_text'])
    recs = get_recommendations(query, df, tfidf_matrix, vectorizer, top_n)
    return recs[['title', 'similarity']]

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:])
    print(cli_recommend(query))
