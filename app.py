import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from difflib import get_close_matches

# Loading datasets and saving them in cache for faster loading, using st.cache
@st.cache_resource
def load_data():
    ratings = pd.read_csv("ml-latest-small/ratings.csv")
    movies = pd.read_csv("ml-latest-small/movies.csv")
    links = pd.read_csv("ml-latest-small/links_with_poster.csv")
    
    # Preprocess movies data
    genres = set("|".join(movies["genres"].unique()).split("|"))
    genres.discard("(no genres listed)")

    for genre in genres:
        movies[genre] = movies["genres"].apply(lambda x: 1 if genre in x else 0)
    
    movies.drop(columns=['genres', 'IMAX'], axis=1, inplace=True)
    
    # Merging data
    merged_data = pd.merge(ratings, movies, on="movieId")
    merged_data = merged_data.drop(["timestamp"], axis=1)
    
    # Preparing content-based filtering
    movies["genres"] = movies.iloc[:, 4:].dot(movies.columns[4:] + " ")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Preparing collaborative filtering
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(merged_data[["userId", "movieId", "rating"]], reader)
    algo = SVD()
    cross_validate(algo, data, cv=5, verbose=False)
    
    # Creating mappings
    movieId_to_index = {movie_id: idx for idx, movie_id in enumerate(movies["movieId"])}
    index_to_movieId = {idx: movie_id for movie_id, idx in movieId_to_index.items()}
    
    # Merge poster URLs
    movies = movies.merge(links[['movieId', 'poster_url']], on='movieId', how='left')
    
    return movies, merged_data, content_similarity, algo, movieId_to_index, index_to_movieId

# Recommendation function
def recommend_movies(movie_name, movies, merged_data, content_similarity, algo, movieId_to_index, index_to_movieId, num_recommendations=10):
    # Validate the movie name or find the closest match
    if movie_name not in movies["title"].values:
        close_match = get_close_matches(movie_name, movies["title"].values, n=1, cutoff=0.5)
        if close_match:
            # st.warning(f"Movie '{movie_name}' not found. Using '{close_match[0]}'.")
            movie_name = close_match[0]
        else:
            st.error(f"Movie '{movie_name}' not found, and no close matches are available.")
            return None

    # Map movieId to matrix index
    movieId = movies[movies["title"] == movie_name]["movieId"].iloc[0]
    current_index = movieId_to_index[movieId]

    # Perform content-based recommendation
    similarity_scores = content_similarity[current_index]
    recommended_movies = [
        (index_to_movieId[idx], score) for idx, score in enumerate(similarity_scores)
    ]

    # Sort and select top recommendations
    recommendations = sorted(recommended_movies, key=lambda x: x[1], reverse=True)[:num_recommendations]

    # Fetch movie details for recommendations
    recommended_movies_df = pd.DataFrame(recommendations, columns=["movieId", "score"])
    recommended_movies_df = recommended_movies_df.merge(movies, on="movieId", how="left")

    return recommended_movies_df[["title", "score", "poster_url"]]

# Streamlit App function, main function
def main():
    st.set_page_config(page_title="Movie Recommender", page_icon=":movie_camera:", layout="wide")
    
    # Load data
    movies, merged_data, content_similarity, algo, movieId_to_index, index_to_movieId = load_data()
    
    # Title of the page
    st.title("🎬 Movie Recommender System")
    
    # Get the movie name from the user
    movie_name = st.text_input("Enter a movie name:")
    
    # Showing a slider to get the number of recommendations from the user
    num_recommendations = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)
    
    # This will create a "Recommendation button"
    if st.button("Get Recommendations"):
        if movie_name:
            # Callong recommend_movies function to get recommendations
            recommendations = recommend_movies(
                movie_name, 
                movies, 
                merged_data, 
                content_similarity, 
                algo, 
                movieId_to_index, 
                index_to_movieId, 
                num_recommendations
            )
            
            # This will Display the recommendations
            if recommendations is not None and not recommendations.empty:
                
                st.subheader(f"Recommendations for '{movie_name}':")
                
                columns = st.columns(5)
                
                for i, (index, row) in enumerate(recommendations.iterrows()):
                    with columns[i % 5]:
                        # Display poster
                        if pd.notna(row['poster_url']):
                            st.image(row['poster_url'], use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/300x450.png?text=No+Poster", use_container_width=True)
                        
                        # Display movie title
                        st.write(row['title'])
                        st.write(f"Similarity Score: {row['score']:.4f}")
            else:
                st.warning("No recommendations found.")
        else:
            st.warning("Please enter a movie name.")

if __name__ == "__main__":
    main()