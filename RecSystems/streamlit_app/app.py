import streamlit as st
import psycopg2
import pandas as pd
import random

# Environment variables (configured in docker-compose.yml)
POSTGRES_HOST = "postgres"
POSTGRES_PORT = "5432"
POSTGRES_DB = "movies"
POSTGRES_USER = "your_user"
POSTGRES_PASSWORD = "your_password"

# Connect to PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    return conn

# Function to fetch all movies from the database
def get_all_movies():
    conn = get_db_connection()
    query = "SELECT * FROM movies;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Function to fetch movie by name
def get_movie_by_name(movie_name):
    conn = get_db_connection()
    query = f"SELECT * FROM movies WHERE title ILIKE '%{movie_name}%';"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Placeholder function for collaborative filtering
def collaborative_filtering(movie_id):
    # This would be replaced with the actual collaborative filtering logic
    return f"Recommended movies based on collaborative filtering for movie ID {movie_id}"

# Placeholder function for content-based filtering
def content_based_filtering(movie_id):
    # This would be replaced with the actual content-based filtering logic
    return f"Recommended movies based on content for movie ID {movie_id}"

# Placeholder function for matrix factorization-based recommendations
def matrix_factorization(movie_id):
    # This would be replaced with actual matrix factorization logic
    return f"Recommended movies based on matrix factorization for movie ID {movie_id}"

# Function to randomly recommend a movie
def random_movie_recommendation():
    movies = get_all_movies()
    random_movie = movies.sample(n=1)
    return random_movie

# Function to display recommendations
def display_recommendations(movie_id):
    st.subheader("Recommendations Based on Selected Movie")

    # Collaborative filtering recommendation
    st.write(collaborative_filtering(movie_id))

    # Content-based filtering recommendation
    st.write(content_based_filtering(movie_id))

    # Matrix factorization recommendation
    st.write(matrix_factorization(movie_id))

# Streamlit layout
def main():
    st.title("Movie Recommendation App")

    # Step 1: Ask the user to choose between random movie or search for a movie
    choice = st.radio("Choose how you want to get movie recommendations", ("Random Movie", "Search for a Movie"))

    if choice == "Random Movie":
        # Step 2: Fetch a random movie from the database
        random_movie = random_movie_recommendation()
        st.write(f"Random Movie Selected: {random_movie['title'].values[0]}")

        # Display recommendations based on the random movie
        display_recommendations(random_movie['id'].values[0])

    elif choice == "Search for a Movie":
        # Step 3: Search for a movie by title
        movie_name = st.text_input("Enter the movie title")

        if movie_name:
            # Fetch the movie from the database
            movie = get_movie_by_name(movie_name)

            if not movie.empty:
                st.write(f"Movie Found: {movie['title'].values[0]}")

                # Display recommendations based on the movie
                display_recommendations(movie['id'].values[0])

            else:
                st.error(f"Movie '{movie_name}' not found in the database.")

if __name__ == "__main__":
    main()
