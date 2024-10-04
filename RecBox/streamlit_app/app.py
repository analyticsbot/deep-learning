import streamlit as st
import psycopg2
import requests

# TMDb API Key (replace with your actual API key)
TMDB_API_KEY = '25255ca7828db7ab3a83f28535c38450'
TMDB_API_URL = 'https://api.themoviedb.org/3'

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="postgres",         # e.g., "localhost" or your PostgreSQL container name
        database="airflow",  # Your database name
        user="airflow",     # Your PostgreSQL username
        password="airflow",  # Your PostgreSQL password
        port=5432
    )

# Function to fetch recommendations from the database
def fetch_user_recommendations(user_id=None):
    conn = get_db_connection()
    cur = conn.cursor()

    if user_id:
        # Fetch recommendations for a specific user
        cur.execute("""
            SELECT r."userId", r."movieId", r."title", r."tmdbId", r."rating"
            FROM recommendations r 
            WHERE r."userId" = %s
            """, (user_id,))
    else:
        # Fetch recommendations for 5 random movies
        cur.execute("""
            SELECT r."userId", r."movieId", r."title", r."tmdbId", r."rating"
            FROM recommendations r
            ORDER BY RANDOM()
            LIMIT 5;
        """)

    recommendations = cur.fetchall()  # Fetch recommendations
    cur.close()
    conn.close()
    return recommendations

# Fetch movie poster from TMDb using the movie's TMDb ID
def fetch_movie_poster(tmdb_id):
    try:
        response = requests.get(f'{TMDB_API_URL}/movie/{tmdb_id}/images?api_key={TMDB_API_KEY}', timeout=10, verify=False)

        if response.status_code == 200:
            data = response.json()
            if 'posters' in data and len(data['posters']) > 0:
                return f"https://image.tmdb.org/t/p/w500{data['posters'][0]['file_path']}"
            else:
                return "https://via.placeholder.com/150"
        else:
            print(f"Error: Unable to fetch data. Status Code: {response.status_code}")
            return "https://via.placeholder.com/150"

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "https://via.placeholder.com/150"

# Title of the application
st.header("Personalized Movie Recommendations")

# Input to search for a specific user by ID
searched_user_id = st.text_input("Enter your User ID to see your recommendations:")

if searched_user_id:
    # Check if the user exists
    recommendations = fetch_user_recommendations(user_id=searched_user_id)
    if recommendations:
        st.subheader(f"Recommendations for User {searched_user_id}:")
        cols = st.columns(5)  # Create columns for displaying movies side by side
        for idx, rec in enumerate(recommendations):
            user_id = rec[0]
            movie_id = rec[1]
            title = rec[2]
            tmdb_id = rec[3]
            rating = rec[4]
            movie_poster = fetch_movie_poster(tmdb_id)
            
            with cols[idx % 5]:  # Display in columns, cycling through the columns
                st.markdown(f"""
                <div style='margin-bottom: 20px; margin-right: 20px;'>
                    <img src="{movie_poster}" width="150"/>
                    <p>{title} (Rating: {round(rating, 2)})</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        # If no recommendations found, show an error message
        st.markdown(f"<p style='color:red;'>User ID {searched_user_id} does not exist.</p>", unsafe_allow_html=True)

else:
    # Display random user recommendations if no user is searched
    recommendations = fetch_user_recommendations()
    if recommendations:
        random_user_id = recommendations[0][0]  # Assuming all recommendations are for the same random user
        st.subheader(f"Random Movie Recommendations for User {random_user_id}:")  # Show the user ID
        cols = st.columns(5)  # Create columns for displaying movies side by side
        for idx, rec in enumerate(recommendations):
            user_id = rec[0]
            movie_id = rec[1]
            title = rec[2]
            tmdb_id = rec[3]
            rating = rec[4]
            movie_poster = fetch_movie_poster(tmdb_id)

            with cols[idx % 5]:  # Display in columns, cycling through the columns
                st.markdown(f"""
                <div style='margin-bottom: 20px; margin-right: 20px;'>
                    <img src="{movie_poster}" width="150"/>
                    <p>{title} (Rating: {round(rating, 2)})</p>
                </div>
                """, unsafe_allow_html=True)
