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
        password="airflow",  # Your PostgreSQL password,
        port=5432
    )

# Function to fetch recommendations from the database
def fetch_user_recommendations(user_id=None):
    conn = get_db_connection()
    cur = conn.cursor()

    if user_id:
        # Fetch recommendations for a specific user
        cur.execute("""
            SELECT r.user_id, r.movie_id, r.title, r.tmdb_id, r.rating
            FROM recommendations r 
            WHERE r.user_id = %s
            """, (user_id,))
    else:
        # Fetch recommendations for 5 random users
        cur.execute("""
            SELECT DISTINCT r.user_id, r.movie_id, r.title, r.tmdb_id, r.rating
                FROM (
                    SELECT r.user_id, r.movie_id, r.title, r.tmdb_id, r.rating
                    FROM recommendations r
                    ORDER BY r.user_id, RANDOM()
                    LIMIT 5
                ) AS r;
            """)

    recommendations = cur.fetchall()  # Fetch recommendations
    cur.close()
    conn.close()
    return recommendations

# Fetch movie poster from TMDb using the movie's TMDb ID
def fetch_movie_poster(tmdb_id):
    try:
        response = requests.get(f'{TMDB_API_URL}/movie/{tmdb_id}/images?api_key={TMDB_API_KEY}', timeout=10,  verify=False        )

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

# 1. Default Random Users and Recommendations
st.subheader("Random User Recommendations")
recommendations = fetch_user_recommendations()

# Create columns for displaying movies side by side
cols = st.columns(5)  # Adjust the number of columns as needed
user_seen = []
for idx, rec in enumerate(recommendations):
    # r.user_id, r.movie_id, r.title, r.tmdb_id, r.rating
    user_id = rec[0]
    movie_id = rec[1]
    title = rec[2]
    tmdb_id = rec[3]
    rating = rec[4]
    movie_poster = fetch_movie_poster(tmdb_id)
    if user_id not in user_seen:
        st.subheader(f"Recommendations for User {user_id}:")
        user_seen.append(user_id)
    
    with cols[idx % 5]:  # Display in columns, cycling through the columns
        # Display the movie poster and title along with the rating
        #st.header("Random User Recommendations {user_id}")
        st.markdown(f"""
                <div style='margin-bottom: 20px; margin-right: 20px;'>
                    <img src="{movie_poster}" width="150"/>
                    <p>{title} <br>(Rating: {round(rating, 2)})</p>
                </div>
            """, unsafe_allow_html=True)

# 2. Searchable User Input
st.subheader("Search for User Recommendations")

# Input to search for a specific user by ID
searched_user_id = st.text_input("Enter your User ID to see your recommendations:")

if searched_user_id:
    # Check if the user exists
    recommendations = fetch_user_recommendations(user_id=searched_user_id)
    if recommendations:
        st.subheader(f"Recommendations for User {searched_user_id}:")
        for idx, rec in enumerate(recommendations):
            # r.user_id, r.movie_id, r.title, r.tmdb_id, r.rating
            user_id = rec[0]
            movie_id = rec[1]
            title = rec[2]
            tmdb_id = rec[3]
            rating = rec[4]
            movie_poster = fetch_movie_poster(tmdb_id)
            if user_id not in user_seen:
                st.subheader(f"Recommendations for User {user_id}:")
                user_seen.append(user_id)
            
            with cols[idx % 5]:  # Display in columns, cycling through the columns
                # Display the movie poster and title along with the rating
                #st.header("Random User Recommendations {user_id}")
                st.markdown(f"""
                <div style='margin-bottom: 20px; margin-right: 20px;'>
                    <img src="{movie_poster}" width="150"/>
                    <p>{title} (Rating: {round(rating, 2)})</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        # If no recommendations found, show an error message
        st.markdown(f"<p style='color:red;'>User ID {searched_user_id} does not exist.</p>", unsafe_allow_html=True)


