import streamlit as st
import psycopg2
import random
import pandas as pd

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="postgres",         # e.g., "localhost" or your PostgreSQL container name
        database="recommendations",       # Your database name
        user="admin",     # Your PostgreSQL username
        password="airflow"  # Your PostgreSQL password
    )

# Function to fetch recommendations from the database
def fetch_user_recommendations(user_id=None):
    conn = get_db_connection()
    cur = conn.cursor()

    if user_id:
        # Fetch recommendations for a specific user
        cur.execute("""
            SELECT r.movie_title, r.image_url 
            FROM recommendations r 
            WHERE r.user_id = %s
            """, (user_id,))
    else:
        # Fetch recommendations for 5 random users
        cur.execute("""
            SELECT DISTINCT r.user_id 
            FROM recommendations r
            ORDER BY RANDOM() LIMIT 5
            """)
        random_users = cur.fetchall()
        return [user[0] for user in random_users]  # Return a list of random user_ids

    recommendations = cur.fetchall()  # Fetch recommendations
    cur.close()
    conn.close()
    return recommendations

# Title of the application
st.title("Personalized Movie Recommendations")

# 1. Default Random Users and Recommendations
st.header("Random User Recommendations")

# Get 5 random users and display their recommendations
random_users = fetch_user_recommendations()
for user_id in random_users:
    st.subheader(f"Recommendations for User {user_id}:")
    recommendations = fetch_user_recommendations(user_id=user_id)
    for rec in recommendations:
        st.image(rec[1], width=150, caption=rec[0])  # rec[0] is movie title, rec[1] is image URL

# 2. Searchable User Input
st.header("Search for User Recommendations")

# Input to search for a specific user by ID
searched_user_id = st.text_input("Enter your User ID to see your recommendations:")

if searched_user_id:
    recommendations = fetch_user_recommendations(user_id=searched_user_id)
    if recommendations:
        st.subheader(f"Recommendations for User {searched_user_id}:")
        for rec in recommendations:
            st.image(rec[1], width=150, caption=rec[0])
    else:
        st.write("No recommendations found for this user.")

# 3. Clickable List of Users (optional, based on your dataset size)
st.header("Click to See Recommendations")

# You can dynamically generate buttons for a list of users, if your dataset is small:
for user_id in random_users:
    if st.button(f"View Recommendations for User {user_id}"):
        st.subheader(f"Recommendations for User {user_id}:")
        recommendations = fetch_user_recommendations(user_id=user_id)
        for rec in recommendations:
            st.image(rec[1], width=150, caption=rec[0])

