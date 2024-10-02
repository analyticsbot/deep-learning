import streamlit as st
import requests
import pandas as pd

st.title("Movie Recommendations")

user_id = st.text_input("Enter your User ID:")
if user_id:
    # Fetch recommendations from Spark output
    recommendations = requests.get(f"http://localhost:5001/recommendations?user_id={user_id}")
    st.write(recommendations.json())

# Display movie images if available
if recommendations:
    for movie_id in recommendations['movie_ids']:
        image_url = f"https://api.themoviedb.org/3/movie/{movie_id}/images?api_key=25255ca7828db7ab3a83f28535c38450"
        st.image(image_url)
