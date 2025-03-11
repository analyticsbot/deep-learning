"""
Streamlit app for demonstrating recommendation methods.
"""

import ast
import json
import os
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from recommend.config import directory, meta_url, sample_size
from recommend.hybrid_recommender import HybridRecommender
from recommend.image_similarity import ImageRecommender
from recommend.semantic_similarity import SemanticRecommender
from recommend.two_tower_model import TwoTowerRecommender
from recommend.utils import get_main_image_url, load_or_download_data, parse_image_data

# Set page title and layout
st.set_page_config(page_title="Amazon Product Recommendations", page_icon="🛒", layout="wide")

# Title and description
st.title("Amazon Product Recommendations")
st.markdown(
    """
This app demonstrates different recommendation methods for Amazon products:
- **Semantic Similarity**: Recommends products based on text similarity using CLIP embeddings
- **Image Similarity**: Recommends products based on image similarity using CLIP embeddings
- **Hybrid Recommender**: Combines semantic and image similarity for recommendations
- **Two-Tower Model**: Uses a two-tower architecture for personalized recommendations
"""
)

# Initialize session state for user history
if "user_history" not in st.session_state:
    st.session_state.user_history = []

# Sidebar for method selection
st.sidebar.title("Recommendation Settings")
method = st.sidebar.selectbox(
    "Select Recommendation Method",
    ["Semantic Similarity", "Image Similarity", "Hybrid Recommender", "Two-Tower Model"],
)


# Function to display a product
def display_product(product, add_to_history_button=False):
    col1, col2 = st.columns([1, 3])

    with col1:
        # Display product image
        if "main_image_url" in product and product["main_image_url"]:
            try:
                response = requests.get(product["main_image_url"], timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=150)
                else:
                    st.info("Image not available")
            except Exception as e:
                st.info("Image not available")
        else:
            # Try to parse images from the images field
            try:
                images = parse_image_data(product["images"])
                main_image_url = get_main_image_url(images)
                if main_image_url:
                    response = requests.get(main_image_url, timeout=10)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=150)
                    else:
                        st.info("Image not available")
                else:
                    st.info("Image not available")
            except Exception as e:
                st.info("Image not available")

    with col2:
        # Display product details
        st.subheader(product["title"])

        # Description
        if "description" in product and product["description"]:
            desc = product["description"]
            if isinstance(desc, list):
                desc = ", ".join(desc)
            st.write(f"**Description:** {desc}")

        # Rating
        if "average_rating" in product and product["average_rating"]:
            rating = float(product["average_rating"])
            rating_count = (
                int(product["rating_number"])
                if "rating_number" in product and product["rating_number"]
                else 0
            )
            st.write(f"**Rating:** {rating:.1f} ({rating_count} ratings)")

        # Price
        if "price" in product and product["price"]:
            price = (
                float(product["price"])
                if not isinstance(product["price"], str)
                else product["price"]
            )
            st.write(f"**Price:** ${price}")

        # Category
        if "main_category" in product and product["main_category"]:
            st.write(f"**Category:** {product['main_category']}")

        # Similarity score
        if "similarity_score" in product:
            st.write(f"**Similarity Score:** {float(product['similarity_score']):.4f}")

        # Add to history button
        if add_to_history_button and "id" in product:
            if st.button(f"Add to History", key=f"add_{product['id']}"):
                product_id = int(product["id"])
                if product_id not in st.session_state.user_history:
                    st.session_state.user_history.append(product_id)
                    st.success(f"Added product to history!")

    st.markdown("---")


# Load data
@st.cache_data
def get_data():
    return load_or_download_data(meta_url, directory, sample_size)


# Create data directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Load data
with st.spinner("Loading product data..."):
    df = get_data()

# Main content area
if method == "Semantic Similarity":
    st.header("Semantic Similarity Recommendations")

    # Initialize recommender
    recommender = SemanticRecommender()

    # Check if index exists
    index_exists = recommender.check_index_exists()

    if not index_exists:
        st.warning("Semantic index does not exist. Building index...")
        with st.spinner("Building semantic index (this may take a while)..."):
            recommender.build_index()
        st.success("Semantic index built successfully!")

    # Tabs for different recommendation modes
    tab1, tab2 = st.tabs(["Recommend by Product", "Recommend by Query"])

    with tab1:
        st.subheader("Find Similar Products")

        # Product selection
        product_id = st.number_input(
            "Enter Product ID", min_value=0, max_value=len(df) - 1, value=42
        )

        if st.button("Find Similar Products"):
            with st.spinner("Finding similar products..."):
                # Display selected product
                st.subheader("Selected Product")
                display_product(df.iloc[product_id])

                # Get recommendations
                recommendations = recommender.recommend_similar_products(product_id)

                # Display recommendations
                st.subheader("Recommended Products")
                for _, product in recommendations.iterrows():
                    display_product(product, add_to_history_button=True)

    with tab2:
        st.subheader("Search by Query")

        # Query input
        query = st.text_input("Enter Search Query", "classical piano music")

        if st.button("Search"):
            with st.spinner("Searching..."):
                # Get recommendations
                recommendations = recommender.recommend_by_text_query(query)

                # Display recommendations
                st.subheader("Recommended Products")
                for _, product in recommendations.iterrows():
                    display_product(product, add_to_history_button=True)

elif method == "Image Similarity":
    st.header("Image Similarity Recommendations")

    # Initialize recommender
    image_recommender = ImageRecommender()

    # Check if index exists
    index_exists = image_recommender.check_index_exists()

    if not index_exists:
        st.warning("Image index does not exist. Building index...")
        with st.spinner("Building image index (this may take a while)..."):
            image_recommender.build_index()
        st.success("Image index built successfully!")

    # Product selection
    product_id = st.number_input("Enter Product ID", min_value=0, max_value=len(df) - 1, value=42)

    if st.button("Find Similar Products"):
        # Find a product with a valid image
        valid_image = False
        for i in range(product_id, len(df)):
            row = df.iloc[i]
            if pd.notna(row["images"]) and row["images"]:
                try:
                    images = parse_image_data(row["images"])
                    main_image_url = get_main_image_url(images)
                    if main_image_url:
                        product_id = i
                        valid_image = True
                        break
                except Exception:
                    continue

        if not valid_image:
            st.error("No product with valid image found starting from the selected ID.")
        else:
            with st.spinner("Finding similar products..."):
                # Display selected product
                st.subheader("Selected Product")
                display_product(df.iloc[product_id])

                try:
                    # Get recommendations
                    recommendations = image_recommender.recommend_similar_products(product_id)

                    # Display recommendations
                    st.subheader("Recommended Products")
                    for _, product in recommendations.iterrows():
                        display_product(product, add_to_history_button=True)
                except Exception as e:
                    st.error(f"Error getting image recommendations: {str(e)}")

elif method == "Hybrid Recommender":
    st.header("Hybrid Recommender")

    # Initialize recommender
    semantic_weight = st.sidebar.slider("Semantic Weight", 0.0, 1.0, 0.5, 0.1)
    hybrid_recommender = HybridRecommender(semantic_weight=semantic_weight)

    # Check if index exists
    index_exists = hybrid_recommender.check_index_exists()

    if not index_exists:
        st.warning("Hybrid index does not exist. Building index...")
        with st.spinner("Building hybrid index (this may take a while)..."):
            hybrid_recommender.build_index()
        st.success("Hybrid index built successfully!")

    # Product selection
    product_id = st.number_input("Enter Product ID", min_value=0, max_value=len(df) - 1, value=42)

    if st.button("Find Similar Products"):
        # Find a product with a valid image
        valid_image = False
        for i in range(product_id, len(df)):
            row = df.iloc[i]
            if pd.notna(row["images"]) and row["images"]:
                try:
                    images = parse_image_data(row["images"])
                    main_image_url = get_main_image_url(images)
                    if main_image_url:
                        product_id = i
                        valid_image = True
                        break
                except Exception:
                    continue

        if not valid_image:
            st.error("No product with valid image found starting from the selected ID.")
        else:
            with st.spinner("Finding similar products..."):
                # Display selected product
                st.subheader("Selected Product")
                display_product(df.iloc[product_id])

                try:
                    # Get recommendations
                    recommendations = hybrid_recommender.recommend_similar_products(product_id)

                    # Display recommendations
                    st.subheader("Recommended Products")
                    for _, product in recommendations.iterrows():
                        display_product(product, add_to_history_button=True)
                except Exception as e:
                    st.error(f"Error getting hybrid recommendations: {str(e)}")

elif method == "Two-Tower Model":
    st.header("Two-Tower Model Recommendations")

    # Initialize recommender
    two_tower_recommender = TwoTowerRecommender()

    # Check if index exists
    index_exists = two_tower_recommender.check_index_exists()

    if not index_exists:
        st.warning("Two-tower index does not exist. Building index...")
        with st.spinner("Building two-tower index (this may take a while)..."):
            two_tower_recommender.build_index()
        st.success("Two-tower index built successfully!")

    # Tabs for different recommendation modes
    tab1, tab2, tab3 = st.tabs(
        ["Recommend by Product", "Recommend by Query", "Personalized Recommendations"]
    )

    with tab1:
        st.subheader("Find Similar Products")

        # Product selection
        product_id = st.number_input(
            "Enter Product ID", min_value=0, max_value=len(df) - 1, value=42
        )

        if st.button("Find Similar Products", key="find_similar_two_tower"):
            with st.spinner("Finding similar products..."):
                # Display selected product
                st.subheader("Selected Product")
                display_product(df.iloc[product_id])

                # Get recommendations
                recommendations = two_tower_recommender.recommend_similar_products(product_id)

                # Display recommendations
                st.subheader("Recommended Products")
                for _, product in recommendations.iterrows():
                    display_product(product, add_to_history_button=True)

    with tab2:
        st.subheader("Search by Query")

        # Query input
        query = st.text_input("Enter Search Query", "jazz music with saxophone")

        if st.button("Search", key="search_two_tower"):
            with st.spinner("Searching..."):
                # Get recommendations
                recommendations = two_tower_recommender.recommend_by_query(query)

                # Display recommendations
                st.subheader("Recommended Products")
                for _, product in recommendations.iterrows():
                    display_product(product, add_to_history_button=True)

    with tab3:
        st.subheader("Personalized Recommendations")

        # Display user history
        st.write("**Your History:**")
        if not st.session_state.user_history:
            st.info(
                "Your history is empty. Add products to your history by clicking 'Add to History' on any product."
            )
        else:
            for product_id in st.session_state.user_history:
                if product_id < len(df):
                    with st.expander(f"Product {product_id}: {df.iloc[product_id]['title']}"):
                        display_product(df.iloc[product_id])

        # Optional query for personalized recommendations
        query = st.text_input("Optional Query for Personalized Recommendations", "")

        if st.button("Get Personalized Recommendations"):
            if not st.session_state.user_history:
                st.warning("Please add products to your history first.")
            else:
                with st.spinner("Getting personalized recommendations..."):
                    try:
                        # Get recommendations
                        recommendations = two_tower_recommender.recommend_personalized(
                            st.session_state.user_history, user_query=query if query else None
                        )

                        # Display recommendations
                        st.subheader("Recommended Products")
                        for _, product in recommendations.iterrows():
                            display_product(product, add_to_history_button=True)
                    except Exception as e:
                        st.error(f"Error getting personalized recommendations: {str(e)}")

# Sidebar - User History
st.sidebar.header("Your History")
if not st.session_state.user_history:
    st.sidebar.info("No products in history")
else:
    for product_id in st.session_state.user_history:
        if product_id < len(df):
            st.sidebar.write(f"- {df.iloc[product_id]['title']}")

    if st.sidebar.button("Clear History"):
        st.session_state.user_history = []
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Amazon Product Recommendation Demo | Built with Streamlit")
