import streamlit as st
import requests
from config import *
from bm25 import get_bm25_results, get_bm_index_status
from elastic_search import get_elasticsearch_results, get_elasticsearch_index_status
import pandas as pd
import ast
import json

# Brief overview of how the search page works
st.title("Multi-Method Search Page")
st.markdown("""
This search page incorporates various search methodologies to provide users with versatile results. 
You can enter a query, and the system will process it using different methods: BM25, Elasticsearch, 
Semantic Embedding, Semantic and Image Embedding, and a Combined Reranking approach. 
Each method uses its own indexing strategy to return relevant results.
""")

# Search input
query = st.text_input("Enter your search query:")

# Create tabs for each method
tabs = ["BM25", "Elasticsearch", "Semantic Embedding", "Semantic and Image Embedding", "Combined Reranking"]
selected_tab = st.sidebar.selectbox("Select Search Method", tabs)

def display_results(results):
    for index, row in results.iterrows():
        st.subheader(row['title'])  # Product title
        # Try to decode the images JSON
        images = []
        if pd.notna(row['images']):
            print(f"Row {index} images data: {row['images']}")  # Print the raw data
            try:
                images = ast.literal_eval(row['images'])  # Attempt to decode the JSON
            except json.JSONDecodeError as e:
                st.warning(f"Error decoding JSON for images in row {index}: {e}. Images data may be malformed.")

        # Display the first image (main image)
        if images:
            main_image_url = images[0]['large'] if 'large' in images[0] else images[0]['thumb']
            st.image(main_image_url, width=150)  # Product image
        st.write(f"**Description:** {', '.join(row['description']) if isinstance(row['description'], list) else row['description']}")  # Product description
        
        # Ratings and pricing information
        average_rating = row['average_rating']  # Directly access the average rating
        rating_number = row['rating_number']  # Directly access the rating count

        if pd.notna(average_rating):
            st.write(f"**Rating:** {average_rating:.1f} ({int(rating_number)} ratings)")
        else:
            st.write("**Rating:** No ratings available")
        
        st.write(f"**Price:** ${row['price']}" if pd.notna(row['price']) else "Price not available")  # Display price
        st.write(f"**Details:** {row['details'] if pd.notna(row['details']) else 'No details available'}")  # Additional product details
        st.write(f"**Category:** {row['main_category']}")  # Product main category
        
        # Display additional metadata if available
        if 'bought_together' in row and pd.notna(row['bought_together']):
            st.write(f"**Frequently Bought Together:** {row['bought_together']}")  # Products bought together

        st.markdown("---")  # Separator line for clarity


if query:
    if selected_tab == "BM25":
        st.header("BM25 Results")
        bm_index = get_bm_index_status(directory, bm_pickle_file)
        print ('bm_index', bm_index)
        # Display the status of the index with color
        if bm_index:
            st.markdown("<h3 style='color: green;'>Index Status: Built</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: red;'>Index Status: Building</h3>", unsafe_allow_html=True)
        # Call BM25 Docker container API to get results
        bm25_results = get_bm25_results(query, meta_url, bm_pickle_file, directory, sample_size)  # Replace with actual API call
        display_results(bm25_results)
        
    elif selected_tab == "Elasticsearch":
        st.header("Elasticsearch Results")
        es_index = get_elasticsearch_index_status(es_index_name)
        print ('es_index', es_index)
        # Display the status of the index with color
        if es_index:
            st.markdown("<h3 style='color: green;'>Index Status: Built</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: red;'>Index Status: Building</h3>", unsafe_allow_html=True)
        # Call BM25 Docker container API to get results
        es_results = get_elasticsearch_results(query, meta_url, es_index_name, directory, sample_size)  # Replace with actual API call
        display_results(es_results)

    elif selected_tab == "Semantic Embedding":
        st.header("Semantic Embedding Results")
        st.write(indices["qdrant_semantic"])  # Show index build status
        # Call Qdrant Docker container API for semantic embedding
        semantic_results = get_semantic_results(query)  # Replace with actual API call
        st.write(semantic_results)

    elif selected_tab == "Semantic and Image Embedding":
        st.header("Semantic and Image Embedding Results")
        st.write(indices["qdrant_image_semantic"])  # Show index build status
        # Call Qdrant Docker container API for image and semantic embedding
        image_semantic_results = get_image_semantic_results(query)  # Replace with actual API call
        st.write(image_semantic_results)

    elif selected_tab == "Combined Reranking":
        st.header("Combined Reranking Results")
        st.write(indices["reranking"])  # Show index build status
        # Call reranking model API to combine results
        reranked_results = get_reranked_results(query)  # Replace with actual API call
        st.write(reranked_results)

# Mock function implementations
def get_bm25_results(query):
    # Simulate API call to BM25 service
    return {"results": [f"BM25 Result for '{query}'"]}

def get_elasticsearch_results(query):
    # Simulate API call to Elasticsearch service
    return {"results": [f"Elasticsearch Result for '{query}'"]}

def get_semantic_results(query):
    # Simulate API call to Qdrant for semantic embedding
    return {"results": [f"Semantic Result for '{query}'"]}

def get_image_semantic_results(query):
    # Simulate API call to Qdrant for image semantic embedding
    return {"results": [f"Image Semantic Result for '{query}'"]}

def get_reranked_results(query):
    # Simulate API call to reranking model
    return {"results": [f"Reranked Result for '{query}'"]}
