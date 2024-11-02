import pandas as pd
import requests
import gzip
import os
import io
from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel
import torch
from qdrant_client.http import models
import ast

# Initialize CLIP model and processor for text embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Function to download and load a gzipped JSONL file into a DataFrame with sampling
def load_jsonl_to_dataframe(url, sample_size=None):
    response = requests.get(url)
    if response.status_code == 200:
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            df = pd.read_json(f, lines=True)
            if sample_size is not None and sample_size < df.shape[0]:
                return df.sample(n=sample_size, random_state=1)
            else:
                return df
    else:
        raise Exception(f"Failed to download data: {response.status_code}")

# Function to save DataFrame to a specified directory
def save_dataframe_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

# Function to get text embeddings using CLIP
def get_clip_embeddings(texts):
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
    return embeddings

# Function to index data into Qdrant
def index_data_into_qdrant(qdrant_client, df, collection_name):
    # Create collection if it doesn't exist
    if not qdrant_client.has_collection(collection_name):
        qdrant_client.create_collection(collection_name, vector_size=512, distance=models.Distance.COSINE)

    # Prepare embeddings for titles and descriptions
    titles = df['title'].tolist()
    descriptions = df['description'].tolist()
    
    # Generate embeddings
    embeddings = get_clip_embeddings(titles + descriptions)

    # Prepare points for Qdrant
    points = []
    for i, row in df.iterrows():
        # Calculate the embedding for the current row
        embedding = embeddings[i].tolist()  # Convert tensor to list
        
        # Prepare the payload with all specified fields
        payload = {
            "title": row['title'],
            "description": row['description'],
            "images": row['images'],
            "average_rating": row['average_rating'],
            "rating_number": row['rating_number'],
            "price": f"${float(row['price']):.2f}" if pd.notna(row['price']) and isinstance(row['price'], (int, float, str)) and str(row['price']).replace('.', '', 1).isdigit() else "Price not available",
            "details": {key: value for key, value in ast.literal_eval(row['details']).items() if key.strip()} or {"Not Available": "Not Available"},
            "main_category": row['main_category'] if pd.notna(row['main_category']) and row['main_category'].strip() else "Not Available"
        }
        
        points.append(models.PointStruct(id=i, vector=embedding, payload=payload))
    
    # Upsert points into Qdrant
    qdrant_client.upsert(collection_name=collection_name, points=points)

# Function to check if a Qdrant index (collection) exists
def check_index_exists(qdrant_client, collection_name):
    return qdrant_client.has_collection(collection_name)

# Function to get results from Qdrant
def get_qdrant_results(query, meta_url, directory, sample_size, collection_name, top_n=5):
    qdrant_client = QdrantClient(url='http://localhost:6333')

    # Get query embedding
    query_embedding = get_clip_embeddings([query])[0].tolist()  # Convert tensor to list

    # Save DataFrame to CSV
    meta_file_path = os.path.join(directory, "Digital_Music_Meta.csv")
    if not os.path.exists(meta_file_path):
        digital_music_meta_df = load_jsonl_to_dataframe(meta_url, sample_size=sample_size)
        save_dataframe_to_csv(digital_music_meta_df, meta_file_path)
    else:
        digital_music_meta_df = pd.read_csv(meta_file_path)

    digital_music_meta_df.fillna('', inplace=True)

    if not check_index_exists(qdrant_client, collection_name):
        # build
        index_data_into_qdrant(qdrant_client, digital_music_meta_df, collection_name)

    # Search Qdrant
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_n
    )
    
    # Convert search results to a DataFrame
    results_df = pd.DataFrame([{
        'id': hit.id,
        'score': hit.score,
        **hit.payload
    } for hit in search_result])
    
    return results_df

