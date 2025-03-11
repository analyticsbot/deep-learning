"""
Semantic similarity-based recommendation module.
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from recommend.config import (
    clip_model_name,
    clip_vector_dim,
    collection_name_semantic,
    directory,
    meta_url,
    num_recommendations,
    sample_size,
)
from recommend.utils import get_main_image_url, load_or_download_data, parse_image_data
from transformers import CLIPModel, CLIPProcessor


class SemanticRecommender:
    """
    Recommender system based on semantic similarity using CLIP text embeddings.
    """

    def __init__(self):
        """Initialize the semantic recommender with CLIP model."""
        # Initialize CLIP model for text embeddings
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url="http://localhost:6333")

        # Check if collection exists
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self) -> None:
        """Create Qdrant collection if it doesn't exist."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        if collection_name_semantic not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=collection_name_semantic,
                vectors_config=models.VectorParams(
                    size=clip_vector_dim, distance=models.Distance.COSINE
                ),
            )

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get CLIP text embedding for a given text.

        Args:
            text: Input text

        Returns:
            Text embedding as numpy array
        """
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embedding = self.clip_model.get_text_features(**inputs)
        return embedding.cpu().numpy()[0]

    def build_index(self) -> None:
        """Build the semantic index using product titles and descriptions."""
        # Load data
        df = load_or_download_data(meta_url, directory, sample_size)

        # Check if collection is already populated
        collection_info = self.qdrant_client.get_collection(collection_name_semantic)
        if collection_info.vectors_count > 0:
            print(
                f"Collection {collection_name_semantic} already contains {collection_info.vectors_count} vectors."
            )
            return

        # Process data in batches to avoid memory issues
        batch_size = 100
        total_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            points = []
            for i, row in batch_df.iterrows():
                # Create combined text from title and description
                combined_text = f"{row['title']} {row['description']}"

                # Get embedding
                embedding = self.get_text_embedding(combined_text)

                # Parse image data
                images = parse_image_data(row["images"])
                main_image_url = get_main_image_url(images)

                # Create payload
                payload = {
                    "id": i,
                    "title": row["title"],
                    "description": row["description"],
                    "main_image_url": main_image_url,
                    "images": row["images"],
                    "average_rating": float(row["average_rating"])
                    if pd.notna(row["average_rating"])
                    else 0.0,
                    "rating_number": int(row["rating_number"])
                    if pd.notna(row["rating_number"])
                    else 0,
                    "price": float(row["price"]) if pd.notna(row["price"]) else 0.0,
                    "main_category": row["main_category"] if pd.notna(row["main_category"]) else "",
                }

                points.append(models.PointStruct(id=i, vector=embedding.tolist(), payload=payload))

            # Upsert batch to Qdrant
            if points:
                self.qdrant_client.upsert(collection_name=collection_name_semantic, points=points)

            print(f"Indexed batch {batch_idx+1}/{total_batches} ({len(points)} items)")

    def check_index_exists(self) -> bool:
        """
        Check if the semantic index exists and is populated.

        Returns:
            True if index exists and is populated, False otherwise
        """
        if not self.qdrant_client.has_collection(collection_name_semantic):
            return False

        collection_info = self.qdrant_client.get_collection(collection_name_semantic)
        return collection_info.vectors_count > 0

    def recommend_similar_products(self, product_id: int) -> pd.DataFrame:
        """
        Recommend similar products based on semantic similarity.

        Args:
            product_id: ID of the product to find similar items for

        Returns:
            DataFrame containing recommended products
        """
        # Load data to get the product details
        df = load_or_download_data(meta_url, directory, sample_size)

        if product_id >= len(df):
            raise ValueError(f"Product ID {product_id} is out of range (max: {len(df)-1})")

        # Get the product details
        product = df.iloc[product_id]

        # Create combined text from title and description
        combined_text = f"{product['title']} {product['description']}"

        # Get embedding
        embedding = self.get_text_embedding(combined_text)

        # Search for similar products
        search_result = self.qdrant_client.search(
            collection_name=collection_name_semantic,
            query_vector=embedding.tolist(),
            limit=num_recommendations + 1,  # +1 because the product itself will be included
        )

        # Convert search results to DataFrame
        results = []
        for hit in search_result:
            # Skip the product itself
            if hit.payload["id"] == product_id:
                continue

            results.append(
                {
                    "id": hit.payload["id"],
                    "title": hit.payload["title"],
                    "description": hit.payload["description"],
                    "main_image_url": hit.payload["main_image_url"],
                    "average_rating": hit.payload["average_rating"],
                    "rating_number": hit.payload["rating_number"],
                    "price": hit.payload["price"],
                    "main_category": hit.payload["main_category"],
                    "similarity_score": hit.score,
                }
            )

        return pd.DataFrame(results[:num_recommendations])

    def recommend_by_text_query(self, query: str) -> pd.DataFrame:
        """
        Recommend products based on a text query.

        Args:
            query: Text query

        Returns:
            DataFrame containing recommended products
        """
        # Get embedding for the query
        embedding = self.get_text_embedding(query)

        # Search for similar products
        search_result = self.qdrant_client.search(
            collection_name=collection_name_semantic,
            query_vector=embedding.tolist(),
            limit=num_recommendations,
        )

        # Convert search results to DataFrame
        results = []
        for hit in search_result:
            results.append(
                {
                    "id": hit.payload["id"],
                    "title": hit.payload["title"],
                    "description": hit.payload["description"],
                    "main_image_url": hit.payload["main_image_url"],
                    "average_rating": hit.payload["average_rating"],
                    "rating_number": hit.payload["rating_number"],
                    "price": hit.payload["price"],
                    "main_category": hit.payload["main_category"],
                    "similarity_score": hit.score,
                }
            )

        return pd.DataFrame(results)
