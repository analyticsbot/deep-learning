"""
Two-tower model recommendation module.
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from recommend.config import (
    collection_name_two_tower,
    directory,
    meta_url,
    num_recommendations,
    sample_size,
    two_tower_model_name,
    two_tower_vector_dim,
)
from recommend.utils import get_main_image_url, load_or_download_data, parse_image_data
from sentence_transformers import SentenceTransformer


class TwoTowerRecommender:
    """
    Recommender system based on a two-tower model architecture.

    This model uses separate encoders for query and item representations,
    allowing for efficient retrieval of relevant items.
    """

    def __init__(self):
        """Initialize the two-tower recommender with SentenceTransformer model."""
        # Initialize SentenceTransformer model
        self.model = SentenceTransformer(two_tower_model_name)

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url="http://localhost:6333")

        # Check if collection exists
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self) -> None:
        """Create Qdrant collection if it doesn't exist."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        if collection_name_two_tower not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=collection_name_two_tower,
                vectors_config=models.VectorParams(
                    size=two_tower_vector_dim, distance=models.Distance.COSINE
                ),
            )

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a given text using SentenceTransformer.

        Args:
            text: Input text

        Returns:
            Text embedding as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)

    def build_index(self) -> None:
        """Build the two-tower index using product titles, descriptions, and categories."""
        # Load data
        df = load_or_download_data(meta_url, directory, sample_size)

        # Check if collection is already populated
        collection_info = self.qdrant_client.get_collection(collection_name_two_tower)
        if collection_info.vectors_count > 0:
            print(
                f"Collection {collection_name_two_tower} already contains {collection_info.vectors_count} vectors."
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
                # Create item representation text
                item_text = f"Title: {row['title']} Description: {row['description']} Category: {row['main_category']}"

                # Get embedding
                embedding = self.get_embedding(item_text)

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
                self.qdrant_client.upsert(collection_name=collection_name_two_tower, points=points)

            print(f"Indexed batch {batch_idx+1}/{total_batches} ({len(points)} items)")

    def check_index_exists(self) -> bool:
        """
        Check if the two-tower index exists and is populated.

        Returns:
            True if index exists and is populated, False otherwise
        """
        if not self.qdrant_client.has_collection(collection_name_two_tower):
            return False

        collection_info = self.qdrant_client.get_collection(collection_name_two_tower)
        return collection_info.vectors_count > 0

    def recommend_similar_products(self, product_id: int) -> pd.DataFrame:
        """
        Recommend similar products based on two-tower model similarity.

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

        # Create item representation text
        item_text = f"Title: {product['title']} Description: {product['description']} Category: {product['main_category']}"

        # Get embedding
        embedding = self.get_embedding(item_text)

        # Search for similar products
        search_result = self.qdrant_client.search(
            collection_name=collection_name_two_tower,
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

    def recommend_by_query(self, query: str) -> pd.DataFrame:
        """
        Recommend products based on a query.

        Args:
            query: Query text

        Returns:
            DataFrame containing recommended products
        """
        # Get embedding for the query
        embedding = self.get_embedding(query)

        # Search for similar products
        search_result = self.qdrant_client.search(
            collection_name=collection_name_two_tower,
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

    def recommend_personalized(
        self, user_history: List[int], user_query: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Recommend products based on user history and optional query.

        Args:
            user_history: List of product IDs the user has interacted with
            user_query: Optional query text

        Returns:
            DataFrame containing recommended products
        """
        # Load data
        df = load_or_download_data(meta_url, directory, sample_size)

        # Get products from user history
        history_products = []
        for product_id in user_history:
            if product_id < len(df):
                product = df.iloc[product_id]
                history_products.append(product)

        if not history_products:
            if user_query:
                # If no valid history but query exists, use query-based recommendation
                return self.recommend_by_query(user_query)
            else:
                raise ValueError("No valid products in user history and no query provided")

        # Create combined representation from history
        history_texts = []
        for product in history_products:
            item_text = f"Title: {product['title']} Category: {product['main_category']}"
            history_texts.append(item_text)

        combined_history = " ".join(history_texts)

        # Combine with query if provided
        if user_query:
            query_text = f"Query: {user_query} History: {combined_history}"
        else:
            query_text = f"History: {combined_history}"

        # Get embedding
        embedding = self.get_embedding(query_text)

        # Search for similar products
        search_result = self.qdrant_client.search(
            collection_name=collection_name_two_tower,
            query_vector=embedding.tolist(),
            limit=num_recommendations + len(user_history),  # Add extra to filter out history items
        )

        # Convert search results to DataFrame, filtering out history items
        results = []
        for hit in search_result:
            if hit.payload["id"] not in user_history:
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
