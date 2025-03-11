"""
Hybrid recommendation module combining semantic and image similarity.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from recommend.config import (
    clip_vector_dim,
    collection_name_hybrid,
    directory,
    meta_url,
    num_recommendations,
    sample_size,
)
from recommend.image_similarity import ImageRecommender
from recommend.semantic_similarity import SemanticRecommender
from recommend.utils import (
    get_main_image_url,
    load_or_download_data,
    normalize_vector,
    parse_image_data,
)


class HybridRecommender:
    """
    Hybrid recommender system combining semantic and image similarity.
    """

    def __init__(self, semantic_weight: float = 0.5):
        """
        Initialize the hybrid recommender.

        Args:
            semantic_weight: Weight for semantic similarity (0-1), with the remainder assigned to image similarity
        """
        self.semantic_weight = semantic_weight
        self.image_weight = 1.0 - semantic_weight

        # Initialize component recommenders
        self.semantic_recommender = SemanticRecommender()
        self.image_recommender = ImageRecommender()

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url="http://localhost:6333")

        # Check if collection exists
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self) -> None:
        """Create Qdrant collection if it doesn't exist."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        if collection_name_hybrid not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=collection_name_hybrid,
                vectors_config=models.VectorParams(
                    size=clip_vector_dim * 2,  # Combined vector size (text + image)
                    distance=models.Distance.COSINE,
                ),
            )

    def get_combined_embedding(self, text: str, image_url: str) -> Optional[np.ndarray]:
        """
        Get combined embedding from text and image.

        Args:
            text: Input text
            image_url: URL of the image

        Returns:
            Combined embedding as numpy array, or None if embeddings couldn't be generated
        """
        # Get text embedding
        text_embedding = self.semantic_recommender.get_text_embedding(text)

        # Get image embedding
        image_embedding = self.image_recommender.get_image_embedding(image_url)

        if image_embedding is None:
            return None

        # Normalize embeddings
        text_embedding = normalize_vector(text_embedding)
        image_embedding = normalize_vector(image_embedding)

        # Combine embeddings
        combined_embedding = np.concatenate(
            [text_embedding * self.semantic_weight, image_embedding * self.image_weight]
        )

        return combined_embedding

    def build_index(self) -> None:
        """Build the hybrid index combining text and image embeddings."""
        # Load data
        df = load_or_download_data(meta_url, directory, sample_size)

        # Check if collection is already populated
        collection_info = self.qdrant_client.get_collection(collection_name_hybrid)
        if collection_info.vectors_count > 0:
            print(
                f"Collection {collection_name_hybrid} already contains {collection_info.vectors_count} vectors."
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

                # Parse image data
                images = parse_image_data(row["images"])
                main_image_url = get_main_image_url(images)

                # Skip if no image is available
                if not main_image_url:
                    continue

                # Get combined embedding
                combined_embedding = self.get_combined_embedding(combined_text, main_image_url)

                # Skip if embedding couldn't be generated
                if combined_embedding is None:
                    continue

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

                points.append(
                    models.PointStruct(id=i, vector=combined_embedding.tolist(), payload=payload)
                )

            # Upsert batch to Qdrant
            if points:
                self.qdrant_client.upsert(collection_name=collection_name_hybrid, points=points)

            print(f"Indexed batch {batch_idx+1}/{total_batches} ({len(points)} items)")

    def check_index_exists(self) -> bool:
        """
        Check if the hybrid index exists and is populated.

        Returns:
            True if index exists and is populated, False otherwise
        """
        if not self.qdrant_client.has_collection(collection_name_hybrid):
            return False

        collection_info = self.qdrant_client.get_collection(collection_name_hybrid)
        return collection_info.vectors_count > 0

    def recommend_similar_products(self, product_id: int) -> pd.DataFrame:
        """
        Recommend similar products based on combined semantic and image similarity.

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

        # Parse image data
        images = parse_image_data(product["images"])
        main_image_url = get_main_image_url(images)

        if not main_image_url:
            raise ValueError(f"Product {product_id} does not have a valid image URL")

        # Get combined embedding
        combined_embedding = self.get_combined_embedding(combined_text, main_image_url)

        if combined_embedding is None:
            raise ValueError(f"Failed to generate combined embedding for product {product_id}")

        # Search for similar products
        search_result = self.qdrant_client.search(
            collection_name=collection_name_hybrid,
            query_vector=combined_embedding.tolist(),
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

    def recommend_by_text_and_image(self, query: str, image_url: str) -> pd.DataFrame:
        """
        Recommend products based on a text query and an image URL.

        Args:
            query: Text query
            image_url: URL of the image

        Returns:
            DataFrame containing recommended products
        """
        # Get combined embedding
        combined_embedding = self.get_combined_embedding(query, image_url)

        if combined_embedding is None:
            raise ValueError(f"Failed to generate combined embedding")

        # Search for similar products
        search_result = self.qdrant_client.search(
            collection_name=collection_name_hybrid,
            query_vector=combined_embedding.tolist(),
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
