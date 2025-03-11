"""
Image similarity-based recommendation module.
"""

import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from recommend.config import (
    clip_model_name,
    clip_vector_dim,
    collection_name_image,
    directory,
    meta_url,
    num_recommendations,
    sample_size,
)
from recommend.utils import get_main_image_url, load_or_download_data, parse_image_data
from transformers import CLIPModel, CLIPProcessor


class ImageRecommender:
    """
    Recommender system based on image similarity using CLIP image embeddings.
    """

    def __init__(self):
        """Initialize the image recommender with CLIP model."""
        # Initialize CLIP model for image embeddings
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
        if collection_name_image not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=collection_name_image,
                vectors_config=models.VectorParams(
                    size=clip_vector_dim, distance=models.Distance.COSINE
                ),
            )

    def get_image_embedding(self, image_url: str) -> Optional[np.ndarray]:
        """
        Get CLIP image embedding for a given image URL.

        Args:
            image_url: URL of the image

        Returns:
            Image embedding as numpy array, or None if the image couldn't be processed
        """
        try:
            # Download the image
            response = requests.get(image_url, timeout=5)
            if response.status_code != 200:
                print(f"Failed to download image from {image_url}: {response.status_code}")
                return None

            # Open the image
            image = Image.open(BytesIO(response.content)).convert("RGB")

            # Process the image
            inputs = self.clip_processor(images=image, return_tensors="pt")

            # Get the embedding
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(**inputs)

            return embedding.cpu().numpy()[0]

        except Exception as e:
            print(f"Error processing image from {image_url}: {e}")
            return None

    def build_index(self) -> None:
        """Build the image index using product images."""
        # Load data
        df = load_or_download_data(meta_url, directory, sample_size)

        # Check if collection is already populated
        collection_info = self.qdrant_client.get_collection(collection_name_image)
        if collection_info.vectors_count > 0:
            print(
                f"Collection {collection_name_image} already contains {collection_info.vectors_count} vectors."
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
                # Parse image data
                images = parse_image_data(row["images"])
                main_image_url = get_main_image_url(images)

                # Skip if no image is available
                if not main_image_url:
                    continue

                # Get image embedding
                embedding = self.get_image_embedding(main_image_url)

                # Skip if embedding couldn't be generated
                if embedding is None:
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

                points.append(models.PointStruct(id=i, vector=embedding.tolist(), payload=payload))

            # Upsert batch to Qdrant
            if points:
                self.qdrant_client.upsert(collection_name=collection_name_image, points=points)

            print(f"Indexed batch {batch_idx+1}/{total_batches} ({len(points)} items)")

    def check_index_exists(self) -> bool:
        """
        Check if the image index exists and is populated.

        Returns:
            True if index exists and is populated, False otherwise
        """
        if not self.qdrant_client.has_collection(collection_name_image):
            return False

        collection_info = self.qdrant_client.get_collection(collection_name_image)
        return collection_info.vectors_count > 0

    def recommend_similar_products(self, product_id: int) -> pd.DataFrame:
        """
        Recommend similar products based on image similarity.

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

        # Parse image data
        images = parse_image_data(product["images"])
        main_image_url = get_main_image_url(images)

        if not main_image_url:
            raise ValueError(f"Product {product_id} does not have a valid image URL")

        # Get image embedding
        embedding = self.get_image_embedding(main_image_url)

        if embedding is None:
            raise ValueError(f"Failed to generate embedding for image {main_image_url}")

        # Search for similar products
        search_result = self.qdrant_client.search(
            collection_name=collection_name_image,
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

    def recommend_by_image_url(self, image_url: str) -> pd.DataFrame:
        """
        Recommend products based on an image URL.

        Args:
            image_url: URL of the image to find similar products for

        Returns:
            DataFrame containing recommended products
        """
        # Get image embedding
        embedding = self.get_image_embedding(image_url)

        if embedding is None:
            raise ValueError(f"Failed to generate embedding for image {image_url}")

        # Search for similar products
        search_result = self.qdrant_client.search(
            collection_name=collection_name_image,
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
