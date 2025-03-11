"""
Demo script to test all recommendation methods.
"""

import argparse
import os

import pandas as pd
from recommend.config import directory, meta_url, sample_size
from recommend.hybrid_recommender import HybridRecommender
from recommend.image_similarity import ImageRecommender
from recommend.semantic_similarity import SemanticRecommender
from recommend.two_tower_model import TwoTowerRecommender
from recommend.utils import load_or_download_data


def display_product(product):
    """Display product information in a readable format."""
    print(f"Title: {product['title']}")
    print(f"Description: {product['description']}")
    if "main_image_url" in product and product["main_image_url"]:
        print(f"Image URL: {product['main_image_url']}")
    print(f"Rating: {product['average_rating']} ({product['rating_number']} ratings)")
    print(f"Price: ${product['price']}")
    print(f"Category: {product['main_category']}")
    if "similarity_score" in product:
        print(f"Similarity Score: {product['similarity_score']:.4f}")
    print("-" * 80)


def test_semantic_recommender():
    """Test the semantic recommender."""
    print("\n" + "=" * 80)
    print("TESTING SEMANTIC RECOMMENDER")
    print("=" * 80)

    # Initialize recommender
    recommender = SemanticRecommender()

    # Check if index exists
    index_exists = recommender.check_index_exists()
    print(f"Semantic index exists: {index_exists}")

    if not index_exists:
        print("Building semantic index...")
        recommender.build_index()

    # Load data
    df = load_or_download_data(meta_url, directory, sample_size)

    # Get a random product ID
    product_id = 42  # Using a fixed ID for reproducibility

    # Display the product
    print("\nSelected product:")
    display_product(df.iloc[product_id])

    # Get recommendations
    print("\nRecommendations based on semantic similarity:")
    recommendations = recommender.recommend_similar_products(product_id)

    for _, product in recommendations.iterrows():
        display_product(product)

    # Test text query recommendation
    query = "classical piano music"
    print(f"\nRecommendations for query: '{query}'")
    query_recommendations = recommender.recommend_by_text_query(query)

    for _, product in query_recommendations.iterrows():
        display_product(product)


def test_image_recommender():
    """Test the image recommender."""
    print("\n" + "=" * 80)
    print("TESTING IMAGE RECOMMENDER")
    print("=" * 80)

    # Initialize recommender
    recommender = ImageRecommender()

    # Check if index exists
    index_exists = recommender.check_index_exists()
    print(f"Image index exists: {index_exists}")

    if not index_exists:
        print("Building image index...")
        recommender.build_index()

    # Load data
    df = load_or_download_data(meta_url, directory, sample_size)

    # Find a product with a valid image
    product_id = None
    for i, row in df.iterrows():
        if pd.notna(row["images"]) and row["images"]:
            try:
                import ast

                images = ast.literal_eval(row["images"])
                if images and "large" in images[0]:
                    product_id = i
                    break
            except Exception:
                continue

    if product_id is None:
        print("No product with valid image found.")
        return

    # Display the product
    print("\nSelected product:")
    display_product(df.iloc[product_id])

    # Get recommendations
    print("\nRecommendations based on image similarity:")
    try:
        recommendations = recommender.recommend_similar_products(product_id)

        for _, product in recommendations.iterrows():
            display_product(product)
    except Exception as e:
        print(f"Error getting image recommendations: {e}")


def test_hybrid_recommender():
    """Test the hybrid recommender."""
    print("\n" + "=" * 80)
    print("TESTING HYBRID RECOMMENDER")
    print("=" * 80)

    # Initialize recommender
    recommender = HybridRecommender(semantic_weight=0.6)

    # Check if index exists
    index_exists = recommender.check_index_exists()
    print(f"Hybrid index exists: {index_exists}")

    if not index_exists:
        print("Building hybrid index...")
        recommender.build_index()

    # Load data
    df = load_or_download_data(meta_url, directory, sample_size)

    # Find a product with a valid image
    product_id = None
    for i, row in df.iterrows():
        if pd.notna(row["images"]) and row["images"]:
            try:
                import ast

                images = ast.literal_eval(row["images"])
                if images and "large" in images[0]:
                    product_id = i
                    break
            except Exception:
                continue

    if product_id is None:
        print("No product with valid image found.")
        return

    # Display the product
    print("\nSelected product:")
    display_product(df.iloc[product_id])

    # Get recommendations
    print("\nRecommendations based on hybrid similarity:")
    try:
        recommendations = recommender.recommend_similar_products(product_id)

        for _, product in recommendations.iterrows():
            display_product(product)
    except Exception as e:
        print(f"Error getting hybrid recommendations: {e}")


def test_two_tower_recommender():
    """Test the two-tower recommender."""
    print("\n" + "=" * 80)
    print("TESTING TWO-TOWER RECOMMENDER")
    print("=" * 80)

    # Initialize recommender
    recommender = TwoTowerRecommender()

    # Check if index exists
    index_exists = recommender.check_index_exists()
    print(f"Two-tower index exists: {index_exists}")

    if not index_exists:
        print("Building two-tower index...")
        recommender.build_index()

    # Load data
    df = load_or_download_data(meta_url, directory, sample_size)

    # Get a random product ID
    product_id = 42  # Using a fixed ID for reproducibility

    # Display the product
    print("\nSelected product:")
    display_product(df.iloc[product_id])

    # Get recommendations
    print("\nRecommendations based on two-tower model:")
    recommendations = recommender.recommend_similar_products(product_id)

    for _, product in recommendations.iterrows():
        display_product(product)

    # Test query recommendation
    query = "jazz music with saxophone"
    print(f"\nRecommendations for query: '{query}'")
    query_recommendations = recommender.recommend_by_query(query)

    for _, product in query_recommendations.iterrows():
        display_product(product)

    # Test personalized recommendation
    user_history = [42, 100, 200]  # Example user history
    print(f"\nPersonalized recommendations based on user history: {user_history}")
    try:
        personalized_recommendations = recommender.recommend_personalized(user_history)

        for _, product in personalized_recommendations.iterrows():
            display_product(product)
    except Exception as e:
        print(f"Error getting personalized recommendations: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test recommendation methods")
    parser.add_argument(
        "--method",
        type=str,
        choices=["semantic", "image", "hybrid", "two-tower", "all"],
        default="all",
        help="Recommendation method to test",
    )

    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    if args.method == "semantic" or args.method == "all":
        test_semantic_recommender()

    if args.method == "image" or args.method == "all":
        test_image_recommender()

    if args.method == "hybrid" or args.method == "all":
        test_hybrid_recommender()

    if args.method == "two-tower" or args.method == "all":
        test_two_tower_recommender()


if __name__ == "__main__":
    main()
