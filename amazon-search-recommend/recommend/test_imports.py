"""
Test script to verify that all modules can be imported without errors.
"""

import os
import sys


def test_imports():
    """Test importing all recommendation modules."""
    print("Testing imports...")

    # Test importing utils
    try:
        from recommend.utils import (
            get_main_image_url,
            load_jsonl_to_dataframe,
            load_or_download_data,
            normalize_vector,
            parse_image_data,
            save_dataframe_to_csv,
        )

        print("✅ Successfully imported utils")
    except ImportError as e:
        print(f"❌ Error importing utils: {e}")
        return False

    # Test importing config
    try:
        from recommend.config import (
            clip_model_name,
            clip_vector_dim,
            collection_name_hybrid,
            collection_name_image,
            collection_name_semantic,
            collection_name_two_tower,
            directory,
            meta_url,
            num_recommendations,
            sample_size,
            two_tower_model_name,
            two_tower_vector_dim,
        )

        print("✅ Successfully imported config")
    except ImportError as e:
        print(f"❌ Error importing config: {e}")
        return False

    # Test importing semantic_similarity
    try:
        from recommend.semantic_similarity import SemanticRecommender

        print("✅ Successfully imported semantic_similarity")
    except ImportError as e:
        print(f"❌ Error importing semantic_similarity: {e}")
        return False

    # Test importing image_similarity
    try:
        from recommend.image_similarity import ImageRecommender

        print("✅ Successfully imported image_similarity")
    except ImportError as e:
        print(f"❌ Error importing image_similarity: {e}")
        return False

    # Test importing hybrid_recommender
    try:
        from recommend.hybrid_recommender import HybridRecommender

        print("✅ Successfully imported hybrid_recommender")
    except ImportError as e:
        print(f"❌ Error importing hybrid_recommender: {e}")
        return False

    # Test importing two_tower_model
    try:
        from recommend.two_tower_model import TwoTowerRecommender

        print("✅ Successfully imported two_tower_model")
    except ImportError as e:
        print(f"❌ Error importing two_tower_model: {e}")
        return False

    print("All imports successful!")
    return True


def test_data_directory():
    """Test that the data directory exists or can be created."""
    from recommend.config import directory

    print(f"Testing data directory: {directory}")

    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created data directory: {directory}")
        except Exception as e:
            print(f"❌ Error creating data directory: {e}")
            return False
    else:
        print(f"✅ Data directory exists: {directory}")

    return True


def main():
    """Run all tests."""
    print("Running tests for recommendation modules...")

    # Test imports
    if not test_imports():
        print("Import tests failed.")
        return False

    # Test data directory
    if not test_data_directory():
        print("Data directory test failed.")
        return False

    print("\nAll tests passed! The recommendation modules are ready to use.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
