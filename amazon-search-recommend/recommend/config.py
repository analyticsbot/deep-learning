"""
Configuration settings for the recommendation module.
"""

# URLs for the Digital Music category
meta_url = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/"
    "raw/meta_categories/meta_Digital_Music.jsonl.gz"
)

# Specify sample size and directory for saving
sample_size = 100000
directory = "/Users/neo/Downloads/Personal/Projects/deep-learning/amazon-search-recommend/data"

# Qdrant collections for different recommendation methods
collection_name_semantic = "semantic_recommend_collection"
collection_name_image = "image_recommend_collection"
collection_name_hybrid = "hybrid_recommend_collection"
collection_name_two_tower = "two_tower_recommend_collection"

# Model settings
clip_model_name = "openai/clip-vit-base-patch16"
two_tower_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Vector dimensions
clip_vector_dim = 512
two_tower_vector_dim = 384

# Number of recommendations to return
num_recommendations = 5
