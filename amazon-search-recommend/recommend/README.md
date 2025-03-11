# Amazon Product Recommendation System

This module implements various recommendation methods for Amazon products, focusing on the Digital Music category. The recommendation methods are designed to provide personalized product suggestions based on different similarity metrics and user preferences.

## Recommendation Methods

### 1. Semantic Similarity

The semantic similarity recommender uses CLIP text embeddings to find products with similar textual descriptions. This method is effective for finding products that are semantically related, even if they don't share the same keywords.

**Key Features:**
- Uses OpenAI's CLIP model for text embeddings
- Indexes product titles and descriptions
- Supports recommendations based on product ID or text query
- Leverages Qdrant vector database for efficient similarity search

### 2. Image Similarity

The image similarity recommender uses CLIP image embeddings to find products with visually similar images. This method is particularly useful for visual product discovery.

**Key Features:**
- Uses OpenAI's CLIP model for image embeddings
- Indexes product images
- Supports recommendations based on product ID or external image URL
- Handles missing or invalid images gracefully

### 3. Hybrid Recommender

The hybrid recommender combines semantic and image similarity to provide more comprehensive recommendations. By weighting the importance of text and image features, it can balance between content-based and visual similarity.

**Key Features:**
- Combines CLIP text and image embeddings
- Configurable weighting between semantic and image similarity
- Supports recommendations based on product ID or combined text/image query
- Provides a more holistic view of product similarity

### 4. Two-Tower Model

The two-tower model uses a separate encoder architecture for query and item representations, enabling efficient retrieval and personalized recommendations based on user history.

**Key Features:**
- Uses SentenceTransformer for efficient text encoding
- Supports recommendations based on product ID, text query, or user history
- Enables personalized recommendations by combining user history with optional queries
- Designed for scalability with large product catalogs

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- Qdrant (running locally or accessible via API)
- Sentence Transformers
- Pandas, NumPy, and other common data science libraries

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/amazon-search-recommend.git
cd amazon-search-recommend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Qdrant server (if using locally):
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Configuration

The module uses a configuration file (`config.py`) to manage settings such as:
- Data source URLs
- Sample size for indexing
- Vector dimensions
- Collection names for Qdrant
- Model names and parameters

You can modify these settings to suit your specific requirements.

## Usage

### Building Indices

Before using the recommendation methods, you need to build the corresponding indices:

```python
from recommend.semantic_similarity import SemanticRecommender
from recommend.image_similarity import ImageRecommender
from recommend.hybrid_recommender import HybridRecommender
from recommend.two_tower_model import TwoTowerRecommender

# Build semantic index
semantic_recommender = SemanticRecommender()
semantic_recommender.build_index()

# Build image index
image_recommender = ImageRecommender()
image_recommender.build_index()

# Build hybrid index
hybrid_recommender = HybridRecommender(semantic_weight=0.6)
hybrid_recommender.build_index()

# Build two-tower index
two_tower_recommender = TwoTowerRecommender()
two_tower_recommender.build_index()
```

### Getting Recommendations

Once the indices are built, you can get recommendations using various methods:

```python
# Get recommendations based on a product ID
semantic_recommendations = semantic_recommender.recommend_similar_products(42)
image_recommendations = image_recommender.recommend_similar_products(42)
hybrid_recommendations = hybrid_recommender.recommend_similar_products(42)
two_tower_recommendations = two_tower_recommender.recommend_similar_products(42)

# Get recommendations based on a text query
text_recommendations = semantic_recommender.recommend_by_text_query("classical piano music")
query_recommendations = two_tower_recommender.recommend_by_query("jazz saxophone")

# Get recommendations based on an image URL
image_url = "https://example.com/product_image.jpg"
image_url_recommendations = image_recommender.recommend_by_image_url(image_url)

# Get personalized recommendations based on user history
user_history = [42, 100, 200]  # Product IDs the user has interacted with
personalized_recommendations = two_tower_recommender.recommend_personalized(user_history, user_query="rock music")
```

### Demo Script

A demo script is provided to test all recommendation methods:

```bash
python -m recommend.demo --method all
```

You can also test specific methods:

```bash
python -m recommend.demo --method semantic
python -m recommend.demo --method image
python -m recommend.demo --method hybrid
python -m recommend.demo --method two-tower
```

### Streamlit App

A Streamlit app is included for interactive exploration of the recommendation methods:

```bash
streamlit run recommend/streamlit_app.py
```

The app provides a user-friendly interface to:
- Select different recommendation methods
- Find similar products based on product ID
- Search for products using text queries
- Build a user history for personalized recommendations
- Visualize product details and recommendations

## Implementation Details

### Vector Databases

The recommendation system uses Qdrant as the vector database for storing and searching embeddings. Qdrant provides:
- Efficient similarity search with HNSW algorithm
- Support for filtering during search
- Payload storage alongside vectors
- High performance for large-scale deployments

### Models

The system uses the following models:
- **CLIP** (Contrastive Language-Image Pretraining): For text and image embeddings in semantic, image, and hybrid recommenders
- **SentenceTransformer**: For efficient text encoding in the two-tower model

### Data Processing

The system handles various data processing tasks:
- Loading and sampling Amazon product data
- Parsing and normalizing product attributes
- Handling missing or invalid data
- Batched processing to manage memory usage

## Future Improvements

Potential enhancements for the recommendation system:
- Integration with user feedback for reinforcement learning
- A/B testing framework for comparing recommendation methods
- Support for additional product categories beyond Digital Music
- Hybrid search combining vector search with traditional filtering
- Real-time recommendation updates based on user behavior

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Amazon product dataset from UCSD
- OpenAI for the CLIP model
- Sentence Transformers for efficient text encodings
- Qdrant for the vector database implementation
