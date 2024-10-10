### # Recommendation System Project

This project implements several recommendation algorithms using PySpark and PyTorch, including Collaborative Filtering, Content-Based Filtering, Matrix Factorization, and Association Rule Mining.

We use the **MovieLens** dataset to demonstrate the recommendation models, and the project is containerized using Docker for reproducibility.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Recommendation Models Implemented](#recommendation-models-implemented)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Docker Build & Run Instructions](#docker-build--run-instructions)
- [Possible Extensions in the JupyterLab Notebook](#possible-extensions-in-the-jupyterlab-notebook)
- [Possible Additions and Improvements](#possible-additions-and-improvements)
- [FPGrowth: A Brief Overview](#fpgrowth-a-brief-overview)

---

## Project Overview

This project demonstrates various recommendation algorithms for building a movie recommendation system using the **MovieLens dataset**. The dataset is loaded and processed using **PySpark** for scalability, and different recommendation models such as Collaborative Filtering, Matrix Factorization, and Content-Based Filtering are implemented using **PyTorch**.

---

## Setup Instructions

### Prerequisites
- Docker installed on your machine.
- Docker Compose.

### Steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/analyticsbot/deep-learning.git
    cd RecSystems
    ```

2. Build the Docker image for non-spark based models:
    ```bash
    docker build -t RecSystems -f Dockerfile.RecSystems
    ```

2. Build the Docker image for spark based models:
    ```bash
    docker build -t RecSystems -f Dockerfile.spark
    ```

This will create a Docker container with the following tools:
- **Spark** for distributed processing.
- **JupyterLab** for code development and execution.

### Configuration:
The dataset (`ml-latest-small.zip`) is automatically downloaded from MovieLens and used to generate recommendations.
https://grouplens.org/datasets/movielens/latest/

---

## Running the Project

Once the Docker container is up and running, navigate to JupyterLab:

1. Start JupyterLab:
    ```bash
    docker run -p 8888:8888 RecSystems
    ```

2. Access the JupyterLab interface by opening your browser and navigating to `http://localhost:8888`.

3. Open the notebook files to explore and run each recommendation model. Each model is implemented in a separate Jupyter notebook.

---

## Recommendation Models Implemented

1. **Collaborative Filtering (User-based and Item-based)**
   - **User-Based Filtering**: Recommends items by finding similar users.
   - **Item-Based Filtering**: Recommends items based on similarities to previously liked items.

2. **Content-Based Filtering**
   - Recommends items based on item features and user preferences (e.g., genre, keywords).

3. **Matrix Factorization**
   - Techniques such as Singular Value Decomposition (SVD) and Non-negative Matrix Factorization (NMF) decompose the user-item interaction matrix to identify latent factors.

4. **Association Rule Mining**
   - The FPGrowth algorithm is used to mine frequent patterns and recommend items based on the relationships between them.

5. **Two Tower Model**
   - The Two-Tower Model is a deep learning architecture commonly used in recommendation systems. It consists of two separate neural networks (or "towers")—one for user data and one for item data—that learn embeddings independently. The outputs (embeddings) from each tower are then combined, often using dot product or cosine similarity, to predict user-item interactions, like clicks or purchases. This model helps efficiently handle large-scale recommendations by focusing on capturing relevant features from both users and items.

6. **Neural Collaborative Filtering**
   - Neural Collaborative Filtering (NCF) is a deep learning approach used in recommendation systems, combining neural networks with collaborative filtering techniques. It replaces traditional matrix factorization by using neural networks to model complex, non-linear relationships between users and items, capturing more intricate patterns. NCF generates personalized recommendations by learning user-item interactions through embedding layers and fully connected neural networks, offering improved performance in large-scale, personalized content recommendation tasks.

---

## Docker Build & Run Instructions

### Build the Docker Image
```bash
docker build -t recommendation-system .
```
### Run the Docker Container
```bash
docker run -p 8888:8888 recommendation-system
```
### Access JupyterLab
Open your browser and go to http://localhost:8888. You will have access to JupyterLab with the code loaded inside it.


## Possible Extensions in the JupyterLab Notebook
- Interactive Visualizations: Use matplotlib or seaborn for visualizing the results of collaborative filtering, matrix factorization, or association rules. Plot the distribution of item ratings, similarities between users/items, etc.
- Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV to tune models like SVD, KNN, or deep learning architectures.
- Real-time Data Updates: Set up streaming data from the MovieLens API to simulate a dynamic recommendation system.
- Model Evaluation: Add metrics like Precision, Recall, F1-Score, etc., to evaluate the performance of different recommendation techniques.

## Possible Additions and Improvements
- Evaluation and Tuning: Add cross-validation for model performance evaluation. Implement grid search for hyperparameter tuning. Use different similarity metrics like Jaccard or Pearson for collaborative filtering.
- Improved Deep Learning: Experiment with more complex architectures (e.g., multi-layer perceptrons, deep autoencoders). Incorporate user and item embeddings using neural network-based models.
- Real-Time Processing: Use Kafka or similar tools for real-time data processing for recommendation updates.
- Personalization: Incorporate user demographics (age, gender, etc.) into models like neural collaborative filtering.
- Production-Level Deployment: Containerize models using Docker and deploy via Kubernetes. Set up APIs to deliver recommendations in real-time. Add monitoring (e.g., using Prometheus) to track model performance over time.


## FPGrowth: A Brief Overview
FPGrowth (Frequent Pattern Growth) is an efficient algorithm used for mining frequent itemsets and association rules in large datasets, particularly in market basket analysis. It is an improvement over the classic Apriori algorithm, as it does not require generating candidate itemsets explicitly, making it faster and more scalable.

Key Concepts:
Frequent Itemsets: A frequent itemset is a group of items that appear together frequently in transactions (e.g., in a retail store, items bought together frequently). The goal of FPGrowth is to identify these frequent itemsets that meet a user-defined minimum support threshold.

Association Rules: Association rules describe relationships between items. They have the form {A} → {B} where A and B are itemsets, suggesting that if itemset A is bought, itemset B is likely to be bought. These rules are evaluated based on confidence and lift.

FPGrowth Algorithm Steps:
Construct a Compact Data Structure (the FP-tree): Instead of generating all possible itemsets like in Apriori, FPGrowth constructs a prefix tree called an FP-tree. The FP-tree is built by scanning the dataset once and capturing the frequent items in a compact structure, which makes it more memory-efficient and faster.

Mine Frequent Itemsets: The algorithm recursively mines the FP-tree, starting from the most frequent items, and extracts frequent itemsets using a divide-and-conquer strategy. It works by recursively splitting the tree based on conditional patterns until no more frequent itemsets can be found.

Generate Association Rules: Once frequent itemsets are found, association rules are generated based on user-specified confidence and lift thresholds:

Confidence: The likelihood that item B is purchased given that item A is purchased.
Lift: Measures how much more likely B is purchased when A is purchased, compared to when B is purchased without A.
Advantages of FPGrowth:
Efficient: FPGrowth avoids generating candidate itemsets explicitly, making it faster than Apriori.
Scalability: Can handle large datasets due to its compact FP-tree representation.
Parallelism: Can be parallelized for distributed processing, which makes it suitable for big data applications (e.g., Spark).
Example Use Case:
In retail, FPGrowth can help analyze transactions to discover rules like:

"People who bought milk also bought bread."
"If a customer buys a smartphone, they are likely to buy a phone case."
PySpark FPGrowth:
In PySpark, FPGrowth is implemented as part of MLlib, Spark's machine learning library. It is easy to use and allows you to work with large datasets in a distributed environment.

Key Parameters:
minSupport: Minimum support threshold, the proportion of transactions in which an itemset should appear.
minConfidence: Minimum confidence threshold for generating association rules.