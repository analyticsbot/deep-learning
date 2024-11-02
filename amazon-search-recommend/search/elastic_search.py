import pandas as pd
import requests
import gzip
import os
import io
from elasticsearch import Elasticsearch, helpers
import ast
from elasticsearch.helpers import BulkIndexError

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

# Function to initialize Elasticsearch and index documents
def create_elasticsearch_index(df, index_name, es):
    # Create the index if it does not exist
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)
    
    # Prepare documents for indexing
    actions = [
    {
        "_index": index_name,
        "_id": str(i),
        "_source": {
            "title": row['title'],
            "description": row['description'],
            "images": row['images'],
            "average_rating": row['average_rating'],
            "rating_number": row['rating_number'],
            "price": f"${float(row['price']):.2f}" if pd.notna(row['price']) and isinstance(row['price'], (int, float, str)) and str(row['price']).replace('.', '', 1).isdigit() else "Price not available",
            "details": {key: value for key, value in ast.literal_eval(row['details']).items() if key.strip()} or {"Not Available": "Not Available"},
            "main_category": row['main_category'] if pd.notna(row['main_category']) and row['main_category'].strip() else "Not Available"


        }
    }
    for i, row in df.iterrows()
]
    
    try:
        helpers.bulk(es, actions)
    except BulkIndexError as e:
        for error in e.errors:
            print(error)

# Function to search Elasticsearch index
def search_elasticsearch(query, index_name, es, top_n=5):
    body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "title": query  # Match against the title field
                        }
                    },
                    {
                        "match": {
                            "description": query  # Match against the description field
                        }
                    }
                ]
            }
        }
    }
    
    response = es.search(index=index_name, body=body, size=top_n)
    return response['hits']['hits']  # List of matched documents


# Function to get results using Elasticsearch
def get_elasticsearch_results(query, meta_url, index_name, directory, sample_size, top_n=5):
    # Save DataFrame to CSV
    meta_file_path = os.path.join(directory, "Digital_Music_Meta.csv")
    if not os.path.exists(meta_file_path):
        digital_music_meta_df = load_jsonl_to_dataframe(meta_url, sample_size=sample_size)
        save_dataframe_to_csv(digital_music_meta_df, meta_file_path)
    else:
        digital_music_meta_df = pd.read_csv(meta_file_path)

    digital_music_meta_df.fillna('', inplace=True)

    # Initialize Elasticsearch client
    es = Elasticsearch(hosts=["http://localhost:9200"], timeout=600, http_auth=('elastic', 'elastic'))

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)
        es.indices.put_settings(index="es_index", body={
            "index": {
                "number_of_replicas": 0
            }
        })

        es.cluster.put_settings(body={
        "transient": {
            "cluster.routing.allocation.enable": "all",
            "cluster.routing.allocation.allow_rebalance": "always",
            "cluster.routing.allocation.disk.threshold_enabled": False
        }
        })


    # Create Elasticsearch index and index documents
    create_elasticsearch_index(digital_music_meta_df, index_name, es)

    # Perform search
    search_results = search_elasticsearch(query, index_name, es, top_n)
    
    # Convert search results to DataFrame
    results_df = pd.DataFrame([hit['_source'] for hit in search_results])
    return results_df

# Function to check if the Elasticsearch index exists
def get_elasticsearch_index_status(index_name):
    # Initialize Elasticsearch client
    es = Elasticsearch(hosts=["http://localhost:9200"], timeout=600, http_auth=('elastic', 'elastic'))
    return es.indices.exists(index=index_name)
