import pandas as pd
import requests
import gzip
import os
import io
from rank_bm25 import BM25Okapi
import pickle

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

# Function to create a combined text for each document in the DataFrame
def create_document_texts(df):
    """
    Create combined text from titles and descriptions of documents in a DataFrame.

    Parameters:
    df (pd.DataFrame): A DataFrame containing 'title' and 'description' columns.

    Returns:
    list: A list of combined title and description strings for each row in the DataFrame.
    """
    return [f"{row['title']} {row['description']}" for _, row in df.iterrows()]

# Function to build a BM25 index from combined document texts
def build_bm25_index(document_texts):
    """
    Builds a BM25 index for the given document texts.

    Parameters:
    document_texts (list): A list of combined title and description texts.

    Returns:
    BM25Okapi: A BM25 index object.
    """
    tokenized_docs = [doc.lower().split(" ") for doc in document_texts]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25

# Save the BM25 index to a file
def save_bm25_index(bm25, filename):
    with open(filename, 'wb') as f:
        pickle.dump(bm25, f)

# Load the BM25 index from a file
def load_bm25_index(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Function to search the BM25 index
def get_bm25_results(query, meta_url, bm_pickle_file, directory, sample_size, top_n=5):
    """
    Retrieve top N results from the BM25 index for a given query, returning full document details.

    Parameters:
    query (str): Search query.
    meta_url (str): URL of the metadata file.
    bm_pickle_file (str): Path to the BM25 index file.
    directory (str): Directory to save or load data.
    sample_size (int): Number of samples to load from the metadata file.
    top_n (int): Number of top results to retrieve.

    Returns:
    pd.DataFrame: DataFrame containing the top N rows matching the query.
    """
    # Save DataFrame to CSV
    meta_file_path = os.path.join(directory, "Digital_Music_Meta.csv")
    if not os.path.exists(meta_file_path):
        # Load metadata into DataFrame
        digital_music_meta_df = load_jsonl_to_dataframe(meta_url, sample_size=sample_size)
        save_dataframe_to_csv(digital_music_meta_df, meta_file_path)
    else:
        digital_music_meta_df = pd.read_csv(meta_file_path)

    # Create combined texts for BM25 indexing
    combined_texts = create_document_texts(digital_music_meta_df)

    # Check if the index already exists
    if os.path.exists(os.path.join(directory, bm_pickle_file)):
        bm25_index = load_bm25_index(os.path.join(directory, bm_pickle_file))
    else:
        bm25_index = build_bm25_index(combined_texts)
        save_bm25_index(bm25_index, os.path.join(directory, bm_pickle_file))

    # Perform search
    tokenized_query = query.lower().split(" ")
    scores = bm25_index.get_scores(tokenized_query)
    top_n_indices = scores.argsort()[-top_n:][::-1]
    
    # Return top N results from the DataFrame
    return digital_music_meta_df.iloc[top_n_indices]

# Function to check if BM25 index file exists
def get_bm_index_status(directory, bm_pickle_file):
    return os.path.exists(os.path.join(directory, bm_pickle_file))
