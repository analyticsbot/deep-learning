"""
Utility functions for the recommendation module.
"""

import ast
import gzip
import io
import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests


def load_jsonl_to_dataframe(url: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Download and load a gzipped JSONL file into a DataFrame with optional sampling.

    Args:
        url: URL of the gzipped JSONL file
        sample_size: Number of samples to take (if None, all data is used)

    Returns:
        DataFrame containing the loaded data
    """
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            df = pd.read_json(f, lines=True)
            if sample_size is not None and sample_size < df.shape[0]:
                return df.sample(n=sample_size, random_state=1)
            else:
                return df
    else:
        raise Exception(f"Failed to download data: {response.status_code}")


def save_dataframe_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to a CSV file.

    Args:
        df: DataFrame to save
        file_path: Path to save the CSV file
    """
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


def load_or_download_data(meta_url: str, directory: str, sample_size: int) -> pd.DataFrame:
    """
    Load data from a local CSV file if it exists, otherwise download and save it.

    Args:
        meta_url: URL of the metadata file
        directory: Directory to save or load data
        sample_size: Number of samples to load

    Returns:
        DataFrame containing the loaded data
    """
    meta_file_path = os.path.join(directory, "Digital_Music_Meta.csv")

    if not os.path.exists(meta_file_path):
        df = load_jsonl_to_dataframe(meta_url, sample_size=sample_size)
        save_dataframe_to_csv(df, meta_file_path)
    else:
        df = pd.read_csv(meta_file_path)

    # Fill NaN values
    df.fillna("", inplace=True)

    return df


def parse_image_data(image_data: str) -> List[Dict[str, str]]:
    """
    Parse image data from string to list of dictionaries.

    Args:
        image_data: String representation of image data

    Returns:
        List of dictionaries containing image URLs
    """
    if not image_data or pd.isna(image_data):
        return []

    try:
        return ast.literal_eval(image_data)
    except (ValueError, SyntaxError):
        try:
            return json.loads(image_data)
        except json.JSONDecodeError:
            return []


def get_main_image_url(images: List[Dict[str, str]]) -> Optional[str]:
    """
    Get the main image URL from a list of image dictionaries.

    Args:
        images: List of image dictionaries

    Returns:
        URL of the main image, or None if no image is available
    """
    if not images:
        return None

    # Try to get the large image first, then fall back to thumb
    return images[0].get("large", images[0].get("thumb", None))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        vector: Input vector

    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector
