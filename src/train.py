"""
train.py

This module contains Phase 4 (NLP Vectorization) and Phase 5 (Model Persistence)
of the Movie Recommendation System. It runs the preprocessing pipeline, applies TF-IDF,
calculates cosine similarity, and pickles the required objects.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_pipeline

def train_model(
    movies_path: str,
    credits_path: str,
    output_dir: str
) -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    """Runs preprocessing, vectorization, and similarity calculation.
    Saves outputs as pickle files in output_dir.

    Args:
        movies_path (str): Path to movies.csv.
        credits_path (str): Path to credits.csv.
        output_dir (str): Directory where pickle files will be saved.

    Returns:
        Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]: Preprocessed dataframe,
                                                         fitted vectorizer,
                                                         cosine similarity matrix.
    """
    print("Step 1: Running Preprocessing Pipeline...")
    movies_df = preprocess_pipeline(movies_path, credits_path)
    print(f"Data preprocessed successfully. Shape: {movies_df.shape}")

    print("Step 2: Fitting TF-IDF Vectorizer...")
    # TfidfVectorizer setup as required
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies_df["tags"])
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")

    print("Step 3: Calculating Cosine Similarity Matrix...")
    similarity = cosine_similarity(tfidf_matrix)
    # Convert similarity matrix to float16 to save storage space if needed,
    # but standard float64 is fine and fits within limits. Let's keep it as is.
    print(f"Cosine Similarity Matrix shape: {similarity.shape}")

    print("Step 4: Persisting Models (Pickling)...")
    os.makedirs(output_dir, exist_ok=True)
    
    movies_pkl_path = os.path.join(output_dir, "movies.pkl")
    similarity_pkl_path = os.path.join(output_dir, "similarity.pkl")
    vectorizer_pkl_path = os.path.join(output_dir, "vectorizer.pkl")

    # Save movies dataframe
    with open(movies_pkl_path, "wb") as f:
        pickle.dump(movies_df, f)
    print(f"Saved: {movies_pkl_path}")

    # Save similarity matrix
    with open(similarity_pkl_path, "wb") as f:
        pickle.dump(similarity, f)
    print(f"Saved: {similarity_pkl_path}")

    # Save vectorizer
    with open(vectorizer_pkl_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved: {vectorizer_pkl_path}")

    return movies_df, vectorizer, similarity

def main() -> None:
    # Set paths relative to script location to ensure it works anywhere
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    movies_csv = os.path.join(project_root, "data", "tmdb_5000_movies.csv")
    credits_csv = os.path.join(project_root, "data", "tmdb_5000_credits.csv")
    
    train_model(movies_csv, credits_csv, project_root)
    print("Training process completed successfully.")

if __name__ == "__main__":
    main()
