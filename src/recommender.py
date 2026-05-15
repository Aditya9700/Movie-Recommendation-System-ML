"""
recommender.py

This module contains Phase 4 (Recommendation Engine) logic.
It loads the serialized model pickle files and provides the recommend() function
with proper error handling, case insensitivity, and close-match suggestions.
"""

import os
import pickle
import difflib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# Module-level cache for models
_movies_df: Optional[pd.DataFrame] = None
_similarity: Optional[np.ndarray] = None
_vectorizer: Optional[Any] = None

class MovieNotFoundError(Exception):
    """Exception raised when a requested movie title is not found in the database."""
    pass

class EmptyInputError(Exception):
    """Exception raised when the input movie name is empty or whitespace-only."""
    pass

def load_recommendation_models(project_root: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray, Any]:
    """Loads the movies dataframe, similarity matrix, and vectorizer pickle files.

    Args:
        project_root (Optional[str]): Path to project root containing pickle files.
                                      If None, uses current working directory.

    Returns:
        Tuple[pd.DataFrame, np.ndarray, Any]: Loaded models.
    """
    global _movies_df, _similarity, _vectorizer

    if _movies_df is not None and _similarity is not None and _vectorizer is not None:
        return _movies_df, _similarity, _vectorizer

    if project_root is None:
        # Resolve path based on standard layout
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)

    movies_pkl = os.path.join(project_root, "movies.pkl")
    similarity_pkl = os.path.join(project_root, "similarity.pkl")
    vectorizer_pkl = os.path.join(project_root, "vectorizer.pkl")

    if not (os.path.exists(movies_pkl) and os.path.exists(similarity_pkl) and os.path.exists(vectorizer_pkl)):
        raise FileNotFoundError(
            f"Required pickle files not found in {project_root}. Please run train.py first."
        )

    with open(movies_pkl, "rb") as f:
        _movies_df = pickle.load(f)

    with open(similarity_pkl, "rb") as f:
        _similarity = pickle.load(f)

    with open(vectorizer_pkl, "rb") as f:
        _vectorizer = pickle.load(f)

    return _movies_df, _similarity, _vectorizer

def recommend(movie_name: str, project_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Recommends top 5 similar movies based on cosine similarity scores.

    Args:
        movie_name (str): Title of the movie.
        project_root (Optional[str]): Root folder of the project containing model pickles.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing title, movie_id, and similarity score
                                 for the top 5 most similar movies.

    Raises:
        EmptyInputError: If input is empty.
        MovieNotFoundError: If movie title is not found (includes suggestion list).
    """
    # 1. Handle empty input
    if not movie_name or not movie_name.strip():
        raise EmptyInputError("Input movie name cannot be empty.")

    # 2. Ensure models are loaded
    movies_df, similarity_matrix, _ = load_recommendation_models(project_root)
    
    cleaned_input = movie_name.strip().lower()
    
    # 3. Find match (case-insensitive and whitespace-insensitive)
    # Check exact match
    match_indices = movies_df[movies_df["title"].str.strip().str.lower() == cleaned_input].index
    
    if len(match_indices) == 0:
        # Try to find close matches to suggest to user
        all_titles = movies_df["title"].tolist()
        close_matches = difflib.get_close_matches(movie_name, all_titles, n=3, cutoff=0.4)
        
        if close_matches:
            suggestions_str = ", ".join([f"'{m}'" for m in close_matches])
            raise MovieNotFoundError(
                f"Movie '{movie_name}' not found. Did you mean: {suggestions_str}?"
            )
        else:
            raise MovieNotFoundError(
                f"Movie '{movie_name}' not found in database, and no close matches were found."
            )
            
    # Take the first matched index
    movie_index = match_indices[0]
    
    # 4. Compute top similar movies
    # Get similarity vector for this movie
    similarity_vector = similarity_matrix[movie_index]
    
    # Get indices sorted by similarity score (descending)
    # enumerate returns index and score, sort by score, excluding the input movie itself
    similar_movies_scores = list(enumerate(similarity_vector))
    
    # Sort by score (item[1]) descending, and filter out the movie itself (index == movie_index)
    sorted_movies = sorted(similar_movies_scores, key=lambda x: x[1], reverse=True)
    
    # Filter the input movie itself from the list to avoid recommending itself
    # If the database has duplicate movies with different IDs, we also make sure to use distinct titles
    recommendations: List[Dict[str, Any]] = []
    seen_titles = {movies_df.iloc[movie_index]["title"].lower()}
    
    for idx, score in sorted_movies:
        title = movies_df.iloc[idx]["title"]
        title_lower = title.lower()
        if title_lower not in seen_titles:
            recommendations.append({
                "title": title,
                "movie_id": int(movies_df.iloc[idx]["movie_id"]),
                "similarity": float(score)
            })
            seen_titles.add(title_lower)
        if len(recommendations) == 5:
            break
            
    return recommendations
