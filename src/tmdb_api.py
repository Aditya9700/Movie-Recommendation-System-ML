"""
tmdb_api.py

This module implements Feature 1 (TMDB API Integration), Feature 5 (Fallback Poster),
and Feature 6 (Performance Caching) of the Movie Recommendation System.
It queries the TMDB API to fetch movie posters and details, falling back
gracefully to the local dataset if the API is offline or the key is invalid/missing.
"""

import os
import ast
import requests
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load local environment variables from .env file
load_dotenv()

# Local fallback cache for raw movie CSV
_local_movies_df: Optional[pd.DataFrame] = None

def get_api_key() -> Optional[str]:
    """Retrieves the TMDB API Key from Streamlit session state, secrets, or environment variables.

    Returns:
        Optional[str]: The API key if found, else None.
    """
    # 1. Check Streamlit session state override (highest priority for user testing)
    if "tmdb_api_key_override" in st.session_state and st.session_state["tmdb_api_key_override"].strip():
        return st.session_state["tmdb_api_key_override"].strip()

    # 2. Check Streamlit secrets
    try:
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass

    # 3. Check environment variables
    return os.getenv("TMDB_API_KEY")

def get_placeholder_poster() -> str:
    """Returns a high-quality fallback placeholder image URL.

    Returns:
        str: Fallback placeholder image URL.
    """
    # A generic clean poster placeholder
    return "https://images.unsplash.com/photo-1440404653325-ab127d49abc1?q=80&w=500&auto=format&fit=crop"

def fetch_local_metadata(movie_id: int) -> Dict[str, Any]:
    """Retrieves movie details from the local CSV file as a fallback.

    Args:
        movie_id (int): TMDB Movie ID.

    Returns:
        Dict[str, Any]: Metadata dict mirroring TMDB format.
    """
    global _local_movies_df
    
    if _local_movies_df is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            csv_path = os.path.join(project_root, "data", "tmdb_5000_movies.csv")
            if os.path.exists(csv_path):
                _local_movies_df = pd.read_csv(csv_path)
            else:
                return {}
        except Exception:
            return {}

    if _local_movies_df is None:
        return {}

    match = _local_movies_df[_local_movies_df["id"] == movie_id]
    if len(match) == 0:
        return {}

    row = match.iloc[0]
    
    # Parse local JSON genres
    genres = []
    if isinstance(row.get("genres"), str):
        try:
            genres_data = ast.literal_eval(row["genres"])
            genres = [g["name"] for g in genres_data if isinstance(g, dict) and "name" in g]
        except Exception:
            genres = []

    return {
        "movie_id": movie_id,
        "title": row.get("title", "Unknown Movie"),
        "poster_path": get_placeholder_poster(),
        "backdrop_path": "",
        "overview": row.get("overview", "No summary available in local database."),
        "release_date": row.get("release_date", "N/A"),
        "vote_average": float(row.get("vote_average", 0.0)),
        "runtime": int(row.get("runtime", 0)) if not pd.isna(row.get("runtime")) else 0,
        "genres": genres,
        "source": "Local Fallback"
    }

# Cache the response to avoid duplicate API requests
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_movie_details(movie_id: int) -> Dict[str, Any]:
    """Fetches movie details (poster, backdrop, overview, release_date, etc.) from the TMDB API.
    If requests fail, key is missing, or rate-limited, falls back to local dataset metadata.

    Args:
        movie_id (int): TMDB Movie ID.

    Returns:
        Dict[str, Any]: Movie details dictionary.
    """
    api_key = get_api_key()
    
    if not api_key:
        # Fall back immediately if no API key is provided
        return fetch_local_metadata(movie_id)

    # TMDB API URL
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": api_key, "language": "en-US"}

    try:
        # Call TMDB API with a timeout
        response = requests.get(url, params=params, timeout=5)
        
        # Handle invalid key
        if response.status_code == 401:
            st.sidebar.warning("⚠️ Invalid TMDB API Key. Using local fallback details.")
            return fetch_local_metadata(movie_id)
            
        # Handle rate limiting
        if response.status_code == 429:
            st.sidebar.warning("⚠️ TMDB API Rate Limited. Using local fallback details.")
            return fetch_local_metadata(movie_id)
            
        # Raise exception for other error codes to trigger local fallback
        response.raise_for_status()
        
        data = response.json()
        
        # Format poster path
        poster_path = data.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else get_placeholder_poster()
        
        # Format backdrop path
        backdrop_path = data.get("backdrop_path")
        backdrop_url = f"https://image.tmdb.org/t/p/w1280{backdrop_path}" if backdrop_path else ""
        
        # Format genres list
        genres_list = data.get("genres", [])
        genres = [item["name"] for item in genres_list if "name" in item]
        
        return {
            "movie_id": movie_id,
            "title": data.get("title", "Unknown Movie"),
            "poster_path": poster_url,
            "backdrop_path": backdrop_url,
            "overview": data.get("overview", "No description available."),
            "release_date": data.get("release_date", "N/A"),
            "vote_average": float(data.get("vote_average", 0.0)),
            "runtime": int(data.get("runtime", 0)) if data.get("runtime") else 0,
            "genres": genres,
            "source": "TMDB API"
        }
        
    except (requests.exceptions.RequestException, ValueError) as e:
        # Handles connection errors, timeouts, or JSON decoding failures
        return fetch_local_metadata(movie_id)
