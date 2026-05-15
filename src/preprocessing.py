"""
preprocessing.py

This module contains functions for Phase 1 (Data Preprocessing),
Phase 2 (Feature Engineering), and Phase 3 (NLP Pipeline - partial/stemming)
of the Movie Recommendation System.
"""

import ast
import pandas as pd
from typing import List, Tuple
from nltk.stem.porter import PorterStemmer

# Initialize Porter Stemmer
stemmer = PorterStemmer()

def load_data(movies_path: str, credits_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the TMDB movies and credits CSV files.

    Args:
        movies_path (str): Path to the tmdb_5000_movies.csv file.
        credits_path (str): Path to the tmdb_5000_credits.csv file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Loaded movies and credits dataframes.
    """
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    return movies, credits

def merge_data(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    """Merges the movies and credits dataframes on the 'title' column.

    Args:
        movies (pd.DataFrame): Movies dataframe.
        credits (pd.DataFrame): Credits dataframe.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    return movies.merge(credits, on="title")

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Selects the useful columns for content-based recommendation.

    Args:
        df (pd.DataFrame): Merged dataframe.

    Returns:
        pd.DataFrame: Dataframe with selected columns.
    """
    useful_cols = ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
    return df[useful_cols].copy()

def clean_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes null values and duplicate entries from the dataframe.

    Args:
        df (pd.DataFrame): Dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.dropna().copy()
    df = df.drop_duplicates(subset=["movie_id"]).copy()
    return df

def safe_literal_eval(val: str) -> List:
    """Safely evaluates a JSON-like string column into a Python list.

    Args:
        val (str): The string representing a list of dicts.

    Returns:
        List: Evaluated list or empty list if error.
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def extract_genres(genres_str: str) -> List[str]:
    """Extracts genre names from the raw JSON-like string.

    Args:
        genres_str (str): Raw string.

    Returns:
        List[str]: List of genre names.
    """
    genres_list = safe_literal_eval(genres_str)
    return [item["name"] for item in genres_list if "name" in item]

def extract_keywords(keywords_str: str) -> List[str]:
    """Extracts keyword names from the raw JSON-like string.

    Args:
        keywords_str (str): Raw string.

    Returns:
        List[str]: List of keyword names.
    """
    keywords_list = safe_literal_eval(keywords_str)
    return [item["name"] for item in keywords_list if "name" in item]

def extract_cast(cast_str: str) -> List[str]:
    """Extracts the top 3 cast members from the raw JSON-like string.

    Args:
        cast_str (str): Raw string.

    Returns:
        List[str]: List of top 3 cast names.
    """
    cast_list = safe_literal_eval(cast_str)
    return [item["name"] for item in cast_list[:3] if "name" in item]

def extract_director(crew_str: str) -> List[str]:
    """Extracts the director's name from the crew list.

    Args:
        crew_str (str): Raw crew string.

    Returns:
        List[str]: List containing the director's name or empty list.
    """
    crew_list = safe_literal_eval(crew_str)
    for member in crew_list:
        if member.get("job") == "Director":
            name = member.get("name")
            return [name] if name else []
    return []

def process_overview(overview_str: str) -> List[str]:
    """Splits overview paragraph into list of individual words.

    Args:
        overview_str (str): Movie overview text.

    Returns:
        List[str]: List of words.
    """
    if isinstance(overview_str, str):
        return overview_str.split()
    return []

def remove_spaces(tags_list: List[str]) -> List[str]:
    """Removes spaces from multi-word entities to create single tokens.

    Example: "Sam Worthington" -> "SamWorthington"

    Args:
        tags_list (List[str]): List of entities.

    Returns:
        List[str]: List with spaces removed from each entity.
    """
    return [item.replace(" ", "") for item in tags_list]

def apply_stemming(text: str) -> str:
    """Applies Porter Stemmer to a space-separated string of words.

    Args:
        text (str): Input space-separated string.

    Returns:
        str: Stemmed space-separated string.
    """
    return " ".join([stemmer.stem(word) for word in text.split()])

def preprocess_pipeline(movies_path: str, credits_path: str) -> pd.DataFrame:
    """Complete preprocessing pipeline from raw datasets to final tags-based dataframe.

    Args:
        movies_path (str): Path to movies.csv.
        credits_path (str): Path to credits.csv.

    Returns:
        pd.DataFrame: Preprocessed dataframe ready for vectorization.
                      Contains columns: 'movie_id', 'title', 'tags'.
    """
    # 1. Load data
    movies, credits = load_data(movies_path, credits_path)
    
    # 2. Merge datasets
    df = merge_data(movies, credits)
    
    # 3. Select columns
    df = select_columns(df)
    
    # 4 & 5. Clean missing and duplicates
    df = clean_missing_and_duplicates(df)
    
    # 6. Extract features (Phase 2)
    df["genres"] = df["genres"].apply(extract_genres)
    df["keywords"] = df["keywords"].apply(extract_keywords)
    df["cast"] = df["cast"].apply(extract_cast)
    df["director"] = df["crew"].apply(extract_director)
    df["overview"] = df["overview"].apply(process_overview)
    
    # 8. Remove spaces from multi-word entities
    df["genres"] = df["genres"].apply(remove_spaces)
    df["keywords"] = df["keywords"].apply(remove_spaces)
    df["cast"] = df["cast"].apply(remove_spaces)
    df["director"] = df["director"].apply(remove_spaces)
    
    # 6. Create a single "tags" column
    # Combine lists
    df["tags"] = df["overview"] + df["genres"] + df["keywords"] + df["cast"] + df["director"]
    
    # Convert list of tags to a single string
    df["tags"] = df["tags"].apply(lambda x: " ".join(x))
    
    # 7. Convert tags to lowercase
    df["tags"] = df["tags"].apply(lambda x: x.lower())
    
    # Phase 3: Stemming
    df["tags"] = df["tags"].apply(apply_stemming)
    
    # Keep final columns
    final_df = df[["movie_id", "title", "tags"]].copy()
    return final_df
