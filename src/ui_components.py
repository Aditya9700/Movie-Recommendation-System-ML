"""
ui_components.py

This module contains modular UI components for the Movie Recommendation System,
including Selected Movie Details (Feature 3), upgraded Recommendation Cards (Feature 2 & 4),
and Sidebar Stats (Feature 8).
"""

import streamlit as st
from typing import Dict, Any

def render_movie_details(details: Dict[str, Any]) -> None:
    """Renders a visually striking information panel for the selected movie.

    Args:
        details (Dict[str, Any]): Movie metadata details.
    """
    if not details:
        st.warning("⚠️ No metadata available for the selected movie.")
        return

    # Extract info safely
    title = details.get("title", "Unknown Title")
    release_date = details.get("release_date", "N/A")
    release_year = release_date[:4] if isinstance(release_date, str) and len(release_date) >= 4 else "N/A"
    rating = details.get("vote_average", 0.0)
    runtime = details.get("runtime", 0)
    genres = details.get("genres", [])
    overview = details.get("overview", "No summary available.")
    poster_path = details.get("poster_path", "")
    backdrop_path = details.get("backdrop_path", "")

    # Main details container with custom styles
    st.markdown(
        """
        <div style="
            background: rgba(255, 255, 255, 0.04);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2.5rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        ">
        """,
        unsafe_allow_html=True
    )

    # 2-column layout (left: poster, right: details)
    col_poster, col_info = st.columns([1, 3.5])

    with col_poster:
        if poster_path:
            st.image(poster_path, width=220)
        else:
            st.warning("No Poster Available")

    with col_info:
        st.markdown(f"<h2 style='margin-top: 0; color: #ffffff;'>{title}</h2>", unsafe_allow_html=True)
        
        # Meta info row
        meta_html = f"""
        <div style='display: flex; gap: 1.5rem; margin-bottom: 1.2rem; align-items: center; color: #cbd5e0; font-size: 0.95rem;'>
            <div>📅 <b>{release_year}</b></div>
            <div>⏱️ <b>{runtime} mins</b></div>
            <div style='
                background: rgba(255, 75, 75, 0.15); 
                color: #ff4b4b; 
                padding: 0.1rem 0.6rem; 
                border-radius: 8px; 
                font-weight: 700;
            '>⭐ {rating:.1f}/10</div>
        </div>
        """
        st.markdown(meta_html, unsafe_allow_html=True)
        
        # Genres pills
        if genres:
            pills = " ".join([
                f"<span style='background: rgba(255, 255, 255, 0.08); color: #ffffff; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; margin-right: 0.5rem; font-weight: 500;'>{g}</span>"
                for g in genres
            ])
            st.markdown(f"<div style='margin-bottom: 1.5rem;'>{pills}</div>", unsafe_allow_html=True)
            
        # Overview
        st.markdown("<h4 style='color: #a0aec0; margin-bottom: 0.5rem;'>Overview</h4>", unsafe_allow_html=True)
        st.write(overview)

    st.markdown("</div>", unsafe_allow_html=True)

def render_recommendation_card(title: str, poster_url: str, score: float) -> None:
    """Renders a single recommendation card with hover style guidelines, poster,
    movie title, and match percentage score.

    Args:
        title (str): Movie title.
        poster_url (str): Movie poster image URL.
        score (float): Cosine similarity score.
    """
    # Card wrapper
    st.markdown(
        """
        <div class="rec-card-container">
        """,
        unsafe_allow_html=True
    )
    
    # Image (Feature 2)
    st.image(poster_url, use_container_width=True)
    
    # Details under the image
    st.markdown(
        f"""
        <div class="rec-card-content">
            <div class="rec-card-title" title="{title}">{title}</div>
            <div class="rec-card-badge">Match: {score*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_sidebar_stats(total_movies: int, vocab_size: int) -> None:
    """Renders system statistics in the sidebar (Feature 8).

    Args:
        total_movies (int): Total number of movies.
        vocab_size (int): Max features/vocabulary size of TF-IDF.
    """
    st.sidebar.markdown("<h3 style='color: #000000;'>⚙️ System Statistics</h3>", unsafe_allow_html=True)
    
    stats_html = f"""
    <div style="
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(0, 0, 0, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.5rem;
        color: #000000;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
            <span style="color: #333333; font-weight: 500;">Total Movies:</span>
            <span style="font-weight: 700; color: #000000;">{total_movies}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
            <span style="color: #333333; font-weight: 500;">Vocabulary Size:</span>
            <span style="font-weight: 700; color: #000000;">{vocab_size}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
            <span style="color: #333333; font-weight: 500;">Methodology:</span>
            <span style="font-weight: 700; color: #e53e3e; font-size: 0.85rem;">TF-IDF + Cosine</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #333333; font-weight: 500;">Dataset:</span>
            <span style="font-weight: 700; color: #000000;">TMDB 5000</span>
        </div>
    </div>
    """
    st.sidebar.markdown(stats_html, unsafe_allow_html=True)
