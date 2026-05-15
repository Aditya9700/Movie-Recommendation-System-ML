"""
app.py

Upgraded Streamlit application for the Movie Recommendation System.
Integrates TMDB API for movie posters and metadata details, displays a
selected movie overview panel, renders a responsive 5-column grid with hover card animations,
and shows system statistics in the sidebar.
"""

import os
import sys
import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Project structure setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MOVIES_PKL = os.path.join(PROJECT_ROOT, "movies.pkl")
SIMILARITY_PKL = os.path.join(PROJECT_ROOT, "similarity.pkl")
VECTORIZER_PKL = os.path.join(PROJECT_ROOT, "vectorizer.pkl")

# Helper function to check if models exist
def models_exist() -> bool:
    return os.path.exists(MOVIES_PKL) and os.path.exists(SIMILARITY_PKL) and os.path.exists(VECTORIZER_PKL)

# Custom CSS for premium glassmorphic dark theme and hover effects (Feature 4)
custom_css = """
<style>
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #0f0c1b, #1d182e, #0f0c1b);
        color: #ffffff;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Title styling */
    .app-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(45deg, #ff4b4b, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
        animation: fadeIn 1.5s ease-in-out;
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* Card container with hover scale, rounded corners, and shadows (Feature 4) */
    .rec-card-container {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 0.6rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 1.2rem;
    }
    
    .rec-card-container:hover {
        transform: translateY(-8px) scale(1.025);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 75, 75, 0.45);
        box-shadow: 0 12px 24px rgba(255, 75, 75, 0.22);
    }
    
    .rec-card-content {
        padding: 0.8rem 0.2rem 0.2rem 0.2rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
        flex-grow: 1;
    }
    
    .rec-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        line-height: 1.3;
        height: 2.6rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .rec-card-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ff4b4b, #8a2387);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(255, 75, 75, 0.25);
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff4b4b, #8a2387) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.6rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4) !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.6) !important;
        color: white !important;
    }
    
    /* Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
"""

# Render custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Add src to system path
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Sidebar Header
st.sidebar.markdown("<h2 style='text-align: center; color: #ff4b4b;'>System Dashboard</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")



# Self-healing training process inside Streamlit
if not models_exist():
    st.markdown("<h1 class='app-title'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='app-subtitle'>Your ultimate film companion, powered by Machine Learning</p>", unsafe_allow_html=True)
    
    st.warning("⚠️ Model files (`movies.pkl`, `similarity.pkl`, `vectorizer.pkl`) were not found.")
    st.info("The system needs to preprocess the TMDB 5000 dataset and compute the similarity matrix before you can search for movies.")
    
    if st.button("🚀 Train Recommendation Engine"):
        with st.spinner("Processing TMDB 5000 datasets and generating model pickles... This may take a moment."):
            try:
                from train import train_model
                
                movies_csv = os.path.join(PROJECT_ROOT, "data", "tmdb_5000_movies.csv")
                credits_csv = os.path.join(PROJECT_ROOT, "data", "tmdb_5000_credits.csv")
                
                train_model(movies_csv, credits_csv, PROJECT_ROOT)
                st.success("🎉 Recommendation models trained and saved successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error training model: {e}")
                
else:
    # Import modules when files are guaranteed to exist
    from tmdb_api import fetch_movie_details
    from ui_components import render_movie_details, render_recommendation_card, render_sidebar_stats
    from recommender import load_recommendation_models, recommend, MovieNotFoundError, EmptyInputError

    # Load recommendation models
    try:
        movies_df, _, _ = load_recommendation_models(PROJECT_ROOT)
        movie_list = sorted(movies_df["title"].unique())
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

    # App Header
    st.markdown("<h1 class='app-title'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='app-subtitle'>Discover similar films instantly using Content-Based Filtering</p>", unsafe_allow_html=True)

    # Search section with clean structure
    col_search, col_btn = st.columns([4, 1])

    with col_search:
        selected_movie = st.selectbox(
            "Search or select a movie from the catalog:",
            options=movie_list,
            placeholder="Type a movie title..."
        )

    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # spacer
        recommend_clicked = st.button("✨ Recommend")

    # Feature 3: Selected Movie Details Panel (renders above recommendations)
    if selected_movie:
        # Fetch ID of the selected movie
        movie_row = movies_df[movies_df["title"] == selected_movie].iloc[0]
        movie_id = int(movie_row["movie_id"])
        
        # Load and render selected movie details
        details = fetch_movie_details(movie_id)
        
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        render_movie_details(details)

    # Recommendation display
    if recommend_clicked:
        if selected_movie:
            with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
                try:
                    # Feature 7: Recommendation Output is now list of dictionaries
                    recommendations = recommend(selected_movie, PROJECT_ROOT)
                    
                    st.markdown(f"### 🍿 Recommendations for **{selected_movie}**:")
                    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
                    
                    # Display as columns (Feature 2: Responsive 5-column grid)
                    cols = st.columns(5)
                    for col, rec_dict in zip(cols, recommendations):
                        rec_title = rec_dict["title"]
                        rec_id = rec_dict["movie_id"]
                        rec_score = rec_dict["similarity"]
                        
                        # Fetch recommendations' poster paths and metadata (Feature 6: cached API calls)
                        rec_details = fetch_movie_details(rec_id)
                        rec_poster = rec_details.get("poster_path")
                        
                        with col:
                            # Render upgraded card components (Feature 2, 4)
                            render_recommendation_card(rec_title, rec_poster, rec_score)
                except MovieNotFoundError as e:
                    st.error(f"🔍 {e}")
                except EmptyInputError as e:
                    st.warning(f"⚠️ {e}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
        else:
            st.warning("⚠️ Please select a movie title.")

    # Sidebar: Render stats (Feature 8) and explanations
    st.sidebar.markdown("### 📋 About the System")
    st.sidebar.write(
        "A content-based recommendation system parsing TF-IDF tokens from TMDB 5000. "
        "Posters and summaries are retrieved dynamically via TMDB API integrations."
    )
    
    # Feature 8: System Statistics rendering
    render_sidebar_stats(len(movies_df), 5000)
            
# Sidebar status check
if models_exist():
    st.sidebar.success("✅ Models are loaded and ready.")
else:
    st.sidebar.warning("⚠️ Models not trained.")
