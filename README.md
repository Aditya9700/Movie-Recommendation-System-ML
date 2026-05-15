# Movie Recommendation System

A content-based movie recommendation system built with Python, Pandas, NLTK, Scikit-Learn, and Streamlit. This application recommends five similar movies based on metadata including genres, keywords, cast, crew, and overviews from the TMDB 5000 dataset.

## Features

- Natural Language Processing: Text cleaning, tokenization, and Porter Stemming via NLTK.
- Vectorization: TF-IDF vectorization and cosine similarity calculations.
- Web Dashboard: A clean Streamlit user interface featuring a searchable dropdown and recommendation cards with similarity scores.
- Automatic Training: Built-in functionality to train the model directly via the Streamlit interface if model files are missing.

## Project Structure

- data/: CSV files containing movie details and credits.
- notebooks/: Jupyter notebook for exploratory data analysis.
- src/: Core Python modules for preprocessing, training, and recommendations.
- app.py: Streamlit dashboard application.
- requirements.txt: Python package dependencies.

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd movie-recommender
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To process the data and generate the similarity matrix:
```bash
python src/train.py
```
This saves movies.pkl, similarity.pkl, and vectorizer.pkl to the root directory.

### Running the App
To start the Streamlit web dashboard:
```bash
streamlit run app.py
```
Access the application at http://localhost:8501.
