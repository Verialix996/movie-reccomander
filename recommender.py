"""
Content-Based Recommender System
---------------------------------
Approach: Item profiles built from genre features, ranked against a user profile
derived from freeform answers using TF-IDF vectorization + cosine similarity.

This is the Content-Based Filtering approach taught in the course:
  u(x, i) = cos(x, i) = (x · i) / (||x|| · ||i||)

The TF-IDF + cosine similarity layer pre-ranks the full 144K movie catalog,
then Gemini (LLM layer) picks the final 2-3 with personalized explanations.
"""

import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Keyword → genre label (used to enrich user query before TF-IDF)
GENRE_MAP = {
    # English
    "funny": "Comedy", "laugh": "Comedy", "comedy": "Comedy", "humour": "Comedy", "humor": "Comedy",
    "scary": "Horror", "horror": "Horror", "creepy": "Horror", "frightening": "Horror",
    "action": "Action",
    "adventure": "Adventure",
    "drama": "Drama",
    "romance": "Romance", "romantic": "Romance", "love story": "Romance", "love": "Romance",
    "sci-fi": "Sci-Fi", "science fiction": "Sci-Fi", "scifi": "Sci-Fi", "space": "Sci-Fi",
    "thriller": "Thriller", "suspense": "Thriller",
    "animation": "Animation", "animated": "Animation", "cartoon": "Animation",
    "documentary": "Documentary", "doc": "Documentary",
    "fantasy": "Fantasy", "magic": "Fantasy",
    "mystery": "Mystery",
    "crime": "Crime",
    "musical": "Musical", "music": "Musical",
    "western": "Western",
    "war": "War",
    "family": "Family", "kids": "Family", "children": "Family",
    "biography": "Biography", "biopic": "Biography",
    "history": "History", "historical": "History",
    "sport": "Sport", "sports": "Sport",
    # Hebrew
    "קומדיה": "Comedy", "מצחיק": "Comedy", "הומור": "Comedy",
    "אימה": "Horror", "פחד": "Horror", "מפחיד": "Horror",
    "אקשן": "Action", "פעולה": "Action",
    "הרפתקה": "Adventure", "הרפתקאות": "Adventure",
    "דרמה": "Drama",
    "רומנטי": "Romance", "רומן": "Romance", "אהבה": "Romance",
    "מדע בדיוני": "Sci-Fi", "מדע-בדיוני": "Sci-Fi", "חלל": "Sci-Fi",
    "מתח": "Thriller", "מותחן": "Thriller",
    "אנימציה": "Animation", "מצויר": "Animation",
    "תיעודי": "Documentary", "דוקומנטרי": "Documentary",
    "פנטזיה": "Fantasy", "קסם": "Fantasy",
    "מסתורין": "Mystery", "חידה": "Mystery",
    "פשע": "Crime",
    "מוסיקלי": "Musical", "מוזיקה": "Musical",
    "מערבון": "Western",
    "מלחמה": "War",
    "ילדים": "Family", "משפחה": "Family",
    "ביוגרפיה": "Biography",
    "היסטורי": "History", "היסטוריה": "History",
    "ספורט": "Sport",
}

ERA_MAP = {
    "classic": (None, 1979), "old": (None, 1979), "vintage": (None, 1979),
    "80s": (1980, 1989), "eighties": (1980, 1989),
    "90s": (1990, 1999), "nineties": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "recent": (2015, None), "new": (2018, None), "latest": (2020, None), "newest": (2020, None),
    "modern": (2010, None),
    "last few years": (2018, None), "last year": (2022, None),
}


@st.cache_data
def load_movies(path: str = "data/movies.csv") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.dropna(subset=["title", "genres"], inplace=True)
    df = df[df["genres"] != "(no genres listed)"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df.dropna(subset=["year"], inplace=True)
    df["year"] = df["year"].astype(int)
    df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce")
    df["ml_rating"] = pd.to_numeric(df["ml_rating"], errors="coerce")
    df["score"] = df["imdb_rating"].combine_first(df["ml_rating"])
    df.dropna(subset=["score"], inplace=True)

    # Normalize genres: replace commas and pipes with spaces for TF-IDF tokenization
    df["genre_text"] = df["genres"].str.replace(r"[|,\-]", " ", regex=True).str.lower()

    return df.reset_index(drop=True)


@st.cache_resource
def build_tfidf_index(genre_texts: tuple) -> TfidfVectorizer:
    """
    Build TF-IDF matrix over genre strings (item profiles).
    Cached as a resource so the matrix is computed once per server process.
    """
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)
    vectorizer.fit(genre_texts)
    return vectorizer


def _build_user_query(answers: list[str]) -> str:
    """
    Build the user profile query string by:
    1. Combining all freeform answers
    2. Appending recognized genre labels (in English) to reinforce the signal
    This mirrors the 'user profile' concept from the lecture.
    """
    combined = " ".join(answers).lower()
    found_genres = []
    for keyword, genre in GENRE_MAP.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', combined):
            g = genre.lower().replace("-", " ")
            if g not in found_genres:
                found_genres.append(g)
    return combined + " " + " ".join(found_genres)


def _extract_era(text: str) -> tuple[int | None, int | None]:
    for keyword, (start, end) in ERA_MAP.items():
        if keyword in text:
            return start, end
    match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', text)
    if match:
        y = int(match.group(1))
        return y - 5, y + 5
    return None, None


def get_candidates(
    df: pd.DataFrame,
    answers: list[str],
    min_score: float = 6.0,
    top_n: int = 60,
) -> pd.DataFrame:
    """
    Content-Based Filtering using TF-IDF + cosine similarity.

    Step 1: Pre-filter by quality (min IMDb score) and era if mentioned.
    Step 2: Build TF-IDF item profiles from genre strings.
    Step 3: Build user profile vector from freeform answers.
    Step 4: Rank movies by cosine similarity between user profile and item profiles.
    Step 5: Return top_n matches for Gemini to make the final pick.
    """
    combined_text = " ".join(answers).lower()
    era_start, era_end = _extract_era(combined_text)

    # Step 1: quality + era filter
    pool = df[df["score"] >= min_score].copy()
    if era_start is not None:
        pool = pool[pool["year"] >= era_start]
    if era_end is not None:
        pool = pool[pool["year"] <= era_end]

    if len(pool) < 20:
        pool = df[df["score"] >= min_score].copy()

    # Step 2: TF-IDF item profiles
    vectorizer = build_tfidf_index(tuple(df["genre_text"].tolist()))
    item_matrix = vectorizer.transform(pool["genre_text"].tolist())

    # Step 3: user profile vector from freeform answers
    user_query = _build_user_query(answers)
    user_vector = vectorizer.transform([user_query])

    # Step 4: cosine similarity — u(x,i) = (x · i) / (||x|| · ||i||)
    similarities = cosine_similarity(user_vector, item_matrix).flatten()
    pool = pool.copy()
    pool["similarity"] = similarities

    # Step 5: rank by similarity, break ties with IMDb score
    pool = pool.sort_values(["similarity", "score"], ascending=[False, False])

    return pool.head(top_n).reset_index(drop=True)
