"""
Tests for recommender.py — Content-Based Filtering (TF-IDF + cosine similarity).

Covers:
  - Dataset loading integrity
  - English genre detection
  - Hebrew genre detection
  - Era filtering (80s, 90s, classic, recent, explicit year)
  - Mixed language input
  - Skip / empty answers (fallback behavior)
  - Minimum score filtering
  - Output shape and column correctness
  - Fallback when filters yield too few results
"""
import pytest
import pandas as pd


# ── Dataset loading ────────────────────────────────────────────────────────────

def test_load_movies_row_count(movies_df):
    """Dataset must have a substantial number of movies (>= 50,000)."""
    assert len(movies_df) >= 50_000, f"Only {len(movies_df)} movies loaded"


def test_load_movies_required_columns(movies_df):
    """Required columns must all be present."""
    required = {"title", "year", "genres", "score", "genre_text"}
    assert required.issubset(movies_df.columns), \
        f"Missing columns: {required - set(movies_df.columns)}"


def test_load_movies_no_null_titles(movies_df):
    """No movie should have a null title."""
    assert movies_df["title"].isna().sum() == 0


def test_load_movies_no_null_genres(movies_df):
    """No movie should have a null genres field."""
    assert movies_df["genres"].isna().sum() == 0


def test_load_movies_no_bad_genre_placeholder(movies_df):
    """'(no genres listed)' placeholder must be filtered out."""
    assert not (movies_df["genres"] == "(no genres listed)").any()


def test_load_movies_year_is_integer(movies_df):
    """Year column must be integer dtype."""
    assert movies_df["year"].dtype in ("int32", "int64")


def test_load_movies_score_range(movies_df):
    """All scores must be in valid IMDb range 1–10."""
    assert movies_df["score"].between(1, 10).all(), "Scores out of 1–10 range"


# ── English genre detection ────────────────────────────────────────────────────

def test_candidates_english_comedy(movies_df):
    """English comedy answers should return comedy-dominant results."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["funny comedy", "I love comedy", "", "", ""])
    comedy_count = candidates["genres"].str.contains("Comedy", case=False, na=False).sum()
    assert comedy_count >= len(candidates) * 0.5, \
        f"Only {comedy_count}/{len(candidates)} are comedy"


def test_candidates_english_horror(movies_df):
    """Horror answers should return horror-heavy results."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["scary horror films", "horror", "", "", ""])
    horror_count = candidates["genres"].str.contains("Horror", case=False, na=False).sum()
    assert horror_count >= len(candidates) * 0.4


def test_candidates_english_action(movies_df):
    """Action answers should return action results."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["action movie", "action", "", "", ""])
    action_count = candidates["genres"].str.contains("Action", case=False, na=False).sum()
    assert action_count >= len(candidates) * 0.4


# ── Hebrew genre detection ─────────────────────────────────────────────────────

def test_candidates_hebrew_comedy(movies_df):
    """Hebrew 'קומדיה' should map to Comedy and return comedy results."""
    from recommender import get_candidates
    candidates = get_candidates(
        movies_df,
        ["אני רוצה קומדיה", "קומדיה", "לא משנה", "דלג", "קל ומצחיק"]
    )
    comedy_count = candidates["genres"].str.contains("Comedy", case=False, na=False).sum()
    assert comedy_count >= len(candidates) * 0.5, \
        f"Only {comedy_count}/{len(candidates)} comedy from Hebrew input"


def test_candidates_hebrew_horror(movies_df):
    """Hebrew 'אימה' should map to Horror."""
    from recommender import get_candidates
    candidates = get_candidates(
        movies_df,
        ["אני אוהב סרטי אימה", "אימה ומתח", "", "", ""]
    )
    horror_count = candidates["genres"].str.contains("Horror", case=False, na=False).sum()
    assert horror_count >= len(candidates) * 0.3


def test_candidates_hebrew_action(movies_df):
    """Hebrew 'אקשן' should map to Action."""
    from recommender import get_candidates
    candidates = get_candidates(
        movies_df,
        ["אקשן", "סרטי פעולה", "", "", ""]
    )
    action_count = candidates["genres"].str.contains("Action", case=False, na=False).sum()
    assert action_count >= len(candidates) * 0.3


def test_candidates_hebrew_romance(movies_df):
    """Hebrew 'רומנטי' should map to Romance."""
    from recommender import get_candidates
    candidates = get_candidates(
        movies_df,
        ["רומנטי", "סרט אהבה", "", "", ""]
    )
    romance_count = candidates["genres"].str.contains("Romance", case=False, na=False).sum()
    assert romance_count >= len(candidates) * 0.3


# ── Era filtering ──────────────────────────────────────────────────────────────

def test_candidates_era_90s(movies_df):
    """'90s' keyword must filter movies to 1990–1999 range."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["", "", "I love 90s movies", "", ""])
    assert candidates["year"].between(1990, 1999).all(), \
        f"Non-90s movies found: {candidates[~candidates['year'].between(1990,1999)][['title','year']].head()}"


def test_candidates_era_80s(movies_df):
    """'80s' keyword must filter movies to 1980–1989 range."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["", "", "80s nostalgia", "", ""])
    assert candidates["year"].between(1980, 1989).all()


def test_candidates_era_classic(movies_df):
    """'classic' keyword must filter movies to pre-1980."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["", "", "classic old films", "", ""])
    assert (candidates["year"] <= 1979).all()


def test_candidates_era_recent(movies_df):
    """'recent' keyword must filter movies to 2015+."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["", "", "recent releases", "", ""])
    assert (candidates["year"] >= 2015).all()


def test_candidates_era_explicit_year(movies_df):
    """An explicit year like '1994' should filter to a ±5 year window."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["", "", "I loved movies from 1994", "", ""])
    assert candidates["year"].between(1989, 1999).all()


# ── Mixed language ─────────────────────────────────────────────────────────────

def test_candidates_mixed_language(movies_df):
    """Mix of Hebrew genre + English era should work correctly."""
    from recommender import get_candidates
    candidates = get_candidates(
        movies_df,
        ["קומדיה", "funny", "90s", "", "light"]
    )
    assert len(candidates) > 0
    comedy_count = candidates["genres"].str.contains("Comedy", case=False, na=False).sum()
    assert comedy_count >= len(candidates) * 0.4


# ── Skip / empty answers (fallback) ───────────────────────────────────────────

def test_candidates_all_skip(movies_df):
    """All 'skip' answers must not crash and must return results via fallback."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["דלג", "דלג", "דלג", "דלג", "דלג"])
    assert len(candidates) > 0


def test_candidates_all_empty(movies_df):
    """All empty answers must return top-rated movies as fallback."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["", "", "", "", ""])
    assert len(candidates) > 0
    assert (candidates["score"] >= 6.0).all()


def test_candidates_nonsense_input(movies_df):
    """Completely unrecognized input ('asdfghjkl') must fall back gracefully."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["asdfghjkl", "zzzzz", "xyzxyz", "", ""])
    assert len(candidates) > 0


def test_candidates_emoji_input(movies_df):
    """Emoji-only input must not crash."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["😀🎬🍿", "❤️", "🕰️", "🎭", "😂"])
    assert len(candidates) > 0


# ── Output shape and quality ───────────────────────────────────────────────────

def test_candidates_returns_top_n(movies_df):
    """get_candidates must return exactly top_n rows (default 60)."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["comedy", "", "", "", ""])
    assert len(candidates) == 60


def test_candidates_custom_top_n(movies_df):
    """get_candidates must respect a custom top_n argument."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["comedy", "", "", "", ""], top_n=20)
    assert len(candidates) == 20


def test_candidates_has_similarity_column(movies_df):
    """Output must include the 'similarity' column from cosine similarity."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["action thriller", "", "", "", ""])
    assert "similarity" in candidates.columns


def test_candidates_similarity_values_valid(movies_df):
    """Cosine similarity scores must be in [0, 1]."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["drama romance", "", "", "", ""])
    assert candidates["similarity"].between(0.0, 1.0).all()


def test_candidates_sorted_by_similarity(movies_df):
    """Results must be sorted by similarity descending."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["comedy funny", "", "", "", ""])
    sims = candidates["similarity"].tolist()
    assert sims == sorted(sims, reverse=True), "Results not sorted by similarity"


def test_candidates_min_score_respected(movies_df):
    """All returned movies must meet the minimum score threshold."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["comedy", "", "", "", ""], min_score=6.0)
    assert (candidates["score"] >= 6.0).all()


def test_candidates_required_output_columns(movies_df):
    """Output DataFrame must have all columns needed by gemini_client."""
    from recommender import get_candidates
    candidates = get_candidates(movies_df, ["comedy", "", "", "", ""])
    for col in ["title", "year", "genres", "score", "similarity"]:
        assert col in candidates.columns, f"Missing column: {col}"


# ── Fallback when filters too restrictive ─────────────────────────────────────

def test_candidates_fallback_on_tiny_era_genre_combo(movies_df):
    """Very niche combination (e.g., Musical + classic) must still return results."""
    from recommender import get_candidates
    candidates = get_candidates(
        movies_df,
        ["musical", "musical", "classic films from before 1940", "", ""]
    )
    assert len(candidates) > 0


def test_candidates_never_returns_empty(movies_df):
    """get_candidates must never return an empty DataFrame."""
    from recommender import get_candidates
    for answers in [
        ["", "", "", "", ""],
        ["xyzabc123", "", "", "", ""],
        ["דלג"] * 5,
        ["🎬"] * 5,
    ]:
        result = get_candidates(movies_df, answers)
        assert len(result) > 0, f"Empty result for answers: {answers}"
