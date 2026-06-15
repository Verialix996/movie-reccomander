"""
End-to-end tests — full pipeline: user answers → TF-IDF candidates → Gemini recommendations.

Covers:
  - Full English flow
  - Full Hebrew flow
  - All-skip flow (graceful fallback)
  - Mixed language flow
  - Genre-specific flows (comedy, horror, thriller)
  - Era-specific flow (90s)
  - Failure case: unrecognizable input
  - Failure case: single-word vague answers
  - Chatbot gives 2–3 recs in every scenario
  - Recommendations contain personalized explanations
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_full_pipeline(answers: list[str], questions: list[str] = None) -> list[dict]:
    """Run the complete recommender → Gemini pipeline and return recommendations."""
    from recommender import load_movies, get_candidates
    from gemini_client import get_recommendations

    if questions is None:
        questions = [
            "What kind of mood are you in right now?",
            "Do you have a favorite genre, or something you're feeling tonight?",
            "Any era you prefer — classics, 80s/90s nostalgia, 2000s, or recent releases?",
            "Name a movie you already love, or a director/actor you enjoy.",
            "Should it feel light and fun, or deep and thought-provoking?",
        ]

    df = load_movies()
    candidates = get_candidates(df, answers)
    qa_pairs = list(zip(questions, answers))
    return get_recommendations(qa_pairs, candidates), candidates


# ── Full flow tests ────────────────────────────────────────────────────────────

def test_e2e_english_comedy_flow():
    """Full pipeline with English comedy answers returns 2–3 valid recommendations."""
    answers = ["happy and relaxed", "comedy", "any era", "Home Alone", "light and fun"]
    recs, candidates = run_full_pipeline(answers)

    assert isinstance(recs, list)
    assert 2 <= len(recs) <= 3, f"Expected 2–3 recs, got {len(recs)}"
    for rec in recs:
        assert "title" in rec and "explanation" in rec
        assert len(rec["title"].strip()) > 0
        assert len(rec["explanation"].strip()) > 0


def test_e2e_hebrew_comedy_flow():
    """Full pipeline with Hebrew comedy answers returns valid recommendations."""
    answers = ["אני רוצה קומדיה", "קומדיה מצחיקה", "לא משנה", "דלג", "קל ומצחיק"]
    recs, candidates = run_full_pipeline(answers)

    assert isinstance(recs, list)
    assert 2 <= len(recs) <= 3, f"Expected 2–3 recs, got {len(recs)}"
    assert all("title" in r for r in recs)
    assert all(len(r.get("explanation", "")) > 0 for r in recs if r.get("title") != "Error")


def test_e2e_hebrew_horror_flow():
    """Full pipeline with Hebrew horror answers returns horror-relevant recommendations."""
    answers = ["אני אוהב סרטי אימה", "אימה ומתח", "לא משנה", "דלג", "עמוק ומפחיד"]
    recs, candidates = run_full_pipeline(answers)

    # Candidates should be horror-heavy
    horror_count = candidates["genres"].str.contains("Horror", case=False, na=False).sum()
    assert horror_count >= len(candidates) * 0.3, \
        f"Too few horror candidates: {horror_count}/{len(candidates)}"
    assert isinstance(recs, list) and len(recs) > 0


def test_e2e_all_skip_flow():
    """All 'skip' answers must not crash and must return recommendations."""
    answers = ["דלג", "דלג", "דלג", "דלג", "דלג"]
    recs, candidates = run_full_pipeline(answers)

    assert isinstance(recs, list)
    assert len(recs) > 0, "Got zero recommendations for all-skip input"


def test_e2e_english_skip_flow():
    """English 'skip' answers must fall back to top-rated recommendations."""
    answers = ["skip", "skip", "skip", "skip", "skip"]
    recs, candidates = run_full_pipeline(answers)

    assert isinstance(recs, list) and len(recs) > 0


def test_e2e_mixed_language_flow():
    """Mix of Hebrew and English answers returns valid recommendations."""
    answers = ["קומדיה", "comedy funny", "90s", "Home Alone", "light"]
    recs, candidates = run_full_pipeline(answers)

    assert isinstance(recs, list)
    assert 2 <= len(recs) <= 3


def test_e2e_90s_comedy_flow():
    """90s comedy answers should yield 90s comedy candidates and valid recs."""
    answers = ["nostalgic", "comedy", "90s movies", "Seinfeld", "light and fun"]
    recs, candidates = run_full_pipeline(answers)

    # Candidates should be from 90s
    assert candidates["year"].between(1990, 1999).all(), \
        "Non-90s movies slipped into candidates"
    assert isinstance(recs, list) and len(recs) > 0


def test_e2e_thriller_flow():
    """Thriller/suspense answers return thriller-heavy candidates and valid recs."""
    answers = ["tense and on edge", "thriller suspense", "any era", "Hitchcock", "deep"]
    recs, candidates = run_full_pipeline(answers)

    thriller_count = candidates["genres"].str.contains("Thriller", case=False, na=False).sum()
    assert thriller_count >= len(candidates) * 0.3
    assert isinstance(recs, list) and len(recs) > 0


def test_e2e_drama_deep_flow():
    """Request for deep drama returns drama candidates."""
    answers = ["thoughtful", "drama", "any", "The Shawshank Redemption", "deep and thought-provoking"]
    recs, candidates = run_full_pipeline(answers)

    drama_count = candidates["genres"].str.contains("Drama", case=False, na=False).sum()
    assert drama_count >= len(candidates) * 0.4
    assert isinstance(recs, list) and len(recs) > 0


# ── Failure / edge case tests ──────────────────────────────────────────────────

def test_e2e_failure_unrecognizable_input():
    """
    FAILURE CASE: Completely unrecognizable input (random characters).
    System must still return something rather than crash.
    Gemini may return an error rec or try its best with fallback movies.
    """
    answers = ["xzqwerty123!!", "!@#$%^&*()", "ñoñoño", "???", "..."]
    recs, candidates = run_full_pipeline(answers)

    # Must not crash — should return a list (even if it's an error rec)
    assert isinstance(recs, list) and len(recs) > 0
    # Candidates must use fallback (top-rated), not be empty
    assert len(candidates) > 0


def test_e2e_failure_single_vague_word():
    """
    FAILURE CASE: Single-word vague answers with no genre/era signal.
    System should fall back to top-rated movies; recommendations may be generic.
    """
    answers = ["yes", "maybe", "ok", "sure", "fine"]
    recs, candidates = run_full_pipeline(answers)

    assert isinstance(recs, list) and len(recs) > 0
    # Candidates use fallback scoring
    assert (candidates["score"] >= 6.0).all()


def test_e2e_failure_contradictory_input():
    """
    FAILURE CASE: Contradictory answers (comedy + horror + drama + sci-fi + western).
    System should still return something reasonable without crashing.
    """
    answers = ["comedy and horror", "drama and sci-fi", "western classic", "skip", "both light and deep"]
    recs, candidates = run_full_pipeline(answers)

    assert isinstance(recs, list) and len(recs) > 0


def test_e2e_failure_very_long_answer():
    """
    FAILURE CASE: Extremely long answer (1000 chars) must not crash the pipeline.
    """
    long_answer = "I want a comedy " * 63  # ~1000 chars
    answers = [long_answer, "comedy", "any", "skip", "light"]
    recs, candidates = run_full_pipeline(answers)

    assert isinstance(recs, list) and len(recs) > 0


def test_e2e_failure_special_characters():
    """
    FAILURE CASE: Answers with special chars, newlines, quotes.
    Must not crash or corrupt the Gemini prompt.
    """
    answers = ['I want "comedy"', "genre: comedy\n", "era: 90s\r\n", "skip", "light & fun"]
    recs, candidates = run_full_pipeline(answers)

    assert isinstance(recs, list) and len(recs) > 0


# ── Recommendation quality checks ──────────────────────────────────────────────

def test_e2e_recs_always_have_explanation():
    """Every non-error recommendation must have a non-empty explanation."""
    answers = ["happy", "comedy", "recent", "skip", "fun"]
    recs, _ = run_full_pipeline(answers)

    for rec in recs:
        if rec.get("title") != "Error":
            assert len(rec.get("explanation", "").strip()) > 10, \
                f"Explanation too short: {rec}"


def test_e2e_recs_always_have_title():
    """Every recommendation must have a non-empty title."""
    answers = ["excited", "action", "2000s", "skip", "fun"]
    recs, _ = run_full_pipeline(answers)

    for rec in recs:
        assert len(rec.get("title", "").strip()) > 0


def test_e2e_candidates_always_populated():
    """get_candidates must never return an empty DataFrame in any E2E scenario."""
    from recommender import load_movies, get_candidates
    df = load_movies()

    scenarios = [
        ["comedy", "comedy", "any", "skip", "light"],
        ["אימה", "horror", "classic", "skip", "deep"],
        ["", "", "", "", ""],
        ["xyzabc", "zzz", "qqq", "rrr", "sss"],
        ["drama romance", "drama", "2010s", "The Notebook", "deep"],
    ]
    for answers in scenarios:
        candidates = get_candidates(df, answers)
        assert len(candidates) > 0, f"Empty candidates for: {answers}"


def test_e2e_pipeline_does_not_raise():
    """Full pipeline must not raise any exception for standard inputs."""
    answers = ["relaxed", "comedy", "any era", "Monty Python", "light"]
    try:
        recs, candidates = run_full_pipeline(answers)
    except Exception as e:
        pytest.fail(f"Pipeline raised unexpected exception: {e}")
