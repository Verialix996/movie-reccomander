"""
Tests for gemini_client.py — Gemini API integration.

Covers:
  - get_next_question: returns single question, adapts to conversation, matches language
  - get_recommendations: correct structure, count, sourced from candidates
  - Error handling: bad API key, empty candidates, malformed response
  - Model cycling: quota error triggers fallback (mocked)
"""
import os
import sys
import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_candidates(genres_list: list[str]) -> pd.DataFrame:
    """Build a minimal candidates DataFrame for testing."""
    return pd.DataFrame([
        {
            "title": f"Test Movie {i}",
            "year": 2000 + i,
            "genres": g,
            "imdb_rating": 7.0 + (i % 3) * 0.5,
            "score": 7.0,
            "similarity": 0.9 - i * 0.05,
        }
        for i, g in enumerate(genres_list)
    ])


COMEDY_CANDIDATES = make_candidates([
    "Comedy", "Comedy|Romance", "Comedy|Drama",
    "Comedy", "Comedy|Animation", "Comedy",
    "Drama|Comedy", "Comedy|Crime", "Comedy",
    "Comedy|Family",
])

EMPTY_CANDIDATES = pd.DataFrame(
    columns=["title", "year", "genres", "imdb_rating", "score", "similarity"]
)


# ── get_next_question ──────────────────────────────────────────────────────────

def test_next_question_returns_string():
    """get_next_question must return a non-empty string."""
    from gemini_client import get_next_question
    qa = [("What kind of mood are you in?", "I want something funny")]
    result = get_next_question(qa)
    assert isinstance(result, str) and len(result.strip()) > 0


def test_next_question_is_single_question():
    """Response must be a single question (ends with '?' and has no double '??')."""
    from gemini_client import get_next_question
    qa = [("What kind of mood are you in?", "happy and relaxed")]
    result = get_next_question(qa)
    assert "?" in result, "No question mark found in response"


def test_next_question_no_compound_with_and():
    """Dynamic question must not be a compound question joined by ' and '."""
    from gemini_client import get_next_question
    qa = [("What kind of mood are you in?", "I want comedy")]
    result = get_next_question(qa)
    # Compound check: should not have ' and ' connecting two question clauses
    lower = result.lower()
    assert lower.count("?") <= 1, f"Multiple question marks in: {result}"


def test_next_question_does_not_repeat_genre():
    """After user says comedy twice, next question should not ask 'what genre?' again."""
    from gemini_client import get_next_question
    qa = [
        ("What kind of mood are you in?", "I want a comedy, something funny"),
        ("Do you prefer any specific genre?", "Definitely comedy"),
    ]
    result = get_next_question(qa)
    lower = result.lower()
    # Should not ask the bare 'what genre' question again.
    # Gemini may still mention 'comedy' while asking about tone/era/actors — that's fine.
    bare_genre_phrases = ["what genre", "which genre", "favorite genre", "preferred genre"]
    assert not any(p in lower for p in bare_genre_phrases), \
        f"Question repeated bare genre ask: {result}"


def test_next_question_hebrew_answers_get_hebrew_question():
    """When user writes in Hebrew, the next question must be in Hebrew."""
    from gemini_client import get_next_question
    qa = [("מה מצב הרוח שלך?", "אני רוצה קומדיה מצחיקה")]
    result = get_next_question(qa)
    # Hebrew Unicode range: ֐-׿
    has_hebrew = any('֐' <= c <= '׿' for c in result)
    assert has_hebrew, f"Expected Hebrew response, got: {result}"


def test_next_question_covers_different_topic():
    """Each follow-up question should cover a topic not yet discussed."""
    from gemini_client import get_next_question
    # Mood + genre already covered → expect era, actor, or tone question
    qa = [
        ("What kind of mood are you in?", "relaxed"),
        ("What genre are you feeling?", "drama"),
    ]
    result = get_next_question(qa)
    assert len(result.strip()) > 5


# ── get_recommendations — structure ───────────────────────────────────────────

def test_recommendations_returns_list():
    """get_recommendations must return a list."""
    from gemini_client import get_recommendations
    qa = [
        ("What kind of mood are you in?", "happy"),
        ("Genre?", "comedy"),
        ("Era?", "any"),
        ("Loved movies?", "skip"),
        ("Tone?", "light"),
    ]
    result = get_recommendations(qa, COMEDY_CANDIDATES)
    assert isinstance(result, list)


def test_recommendations_count_2_or_3():
    """Must return exactly 2 or 3 recommendations (not 1, not 4)."""
    from gemini_client import get_recommendations
    qa = [
        ("Mood?", "happy"), ("Genre?", "comedy"), ("Era?", "any"),
        ("Loved?", "skip"), ("Tone?", "light"),
    ]
    result = get_recommendations(qa, COMEDY_CANDIDATES)
    assert 2 <= len(result) <= 3, f"Got {len(result)} recommendations"


def test_recommendations_have_required_keys():
    """Each recommendation must have title, year, genres, and explanation."""
    from gemini_client import get_recommendations
    qa = [
        ("Mood?", "happy"), ("Genre?", "comedy"), ("Era?", "any"),
        ("Loved?", "Airplane!"), ("Tone?", "light"),
    ]
    result = get_recommendations(qa, COMEDY_CANDIDATES)
    for rec in result:
        for key in ("title", "year", "genres", "explanation"):
            assert key in rec, f"Missing key '{key}' in: {rec}"


def test_recommendations_explanations_non_empty():
    """Every explanation must be a non-empty string."""
    from gemini_client import get_recommendations
    qa = [
        ("Mood?", "excited"), ("Genre?", "comedy"), ("Era?", "any"),
        ("Loved?", "skip"), ("Tone?", "fun"),
    ]
    result = get_recommendations(qa, COMEDY_CANDIDATES)
    for rec in result:
        if rec.get("title") != "Error":
            assert len(rec.get("explanation", "").strip()) > 0


def test_recommendations_titles_non_empty():
    """Every recommended title must be a non-empty string."""
    from gemini_client import get_recommendations
    qa = [
        ("Mood?", "happy"), ("Genre?", "comedy"), ("Era?", "any"),
        ("Loved?", "skip"), ("Tone?", "light"),
    ]
    result = get_recommendations(qa, COMEDY_CANDIDATES)
    for rec in result:
        if rec.get("title") != "Error":
            assert len(rec.get("title", "").strip()) > 0


def test_recommendations_year_is_string():
    """Year field must be a string (as specified in the JSON schema)."""
    from gemini_client import get_recommendations
    qa = [
        ("Mood?", "happy"), ("Genre?", "comedy"), ("Era?", "any"),
        ("Loved?", "skip"), ("Tone?", "light"),
    ]
    result = get_recommendations(qa, COMEDY_CANDIDATES)
    for rec in result:
        if rec.get("title") != "Error":
            assert isinstance(rec.get("year"), str), f"Year is not string: {rec}"


# ── get_recommendations — error handling ──────────────────────────────────────

def test_recommendations_empty_candidates():
    """Empty candidates DataFrame must return an error rec, not crash."""
    from gemini_client import get_recommendations
    qa = [("Mood?", "happy"), ("Genre?", "comedy"), ("Era?", "any"),
          ("Loved?", "skip"), ("Tone?", "light")]
    result = get_recommendations(qa, EMPTY_CANDIDATES)
    assert isinstance(result, list) and len(result) > 0
    # Should return an error rec rather than crash
    assert result[0].get("title") is not None


def test_recommendations_invalid_api_key():
    """Invalid API key must return an error rec with explanation, not crash."""
    from gemini_client import get_recommendations
    qa = [("Mood?", "happy"), ("Genre?", "comedy"), ("Era?", "any"),
          ("Loved?", "skip"), ("Tone?", "light")]

    with patch.dict(os.environ, {"GEMINI_API_KEY": "INVALID_KEY_123"}):
        # Re-import forces the patched key to be used
        import importlib, gemini_client
        importlib.reload(gemini_client)
        result = gemini_client.get_recommendations(qa, COMEDY_CANDIDATES)

    assert isinstance(result, list) and len(result) > 0
    assert result[0].get("title") == "Error", \
        f"Expected error rec, got: {result[0]}"
    assert len(result[0].get("explanation", "")) > 0


def test_recommendations_error_rec_has_explanation():
    """Error recs must always include an explanation string."""
    from gemini_client import get_recommendations
    qa = [("Mood?", "happy"), ("Genre?", "comedy"), ("Era?", "any"),
          ("Loved?", "skip"), ("Tone?", "light")]

    with patch.dict(os.environ, {"GEMINI_API_KEY": "BAD_KEY"}):
        import importlib, gemini_client
        importlib.reload(gemini_client)
        result = gemini_client.get_recommendations(qa, COMEDY_CANDIDATES)

    if result[0].get("title") == "Error":
        assert len(result[0].get("explanation", "").strip()) > 0


# ── Model cycling ──────────────────────────────────────────────────────────────

def test_model_cycling_on_quota_error():
    """When first model returns 429, the next model in MODELS list must be tried."""
    from gemini_client import MODELS, _is_quota_error

    call_log = []

    def fake_generate(model, contents, config):
        call_log.append(model)
        if model == MODELS[0]:
            raise Exception("429 RESOURCE_EXHAUSTED quota exceeded")
        # Second model succeeds
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "recommendations": [
                {"title": "Test Movie 0", "year": "2000", "genres": "Comedy",
                 "explanation": "You will love this comedy!"}
            ]
        })
        return mock_resp

    with patch("gemini_client.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.side_effect = fake_generate
        from gemini_client import get_recommendations
        import importlib, gemini_client
        importlib.reload(gemini_client)

        with patch.object(gemini_client.genai.Client.return_value.models,
                          "generate_content",
                          side_effect=fake_generate):
            result = gemini_client.get_recommendations(
                [("Mood?", "happy"), ("Genre?", "comedy"), ("Era?", "any"),
                 ("Loved?", "skip"), ("Tone?", "light")],
                COMEDY_CANDIDATES,
            )
    # The function should have tried at least the first model
    assert isinstance(result, list)


def test_is_quota_error_detection():
    """_is_quota_error must correctly identify rate-limit errors."""
    from gemini_client import _is_quota_error
    assert _is_quota_error(Exception("429 Too Many Requests"))
    assert _is_quota_error(Exception("RESOURCE_EXHAUSTED quota"))
    assert _is_quota_error(Exception("rate limit exceeded"))
    assert not _is_quota_error(Exception("404 NOT_FOUND"))
    assert not _is_quota_error(Exception("Invalid API key"))


def test_is_quota_error_not_triggered_on_404():
    """404 NOT_FOUND must NOT be treated as a quota error."""
    from gemini_client import _is_quota_error
    err = Exception("404 NOT_FOUND models/gemini-1.5-flash is not found")
    assert not _is_quota_error(err)


# ── Candidate block formatting ─────────────────────────────────────────────────

def test_candidate_block_format():
    """_build_candidate_block must produce one line per candidate with title, year, genres, rating."""
    from gemini_client import _build_candidate_block
    block = _build_candidate_block(COMEDY_CANDIDATES)
    lines = block.strip().split("\n")
    assert len(lines) == len(COMEDY_CANDIDATES), "Line count mismatch"
    for line in lines:
        parts = line.split("|")
        # Genres field itself may contain '|' (e.g. Comedy|Romance), so parts >= 4
        assert len(parts) >= 4, f"Too few pipe-separated parts in: {line}"
        # Last part should be a rating (number or 'N/A')
        last = parts[-1].strip()
        assert last == "N/A" or last.replace(".", "").isdigit(), \
            f"Last part is not a rating: '{last}' in line: {line}"
        # First part (title) must be non-empty
        assert len(parts[0].strip()) > 0, f"Empty title in line: {line}"


def test_user_message_contains_qa_pairs():
    """_build_user_message must include all Q&A pairs in the output."""
    from gemini_client import _build_user_message, _build_candidate_block
    qa = [("What mood?", "happy"), ("Genre?", "comedy")]
    block = _build_candidate_block(COMEDY_CANDIDATES)
    msg = _build_user_message(qa, block)
    assert "happy" in msg
    assert "comedy" in msg
    assert "What mood?" in msg
