import os
import json
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Models tried in order when one hits a quota/rate-limit (free tier cycles through these)
MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
]

SYSTEM_INSTRUCTION = """You are a knowledgeable movie expert helping someone pick a film to watch.
Your job is to recommend exactly 2 or 3 movies ONLY from the provided candidate list.
Choose movies that best match the user's mood, genre preference, era, and tone from their answers.
Write a short, warm, personalized explanation (1-2 sentences) for each pick.
Return ONLY valid JSON in this exact format:
{
  "recommendations": [
    {
      "title": "movie title exactly as given",
      "year": "YYYY",
      "genres": "Genre1|Genre2",
      "explanation": "personalized reason why this fits the user"
    }
  ]
}
Never invent movies. Never recommend a movie not in the candidate list."""


def _build_candidate_block(candidates: pd.DataFrame) -> str:
    lines = []
    for _, row in candidates.iterrows():
        rating = f"{row['imdb_rating']:.1f}" if pd.notna(row.get("imdb_rating")) else "N/A"
        lines.append(f"{row['title']} | {row['year']} | {row['genres']} | {rating}")
    return "\n".join(lines)


def _build_user_message(qa_pairs: list[tuple[str, str]], candidate_block: str) -> str:
    qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)
    return f"""User preferences:
{qa_text}

Candidate movies (title | year | genres | imdb_rating):
{candidate_block}

Pick 2-3 movies from the list above that best match this user."""


QUESTION_SYSTEM = """You are helping find the perfect movie for someone.
Ask ONE short, focused question to learn more about what they want to watch.
Rules:
- Cover topics NOT already answered: genre, era/decade, favorite movies or actors, tone (light vs deep), language preference.
- Never ask about something the user already mentioned.
- Ask exactly one thing — no "and", no compound questions.
- Keep it short (one sentence).
- Ask in the same language the user is writing in.
Return ONLY the question text — no numbering, no prefix, nothing else."""


def get_next_question(qa_history: list[tuple[str, str]]) -> str:
    """Generate the next question based on what's already been asked and answered."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

    history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_history)
    user_message = f"Conversation so far:\n{history_text}\n\nWhat is the next question to ask?"

    last_error: Exception | None = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=QUESTION_SYSTEM,
                    temperature=0.5,
                ),
            )
            return response.text.strip()
        except Exception as e:
            if _is_quota_error(e):
                last_error = e
                continue
            return "Any specific actors, directors, or a movie you already love?"

    return "Any specific actors, directors, or a movie you already love?"


def _is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("429", "quota", "rate limit", "resource exhausted", "too many requests"))


def get_recommendations(
    qa_pairs: list[tuple[str, str]],
    candidates: pd.DataFrame,
) -> list[dict]:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

    candidate_block = _build_candidate_block(candidates)
    user_message = _build_user_message(qa_pairs, candidate_block)

    last_error: Exception | None = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=0.7,
                ),
            )
            data = json.loads(response.text)
            recs = data.get("recommendations", [])
            if not recs or not isinstance(recs, list):
                raise ValueError("Empty or invalid recommendations list")
            return recs
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return [{"title": "Error", "year": "", "genres": "", "explanation": f"Could not parse recommendations: {e}"}]
        except Exception as e:
            if _is_quota_error(e):
                last_error = e
                continue  # try next model
            return [{"title": "Error", "year": "", "genres": "", "explanation": f"Gemini API error: {e}"}]

    return [{"title": "Error", "year": "", "genres": "", "explanation": f"All models exhausted their quota. Try again later. ({last_error})"}]
