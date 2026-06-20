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

SYSTEM_INSTRUCTION = """You are a movie expert. Recommend exactly 2 or 3 movies from the provided candidate list.
Choose movies that best match the user's mood, genre preference, era, and tone.
For each pick write ONE sentence explaining why it fits this user specifically.
No filler phrases ("Great choice!", "You'll love this!", "Based on your answers...").
No long descriptions. No plot summaries. Just the specific reason it matches.
Return ONLY valid JSON in this exact format:
{
  "recommendations": [
    {
      "title": "movie title exactly as given",
      "year": "YYYY",
      "genres": "Genre1|Genre2",
      "explanation": "one sentence: specific reason it fits this user"
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


QUESTION_SYSTEM = """You are helping find a movie for someone.
Ask ONE short question to learn what they want to watch.
Rules:
- Topics not yet covered: genre, decade/era, favorite movie or actor, light vs serious tone, language preference.
- Never repeat something already answered.
- One question only — no "and", no compound questions.
- Maximum one sentence. No preamble, no filler, no "Great answer!".
- Match the language the user is writing in.
Return ONLY the question text — nothing else."""


VALIDATE_SYSTEM = """You are a guard for a movie recommendation chatbot.
Decide if the user's message is a usable response for finding a movie to watch.
VALID: any opinion, genre, mood, actor, era, language, "skip", "I don't know", "anything", even vague answers.
INVALID: pure gibberish (random keys, symbols), questions/requests unrelated to movies (weather, math, coding, news), or clear attempts to misuse the bot.
Reply with exactly one word: VALID or INVALID."""


def is_valid_answer(text: str) -> bool:
    """Return True if the input is a usable answer for the movie Q&A."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=f'User message: "{text}"',
                config=types.GenerateContentConfig(
                    system_instruction=VALIDATE_SYSTEM,
                    temperature=0.0,
                    max_output_tokens=5,
                ),
            )
            return response.text.strip().upper().startswith("VALID")
        except Exception as e:
            if _is_quota_error(e):
                continue
            return True  # on unexpected error, let it through
    return True  # if all models fail, don't block the user


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


REFINE_SYSTEM = """You are a movie expert continuing a recommendation conversation.
The user already received movie picks and may want to refine them or ask about movies.

Decide which case applies:

CASE A — user wants DIFFERENT movies (asks to change genre, mood, era, director, actor, tone, language, or says "something else", "not that", "more like X"):
- Pick 2-3 new movies from the candidate list.
- Return: {"type": "recs", "intro": "one short sentence", "recommendations": [{"title": "...", "year": "...", "genres": "...", "explanation": "one sentence why it fits"}]}

CASE B — user asks a factual question ABOUT a movie (what language, who directed, what year, plot summary, cast):
- Answer the question directly and briefly. Do NOT suggest similar movies.
- Return: {"type": "chat", "message": "your factual answer"}

CASE C — user asks something completely unrelated to movies (politics, weather, math, coding, news):
- Politely redirect them.
- Return: {"type": "chat", "message": "I can only help with movie recommendations. Ask me to refine your picks or tell me what kind of film you're in the mood for."}

CASE D — anything else (follow-up question, general movie chat):
- Return: {"type": "chat", "message": "your short reply"}

Rules:
- Never invent movies. In CASE A, only recommend from the candidate list provided.
- No filler phrases. Keep responses short.
- Reply in the same language the user is writing in.
- Always return valid JSON matching one of the formats above."""


def refine_recommendations(
    qa_pairs: list[tuple[str, str]],
    previous_recs: list[dict],
    refinement_history: list[dict],
    candidates: pd.DataFrame,
) -> dict:
    """Handle free-text follow-up after initial recommendations."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

    qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)
    prev_text = "\n".join(
        f"- {r['title']} ({r.get('year', '')}): {r.get('explanation', '')}"
        for r in previous_recs if r.get("title") != "Error"
    )
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Bot'}: {m['content']}"
        for m in refinement_history
        if m.get("content")
    )
    candidate_block = _build_candidate_block(candidates)

    user_message = f"""Original Q&A:
{qa_text}

Previous recommendations:
{prev_text}

Follow-up conversation:
{history_text}

Candidate movies (title | year | genres | imdb_rating):
{candidate_block}"""

    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=REFINE_SYSTEM,
                    response_mime_type="application/json",
                    temperature=0.7,
                ),
            )
            data = json.loads(response.text)
            if isinstance(data, dict) and data.get("type") in ("recs", "chat"):
                return data
            # Gemini returned a JSON array or unexpected shape — treat as chat
            return {"type": "chat", "message": response.text.strip()}
        except (json.JSONDecodeError, KeyError):
            return {"type": "chat", "message": "Sorry, I couldn't process that. Try rephrasing?"}
        except Exception as e:
            if _is_quota_error(e):
                continue
            return {"type": "chat", "message": f"Error: {e}"}

    return {"type": "chat", "message": "All models are busy right now. Try again in a moment."}


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
