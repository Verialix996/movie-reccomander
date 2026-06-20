# Movie Recommendation Chatbot

A conversational movie recommender that asks 4 short questions, returns personalised picks from a 144,000-movie catalogue, and lets you keep refining in free text — deployed on [Render](https://render.com).

**Live demo:** https://cyber-threat-prioritization-agent.onrender.com

---

## How it works

The system uses two layers:

1. **ML layer — TF-IDF + Cosine Similarity (Content-Based Filtering)**  
   Scores all 144K movies against the user's answers using term-frequency vectors built from genre strings. Returns the top 60 candidates.

2. **LLM layer — Gemini 2.0 Flash**  
   Receives the 60 candidates and picks the best 2–3 with a one-sentence personalised explanation. Also generates each follow-up question dynamically, and handles free-text refinement after the initial picks.

```
User answers (4 questions)
        │
        ▼
  TF-IDF vectoriser
  (built on genre strings of all 144K movies)
        │
        ▼
  Cosine similarity  →  top 60 candidates
        │
        ▼
  Gemini 2.0 Flash   →  final 2–3 picks + explanations
        │
        ▼
  Free-text refinement loop
  (user can keep chatting to refine picks)
```

---

## Data sources

The dataset is built by `data_prep.py` from two public sources:

### 1. IMDb Non-Commercial Datasets
- **URL:** https://datasets.imdbws.com
- **Files used:** `title.basics.tsv.gz` (titles, genres, year) and `title.ratings.tsv.gz` (average rating, vote count)
- **Filters applied:**
  - `titleType == "movie"` — movies only, no TV/shorts
  - `isAdult == 0` — non-adult content only
  - `numVotes >= 100` — removes obscure entries with unreliable ratings
  - Rows with missing genre or year are dropped

### 2. MovieLens Latest Dataset (GroupLens)
- **URL:** https://files.grouplens.org/datasets/movielens/ml-latest.zip
- **Content:** ~33M user ratings for ~87K movies
- **Usage:** Provides a community rating (`ml_rating`) for movies that exist in MovieLens but may have sparse IMDb votes. Ratings are normalised from the 0–5 scale to 0–10 to match IMDb.
- **Joined to IMDb** via the `links.csv` file which maps MovieLens IDs → IMDb IDs.

### Merge logic
IMDb is the primary source. MovieLens ratings are joined as a fallback:
- `score = imdb_rating` if available, else `ml_rating`
- Duplicates are dropped by IMDb ID

**Final dataset:** `data/movies.csv` — **144,080 movies**, columns: `imdb_id`, `title`, `year`, `genres`, `imdb_rating`, `imdb_votes`, `ml_rating`, `ml_votes`

---

## Recommendation algorithm

### Step 1 — User query construction (`recommender.py`)
All 4 free-text answers are concatenated. A keyword map (English + Hebrew) expands informal phrases to canonical genre labels — e.g. "מפחיד" → `Horror`, "tense" → `Thriller`. The expanded query is the **user profile vector**.

### Step 2 — Pre-filtering
Movies with `score < 6.0` are excluded. If the user mentioned an era ("90s", "classic", "recent"), the year range is narrowed accordingly.

### Step 3 — TF-IDF item profiles
A TF-IDF vectoriser is fitted on the genre strings of all 144K movies (1-gram + 2-gram, `min_df=2`). Each movie becomes a vector in genre-term space — this is the **item profile**.

### Step 4 — Cosine similarity ranking
```
similarity(user, movie) = (user_vector · movie_vector) / (‖user‖ · ‖movie‖)
```
Movies are ranked by similarity, with IMDb score as a tiebreaker. Top 60 are passed to Gemini.

### Step 5 — LLM re-ranking
Gemini receives the 60 candidates with their metadata and the full Q&A transcript. It selects the 2–3 best matches and writes one sentence per pick explaining exactly why it fits this user.

### Step 6 — Free-text refinement (optional)
After recommendations are shown, the user can keep chatting: "something older", "more action", "in Hebrew", "not horror". Each message re-runs the TF-IDF search enriched with the new text, then Gemini picks from that updated candidate set. The conversation history is passed so context is preserved across multiple refinements.

Gemini also handles factual questions about movies ("what language is Simhadri?") and redirects off-topic input ("who is the president?") without crashing.

### Why this two-layer design?
| | TF-IDF alone | LLM alone |
|---|---|---|
| Speed | Fast (milliseconds) | Slow (seconds) |
| Handles 144K movies | Yes | No (context window) |
| Understands nuance | No | Yes |
| Personalised explanation | No | Yes |

The ML layer handles scale; the LLM handles understanding.

---

## Input validation

Before advancing each question, the user's answer is validated by Gemini:
- **Valid:** any mood, genre, era, "skip", "I don't know", vague answers
- **Invalid:** pure gibberish (`xzqwerty!!`), off-topic questions (weather, coding), symbols only

Invalid input gets a polite redirect without advancing the question counter.

---

## Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the dataset (one-time, ~10 min, downloads ~1 GB)
python data_prep.py

# 3. Set your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 4. Run the app
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com

---

## Deployment (Render)

The `render.yaml` configures a Python web service:
```yaml
buildCommand: pip install -r requirements.txt
startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

Set `GEMINI_API_KEY` as an environment variable in the Render dashboard. The `data/movies.csv` file must be committed to the repo (or built as part of a custom build step).

---

## Project structure

```
├── app.py              # Streamlit UI and Q&A flow
├── recommender.py      # TF-IDF + cosine similarity pipeline
├── gemini_client.py    # Gemini API calls (questions + recommendations + validation)
├── data_prep.py        # One-time dataset builder (IMDb + MovieLens → movies.csv)
├── data/
│   └── movies.csv      # 144,080 movies (built by data_prep.py)
├── tests/
│   ├── test_browser_e2e.py     # Playwright headed tests (local server)
│   ├── test_live_headed.py     # Playwright headed tests (live Render URL)
│   ├── test_recommender.py     # Unit tests for the ML pipeline
│   └── test_gemini_client.py   # Unit tests for Gemini client
└── render.yaml         # Render deployment config
```

---

## Tech stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| ML (candidate ranking) | scikit-learn — TfidfVectorizer + cosine_similarity |
| LLM (questions + picks) | Google Gemini 2.0 Flash (`google-genai`) |
| Data processing | pandas, numpy |
| Testing | pytest, Playwright (Chromium) |
| Deployment | Render.com |
