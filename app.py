import os
import subprocess
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

if not os.environ.get("GEMINI_API_KEY"):
    st.error("Missing GEMINI_API_KEY. Create a .env file with GEMINI_API_KEY=your_key_here")
    st.stop()

if not os.path.exists("data/movies.csv"):
    with st.spinner("Dataset not found. Building movie database..."):
        try:
            subprocess.run([sys.executable, "data_prep.py"], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Dataset build failed. Run `python data_prep.py` manually and check the logs. ({e})")
            st.stop()

    if not os.path.exists("data/movies.csv"):
        st.error("Dataset build finished, but data/movies.csv was not created.")
        st.stop()

from recommender import load_movies, get_candidates
from gemini_client import get_recommendations, get_next_question, is_valid_answer, refine_recommendations

FIRST_QUESTION = "What kind of mood are you in right now?"
TOTAL_QUESTIONS = 4
OUT_OF_SCOPE_MSG = (
    "I'm a movie recommendation chatbot — I can only help you find a film to watch. "
    "Just answer the question above, or type **skip** to move on."
)

st.title("Movie Recommendation Chatbot")
st.caption("Answer a few quick questions and I'll find movies you'll love.")

# Initialize session state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("current_q", 0)
st.session_state.setdefault("answers", [])
st.session_state.setdefault("questions_asked", [])
st.session_state.setdefault("recommendations", [])
st.session_state.setdefault("refinement_history", [])  # follow-up messages after recs
st.session_state.setdefault("df", None)
st.session_state.setdefault("done", False)

# Load dataset once
if st.session_state.df is None:
    with st.spinner("Loading movie database..."):
        st.session_state.df = load_movies()

# Inject first question if chat is empty
if not st.session_state.messages and st.session_state.current_q == 0:
    intro = f"Hi! I'm going to ask you {TOTAL_QUESTIONS} short questions to find the perfect movie for you.\n\n**Question 1/{TOTAL_QUESTIONS}:** {FIRST_QUESTION}"
    st.session_state.messages.append({"role": "assistant", "content": intro})
    st.session_state.questions_asked.append(FIRST_QUESTION)

# Replay Q&A chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def render_rec_cards(recs: list[dict]):
    """Render movie recommendation cards."""
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        if rec.get("title") == "Error":
            st.error(rec.get("explanation", "An error occurred."))
            continue
        genres_display = rec.get("genres", "").replace("|", " · ")
        year = rec.get("year", "")
        with st.chat_message("assistant"):
            st.markdown(
                f"### {rec['title']} ({year})\n"
                f"*{genres_display}*\n\n"
                f"{rec.get('explanation', '')}"
            )


# ── Recommendation + refinement phase ──────────────────────────────────────────
if st.session_state.done and st.session_state.recommendations:

    # Render original recommendation cards
    render_rec_cards(st.session_state.recommendations)

    # Render refinement conversation history
    for entry in st.session_state.refinement_history:
        if entry["role"] == "recs":
            if entry.get("intro"):
                with st.chat_message("assistant"):
                    st.markdown(entry["intro"])
            render_rec_cards(entry["recs"])
        else:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])

    st.divider()
    if st.button("Start over", type="secondary"):
        for key in ["messages", "current_q", "answers", "questions_asked",
                    "recommendations", "refinement_history", "done"]:
            del st.session_state[key]
        st.rerun()

    # Free-text refinement input
    if follow_up := st.chat_input("Want something different? Tell me more..."):
        with st.chat_message("user"):
            st.markdown(follow_up)
        st.session_state.refinement_history.append({"role": "user", "content": follow_up})

        with st.spinner(""):
            # Enrich candidate search with the follow-up text
            enriched_answers = st.session_state.answers + [follow_up]
            candidates = get_candidates(st.session_state.df, enriched_answers)
            qa_pairs = list(zip(st.session_state.questions_asked, st.session_state.answers))
            result = refine_recommendations(
                qa_pairs,
                st.session_state.recommendations,
                st.session_state.refinement_history,
                candidates,
            )

        if result["type"] == "recs":
            entry = {"role": "recs", "intro": result.get("intro", ""), "recs": result.get("recommendations", [])}
            st.session_state.refinement_history.append(entry)
        else:
            msg = result.get("message", "")
            st.session_state.refinement_history.append({"role": "assistant", "content": msg})

        st.rerun()

# ── Q&A phase ──────────────────────────────────────────────────────────────────
elif not st.session_state.done:
    if prompt := st.chat_input("Your answer..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Validate before advancing
        with st.spinner(""):
            valid = is_valid_answer(prompt)

        if not valid:
            with st.chat_message("assistant"):
                st.markdown(OUT_OF_SCOPE_MSG)
            st.session_state.messages.append({"role": "assistant", "content": OUT_OF_SCOPE_MSG})
            st.stop()

        st.session_state.answers.append(prompt)
        st.session_state.current_q += 1

        next_q_num = st.session_state.current_q

        if next_q_num < TOTAL_QUESTIONS:
            with st.spinner(""):
                qa_so_far = list(zip(st.session_state.questions_asked, st.session_state.answers))
                next_question = get_next_question(qa_so_far)

            st.session_state.questions_asked.append(next_question)
            reply = f"**Question {next_q_num + 1}/{TOTAL_QUESTIONS}:** {next_question}"
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            with st.spinner("Searching through the movie database..."):
                candidates = get_candidates(st.session_state.df, st.session_state.answers)
                qa_pairs = list(zip(st.session_state.questions_asked, st.session_state.answers))
                recs = get_recommendations(qa_pairs, candidates)
                st.session_state.recommendations = recs
                st.session_state.done = True

            st.rerun()
