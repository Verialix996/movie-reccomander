import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

if not os.environ.get("GEMINI_API_KEY"):
    st.error("Missing GEMINI_API_KEY. Create a .env file with GEMINI_API_KEY=your_key_here")
    st.stop()

if not os.path.exists("data/movies.csv"):
    st.error("Dataset not found. Run `python data_prep.py` first to download and build the movie database.")
    st.stop()

from recommender import load_movies, get_candidates
from gemini_client import get_recommendations, get_next_question

FIRST_QUESTION = "What kind of mood are you in right now?"
TOTAL_QUESTIONS = 5

st.title("Movie Recommendation Chatbot")
st.caption("Answer a few quick questions and I'll find movies you'll love.")

# Initialize session state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("current_q", 0)
st.session_state.setdefault("answers", [])
st.session_state.setdefault("questions_asked", [])  # tracks actual questions shown
st.session_state.setdefault("recommendations", [])
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

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Render recommendation cards if done
if st.session_state.done and st.session_state.recommendations:
    for rec in st.session_state.recommendations:
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

    st.divider()
    if st.button("Start over", type="secondary"):
        for key in ["messages", "current_q", "answers", "questions_asked", "recommendations", "done"]:
            del st.session_state[key]
        st.rerun()

# Accept user input while Q&A is in progress
elif not st.session_state.done:
    if prompt := st.chat_input("Your answer..."):
        # Display and store user answer
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.answers.append(prompt)
        st.session_state.current_q += 1

        next_q_num = st.session_state.current_q

        if next_q_num < TOTAL_QUESTIONS:
            # Generate a dynamic follow-up question based on conversation so far
            with st.spinner(""):
                qa_so_far = list(zip(st.session_state.questions_asked, st.session_state.answers))
                next_question = get_next_question(qa_so_far)

            st.session_state.questions_asked.append(next_question)
            reply = f"**Question {next_q_num + 1}/{TOTAL_QUESTIONS}:** {next_question}"
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            # All questions answered — get recommendations
            thinking_msg = "Great answers! Let me find some movies for you..."
            with st.chat_message("assistant"):
                st.markdown(thinking_msg)
            st.session_state.messages.append({"role": "assistant", "content": thinking_msg})

            with st.spinner("Searching through the movie database..."):
                candidates = get_candidates(st.session_state.df, st.session_state.answers)
                qa_pairs = list(zip(st.session_state.questions_asked, st.session_state.answers))
                recs = get_recommendations(qa_pairs, candidates)
                st.session_state.recommendations = recs
                st.session_state.done = True

            st.rerun()
