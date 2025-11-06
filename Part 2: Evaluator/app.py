# app.py
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from textblob import TextBlob

from evaluator import score_answer
from utils import load_qa, make_question, pick_index



st.set_page_config(page_title="ML Q&A Evaluator", page_icon="🤖", layout="centered")
st.title("🤖 ML Q&A Evaluator")

qa = load_qa()

# -- Session state init --
if "session_ended" not in st.session_state:
    st.session_state.session_ended = False
if "idx" not in st.session_state:
    st.session_state.idx = pick_index(len(qa))
if "log" not in st.session_state:
    st.session_state.log = []
if "student_text" not in st.session_state:
    st.session_state.student_text = ""
if "user_eval" not in st.session_state:
    st.session_state.user_eval = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None        # stores last evaluation result (score + feedback)
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False     # controls visibility of comment box

# -- End session screen --
if st.session_state.session_ended:
    # Build a DataFrame from the log 
    cols = ["ts","question","student_answer","reference_answer","score","feedback","user_comment","user_sentiment"]
    df = pd.DataFrame(st.session_state.log, columns=cols)

    # CSV bytes
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    # White screen with download button for log csv
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.download_button(
        "Download session log (CSV)",
        data=csv_bytes,
        file_name="study_session_log.csv",
        mime="text/csv",
        type="primary",
    )

    # Start fresh button 
    if st.button("Start a new session"):
        st.session_state.session_ended = False
        st.session_state.log = []
        st.session_state.student_text = ""
        st.session_state.user_eval = ""
        st.session_state.last_result = None
        st.session_state.show_feedback = False
        st.session_state.idx = pick_index(len(qa))
        st.rerun()

    st.stop()  # prevent rest of app from rendering

# -- Helpers --
def next_question_cb():
    st.session_state.idx = pick_index(len(qa))
    st.session_state.student_text = ""
    st.session_state.user_eval = ""
    st.session_state.last_result = None
    st.session_state.show_feedback = False

# -- Current item --
item = qa[st.session_state.idx]
concept = item["concept"]
ref = item["answer"]

q = make_question(concept, seed=st.session_state.idx)
st.markdown("### Question")
st.info(q)

# -- Answer box --
student = st.text_area(
    "Your answer",
    height=180,
    placeholder="Type your answer here...",
    key="student_text",
)

col1, col2 = st.columns(2)
with col1:
    evaluate = st.button("Evaluate")
with col2:
    st.button("Next question", on_click=next_question_cb)

# -- Evaluate --
if evaluate:
    if not student.strip():
        st.warning("Please write an answer first.")
    else:
        result = score_answer(student, ref)
        st.session_state.last_result = result      # persist across reruns
        st.session_state.show_feedback = True      # now show feedback/comment box

# If we have a prior result, show it so it persists across reruns
if st.session_state.last_result:
    res = st.session_state.last_result
    st.success(f"Score: **{res['score']} / 100**")
    # Render feedback with line breaks
    formatted = res["feedback"].replace("\n", "<br>")
    st.markdown(f"**Feedback:**<br>{formatted}", unsafe_allow_html=True)

# -- Feedback box (only after Evaluate) --
user_eval = None
sent = None
if st.session_state.show_feedback and st.session_state.last_result:
    st.markdown("##### 💬 Do you have any feedback about this question?")
    user_eval = st.text_input("How was this assessment? (optional)", key="user_eval")
    if user_eval:
        sent = TextBlob(user_eval).sentiment.polarity
        st.caption(f"Detected sentiment: {sent:+.2f}")

    # ---- Log / update current question entry ----
    # Build the row using the latest known values
    row = {
        "ts": datetime.utcnow().isoformat(),
        "question": q,
        "student_answer": student,
        "reference_answer": ref,
        "score": res["score"],
        "feedback": res["feedback"],
        "user_comment": user_eval if user_eval else None,
        "user_sentiment": sent if user_eval else None,
    }

    # Append once per question; update if the same question is being edited
    if not st.session_state.log or st.session_state.log[-1]["question"] != q:
        st.session_state.log.append(row)
    else:
        st.session_state.log[-1] = row

    os.makedirs("runs", exist_ok=True)
    pd.DataFrame(st.session_state.log).to_csv("runs/session_log.csv", index=False)
    st.caption("Saved to runs/session_log.csv")

# -- End session --
st.divider()
if st.button("End study session", type="primary"):
    st.session_state.session_ended = True
    st.rerun()