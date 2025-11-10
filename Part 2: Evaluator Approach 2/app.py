# app.py
import os
import random
from datetime import datetime

import pandas as pd
import streamlit as st
from textblob import TextBlob

from evaluator_llm import score_answer
from utils import load_qa, make_question, pick_index



st.set_page_config(page_title="ML Q&A Evaluator", page_icon="🤖", layout="centered")
st.markdown(
    """
    <div style='text-align:center; margin-bottom:30px;'>
        <div style='font-size:2.5rem; font-weight:900; color:#222;'>
            ML Q&amp;A Evaluator
        </div>
        <div style='font-size:1.25rem; font-style:italic; color:#4da6ff; margin-top:5px;'>
            Your personal tool for learning!
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True) #empty line


# -- Load Q&A data --
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
if "seen_count" not in st.session_state:      
    st.session_state.seen_count = 1       # number of questions seen so far for stats at the top
if "mastered" not in st.session_state:
    st.session_state.mastered = set()     # indices of concepts scored >= 80 to avoid asking them again
if "rate_clarity" not in st.session_state:
    st.session_state.rate_clarity = None
if "rate_relevance" not in st.session_state:
    st.session_state.rate_relevance = None
if "rate_credibility" not in st.session_state:
    st.session_state.rate_credibility = None
if "rate_overall" not in st.session_state:
    st.session_state.rate_overall = None


COLUMNS = [
    "ts", "question", "student_answer", "reference_answer",
    "score", "feedback",
    "user_comment", "user_sentiment",
    "rate_clarity", "rate_relevance", "rate_credibility", "rate_overall",
]


# -- Helper functions --
def pick_new_idx(): # pick a new question index avoiding mastered concepts
    n = len(qa)
    pool = [i for i in range(n) if i not in st.session_state.mastered] # avoid mastered concepts
    if not pool:
        # everything is mastered: reset pool (but keep their achievements shown in metrics/log)
        st.toast("🎉 You’ve mastered all concepts this session. Resetting the pool.", icon="🎉")
        st.session_state.mastered.clear()
        pool = list(range(n))
    return random.choice(pool)

def next_question_cb(): 
    st.session_state.idx = pick_new_idx()
    st.session_state.student_text = ""
    st.session_state.user_eval = ""
    st.session_state.last_result = None
    st.session_state.show_feedback = False
    st.session_state.seen_count += 1
    for k in ["rate_clarity", "rate_relevance", "rate_credibility", "rate_overall"]:
        st.session_state.pop(k, None)

def do_evaluate(student_text, reference, question, top_buttons_placeholder=None):
    if not student_text.strip():
        st.warning("Please write an answer first.")
        return
    result = score_answer(student_text, reference, question=question)
    st.session_state.last_result = result
    st.session_state.show_feedback = True
    if top_buttons_placeholder is not None:
        top_buttons_placeholder.empty()  # hide top buttons immediately
    # mark mastered if high enough
    try:
        if result.get("score", 0) >= 75:
            st.session_state.mastered.add(st.session_state.idx)
            st.caption("✅ Concept marked as mastered (won’t be asked again this session).")
    except Exception:
        pass

# -- Display metrics at the top --
answered = len(st.session_state.log)

# average score over answered questions
if answered > 0:
    avg_score = sum(item["score"] for item in st.session_state.log if item.get("score") is not None) / answered
    avg_text = f"{avg_score:.1f}"
    # color rule
    if avg_score < 40:
        color = "#d32f2f"  # red
    elif avg_score <= 69:
        color = "#f9a825"  # yellow
    else:
        color = "#43b649"  # green
else:
    avg_text = "—"
    color = "#888888"


left, right = st.columns([1, 4])  
with left:
    st.markdown(
        f"<div style='font-size:0.9rem;color:#666;'>Average score</div>"
        f"<div style='font-size:1.8rem;font-weight:800;color:{color};'>{avg_text}</div>",
        unsafe_allow_html=True,
    )
with right:
    with st.expander("📊 Session progress", expanded=False):
        st.write(f"Questions seen: **{st.session_state.seen_count}**")
        st.write(f"Questions answered: **{answered}**")
        st.write(f"Mastered concepts: **{len(st.session_state.mastered)}**")

st.divider()



# -- End session screen --
if st.session_state.session_ended:
    # Build a DataFrame from the log 
    df = pd.DataFrame(st.session_state.log, columns=COLUMNS)


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
        st.session_state.idx = pick_new_idx()
        st.session_state.seen_count = 1
        st.session_state.mastered = set()
        st.rerun()

    st.stop()  # prevent rest of app from rendering



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

# --- Top buttons in a placeholder so we can remove them and move to bottom when feedback is shown ---
top_buttons = st.empty()

if not st.session_state.show_feedback:
    with top_buttons.container():
        col1, col2 = st.columns(2)
        with col1:
            evaluate = st.button("Evaluate")
        with col2:
            nextq = st.button("Next question", on_click=next_question_cb)
else:
    evaluate = nextq = None 


# -- Evaluate --
if evaluate:
    do_evaluate(st.session_state.student_text, ref, q, top_buttons_placeholder=top_buttons)

# If we have a prior result, show it so it persists across reruns
if st.session_state.last_result:
    res = st.session_state.last_result
    st.success(f"Score: **{res['score']} / 100**")
    formatted = res["feedback"].replace("\n", "<br>")
    st.markdown(f"**Feedback:**<br>{formatted}", unsafe_allow_html=True)

# -- Feedback box (only after Evaluate) --
user_eval = None
sent = None

if st.session_state.show_feedback and st.session_state.last_result:
    st.markdown("##### 💬 Rate this interaction (optional)")

    # helper to render stars in options and the selected value
    def fmt_opt(v):
        if v is None:
            return "— (skip)"
        return "★" * v + "☆" * (5 - v) + f"  ({v}/5)"

    options = [None, 0, 1, 2, 3, 4, 5]

    c1, c2 = st.columns(2)
    with c1:
        clarity = st.select_slider("Clarity of feedback", options=[None,0,1,2,3,4,5], value=None, format_func=fmt_opt, key="rate_clarity")
        credibility = st.select_slider("Credibility of feedback", options=[None,0,1,2,3,4,5], value=None, format_func=fmt_opt, key="rate_credibility")
    with c2:
        relevance = st.select_slider("Relevance of feedback", options=[None,0,1,2,3,4,5], value=None, format_func=fmt_opt, key="rate_relevance")
        overall = st.select_slider("Overall satisfaction", options=[None,0,1,2,3,4,5], value=None, format_func=fmt_opt, key="rate_overall")

    st.markdown("###### Optional comments")
    user_eval = st.text_input("Add a short comment (optional)", key="user_eval")
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
        "rate_clarity": st.session_state.get("rate_clarity"),
        "rate_relevance": st.session_state.get("rate_relevance"),
        "rate_credibility": st.session_state.get("rate_credibility"),
        "rate_overall": st.session_state.get("rate_overall"),
    }

    # Append once per question; update if the same question is being edited
    if not st.session_state.log or st.session_state.log[-1].get("question") != q:
        st.session_state.log.append(row)
    else:
        st.session_state.log[-1] = row

    # Always write with the same columns so ratings don't disappear
    os.makedirs("runs", exist_ok=True)
    pd.DataFrame(st.session_state.log, columns=COLUMNS).to_csv("runs/session_log.csv", index=False)
    st.caption("Saved to runs/session_log.csv")

# -- Bottom buttons (after feedback section) --
if st.session_state.show_feedback:
    st.markdown("<br>", unsafe_allow_html=True)
    col1b, col2b = st.columns(2)
    with col1b:
        eval_again = st.button("Evaluate again", key="eval_bottom")
    with col2b:
        st.button("Next question", on_click=next_question_cb, key="next_bottom")

    if eval_again:
        do_evaluate(st.session_state.student_text, ref, q)  
        st.rerun()



# -- End session --
st.divider()
if st.button("End study session", type="primary"):
    st.session_state.session_ended = True
    st.rerun()