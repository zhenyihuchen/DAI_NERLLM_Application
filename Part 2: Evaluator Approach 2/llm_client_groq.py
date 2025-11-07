import os, json, re
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  

_JSON_EXTRACT_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_EXTRACT_RE.search(s)
        if m:
            return json.loads(m.group(0))
        return {"score": 0, "feedback": "Could not parse JSON from model output."}

PROMPT = """You are an impartial ML instructor. Evaluate the student's answer against the ground-truth.
Return ONLY a JSON object:
{{
  "score": number (0..100),
  "feedback": string (<= 120 words)
}}


Feedback guidelines:
- Address the student directly using "you".
- Be concise, positive and constructive.
- Begin by acknowledging what is correct.
- Then clearly explain what is missing, inaccurate, or vague compared to the reference.
- Ignore minor grammar or spelling issues unless they change meaning.
- Avoid starting too many sentences with "However" or "Additionally".
- Do NOT end with motivational or generic compliments.


Scoring guidelines (be fair and give partial credit):
- 95–100: Comprehensive and technically precise; all key aspects covered.
- 80–94: Main idea correct and clear; a few secondary details missing.
- 55–79: Core idea present but lacks important details or precision.
- 25–54: Partially correct or vague; significant gaps or misconceptions.
- 1–24: Mostly incorrect or off-target.
- 0: Irrelevant or empty.

Calibration rules:
- If the main idea is correct but brief (e.g., states purpose without mechanism), place in 55–65.
- If unsure between two bands, choose the higher one.
- Prioritize correctness of the central concept over exhaustive coverage.

Now evaluate:
Question: {question}

Reference answer:
\"\"\"{reference}\"\"\"

Student answer:
\"\"\"{student}\"\"\"

ONLY OUTPUT THE JSON OBJECT.
"""

def judge_answer_with_llm(question: str, reference: str, student: str):
    client = Groq(api_key=GROQ_API_KEY)
    prompt = PROMPT.format(question=question, reference=reference, student=student)
    # chat = client.chat.completions.create(
    #     model=MODEL,
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.2,
    # )
    # # llm_client_groq.py  (inside judge_answer_with_llm)
    chat = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},  # <— enforce JSON
    )
    out = chat.choices[0].message.content
    obj = _extract_json(out)
    score = float(obj.get("score", 0))
    feedback = str(obj.get("feedback", "")).strip()
    score = max(0.0, min(100.0, score))
    if not feedback:
        feedback = "No feedback provided."
    return {"score": round(score, 1), "feedback": feedback}
