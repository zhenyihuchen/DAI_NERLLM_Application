# llm_client_groq.py
import os, json, re
from groq import Groq
from dotenv import load_dotenv  


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  
#MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  

if not GROQ_API_KEY:
    raise ValueError(
        "❌ Missing GROQ_API_KEY. Please create a .env file with your key, e.g.:\n"
        "GROQ_API_KEY=sk_your_key_here"
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

def _strip_code_fences(s: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s

def _repair_invalid_escapes(s: str) -> str:
    return re.sub(r'\\(?![\\/"bfnrtu])', r'\\\\', s)

def _extract_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        candidate = _strip_code_fences(s)
        m = _JSON_OBJECT_RE.search(candidate)
        if m:
            cand = m.group(0).strip()
            cand = _repair_invalid_escapes(cand).replace("“","\"").replace("”","\"").replace("’","'")
            try:
                return json.loads(cand)
            except Exception:
                pass
    return {"aspects": {}, "overall_feedback": "Could not parse JSON from model output."}


PROMPT = """You are an impartial ML instructor. Evaluate the student's answer against the ground-truth.
Return ONLY a JSON object with this schema:
{{
  "aspects": {{
    "correctness":  {{ "score": number (0..100), "feedback": string (<= 30 words) }},
    "completeness": {{ "score": number (0..100), "feedback": string (<= 30 words) }},
    "precision":    {{ "score": number (0..100), "feedback": string (<= 30 words) }}
    }},
  "overall_feedback": string (<= 110 words)
}}


Scoring guidance:
- Correctness: technical accuracy of statements; no contradictions.
- Completeness: covers the key points in the reference; missing majors = larger penalty.
- Precision: clear, specific, and concise; avoid vague or rambling text.
For each aspect, evaluate ONLY that aspect, ignoring others.

"overall_feedback" guidelines:
- Address the student directly using "you".
- Be concise, positive and constructive.
- Begin by acknowledging what is correct.
- Then clearly explain what is missing, inaccurate, or vague compared to the reference.
- Ignore minor grammar or spelling issues unless they change meaning.
- Avoid starting too many sentences with "However" or "Additionally".
- Do NOT end with motivational or generic compliments.

Important:
- Use whole numbers 0..100.
- Be fair and give partial credit.
- Keep feedback short and actionable.
- If correctness ≥ 95 then completeness ≥ 90 (cannot be near zero).
- If correctness ≤ 20, completeness cannot exceed 40.
- If the student's answer is textually identical to the reference, set all three aspect scores to 100.


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

    chat = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},  # enforce JSON
    )
    out = chat.choices[0].message.content
    obj = _extract_json(out)

    # We return the raw aspects evaluation + overall text; compute the final weighted score.
    aspects = obj.get("aspects", {}) or {}
    overall_feedback = obj.get("overall_feedback", "").strip() or "No feedback provided."
    for k in ["correctness", "completeness", "precision"]:
        aspects.setdefault(k, {"score": 0, "feedback": ""})
        try:
            aspects[k]["score"] = int(aspects[k].get("score", 0))
        except Exception:
            aspects[k]["score"] = 0
        aspects[k]["feedback"] = str(aspects[k].get("feedback", "")).strip()

    return {"aspects": aspects, "overall_feedback": overall_feedback}