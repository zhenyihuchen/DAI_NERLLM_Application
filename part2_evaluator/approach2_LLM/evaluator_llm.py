# evaluator_llm.py
from llm_client_groq import judge_answer_with_llm

# Weights for overall score
WEIGHTS = {
    "correctness": 0.50,
    "completeness": 0.30,
    "precision": 0.20,
}

def score_answer(student: str, reference: str, *, question: str = "") -> dict:
    judged = judge_answer_with_llm(question=question, reference=reference, student=student)
    aspects = judged["aspects"]
    overall_feedback = judged["overall_feedback"]

    # Weighted average (clip to 0..100)
    num = 0.0
    den = 0.0
    for k, w in WEIGHTS.items():
        num += w * aspects.get(k, {}).get("score", 0)
        den += w
    overall = max(0.0, min(100.0, num / max(den, 1e-9)))

    # Build a compact feedback block (overall + small aspect lines)
    lines = [overall_feedback.strip()]
    # Small lines: use short prefix emojis to visually separate
    for label, emoji in [("correctness", "✅"), ("completeness", "🧩"), ("precision", "🎯")]:
        a = aspects[label]
        lines.append(f"<span style='font-size:0.90em;color:#666;'>{emoji} {label.title()}: {a['score']}/100 — {a['feedback']}</span>")
    lines.append(f"<br><b>Example of correct answer (for reference):</b> {reference}")
    feedback_html = "<br>".join(lines)

    return {"score": round(overall, 1), "feedback": feedback_html}
