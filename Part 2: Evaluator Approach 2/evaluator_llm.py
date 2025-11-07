from llm_client_groq import judge_answer_with_llm

def score_answer(student: str, reference: str, *, question: str = "") -> dict:

    result = judge_answer_with_llm(question=question, reference=reference, student=student)

    result["feedback"] += f"\n\nExample of correct answer for reference: {reference}" # Append reference answer for clarity
    return result
