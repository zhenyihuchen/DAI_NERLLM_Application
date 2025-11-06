# utils.py
import json, random, re


def load_qa(path="Q&A_db_practice.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"concept": d["question"], "answer": d["answer"]} for d in data]



def _leading_word(s: str) -> str:
    return re.sub(r"^[^A-Za-z]*", "", s).split()[0]


def question_variants(concept: str) -> list[str]:
    concept = concept.strip()
    #base = concept.split("(")[0].strip()  # drop "(AUC)" part if present
    base = concept

    variants = [
        f"Define {base}.",
        f"What does {base} mean?",
        f"Explain {base} in simple terms.",
        f"Give a concise definition of {base}.",
        f"What is {base}?",
        f"Describe the key idea behind {base}.",
    ]

    # De-duplicate while preserving order
    seen, deduped = set(), []
    for v in variants:
        if v not in seen:
            deduped.append(v); seen.add(v)
    return deduped

def make_question(concept: str, seed: int | None = None) -> str:
    rng = random.Random(seed)
    return rng.choice(question_variants(concept))

def pick_index(n):  # unchanged
    return random.randint(0, n-1)