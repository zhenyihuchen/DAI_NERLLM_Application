# evaluator.py
import json, os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

_sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # ~80MB, CPU OK
_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

print("Evaluator model and scorer loaded.")

def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())


# For evaluating key word presence. Build a TF-IDF vectorizer once (corpus = all reference answers) to define key terms
_QA_PATH = os.getenv("QA_JSON_PATH", "Q&A_db_practice.json")

def _load_corpus(path=_QA_PATH):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [d["answer"] for d in data]
    except Exception:
        return [""]

_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),          # unigrams + bigrams
    stop_words="english",
    lowercase=True,
    max_features=8000
)
_vectorizer.fit(_load_corpus())

def _tfidf_keyterms(reference: str, top_k: int = 8, min_idf: float = 1.5):
    """Return top_k informative terms from reference using the fitted vectorizer."""
    if not reference.strip():
        return []

    X = _vectorizer.transform([reference])
    scores = X.toarray()[0]
    if scores.sum() == 0:
        return []

    vocab = _vectorizer.get_feature_names_out()
    # idf_ is aligned with vocab indices
    idf = _vectorizer.idf_

    # sort by TF-IDF descending, keep terms with decent IDF (not too common)
    idx_sorted = np.argsort(scores)[::-1]
    terms = []
    for i in idx_sorted:
        if scores[i] <= 0:
            continue
        if idf[i] < min_idf:
            continue
        term = vocab[i]
        # light filter: skip very short or numeric-ish tokens
        if re.fullmatch(r"[a-z][a-z0-9\-]{2,}", term):
            terms.append(term)
        if len(terms) >= top_k:
            break
    return terms

def _keyword_coverage(student: str, reference: str, top_k: int = 8):
    """Coverage over TF-IDF keyterms (case-insensitive, substring match)."""
    keyterms = _tfidf_keyterms(reference, top_k=top_k)
    if not keyterms:
        return 0.0, []
    s = student.lower()
    hits = sum(1 for t in keyterms if t in s)
    return hits / len(keyterms), keyterms




# Calculate score and give feedback:

def score_answer(student: str, reference: str) -> dict:
    student, reference = _clean(student), _clean(reference) # Remove extra spaces/newlines

    # 1) Semantic similarity (SBERT cosine)
    e_stu = _sbert.encode(student, normalize_embeddings=True)
    e_ref = _sbert.encode(reference, normalize_embeddings=True)
    sim = float(util.cos_sim(e_stu, e_ref)) # Similarity between both embeddings to define semantic match

    # 2) ROUGE-L (overlap/coverage)
    rougeL = _scorer.score(reference, student)["rougeL"].fmeasure # Rouge checks how many words or sequences overlap between your answer and the reference

    # 3) Keyword coverage (with key words defined with TF-IDF)
    kw_cov, keyterms = _keyword_coverage(student, reference, top_k=8)


    # Weighted score -> 0..100
    final = 100 * (0.6 * sim + 0.3 * rougeL + 0.1 * kw_cov)
    final = float(np.clip(final, 0, 100))

    # Short feedback
    def pct(x): return f"{round(100*x):d}%"
    missed = [t for t in keyterms if t not in student.lower()]
    feedback = (
        f"Semantic match {pct(sim)}<br>"
        f"Content overlap {pct(rougeL)}<br>"
        f"Keyword coverage {pct(kw_cov)}"
        f".  -> Some key words you missed: {( ', '.join(missed[:5]) if missed else '—')}.<br>"
        f"<b>Example of correct answer:</b> {reference}"
    )
    return {"score": round(final, 1), "feedback": feedback}


print("Evaluator ready.")
