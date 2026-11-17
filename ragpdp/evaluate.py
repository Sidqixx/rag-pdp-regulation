"""Measure quality using eval/questions.jsonl.

python -m ragpdp.evaluate            # retrieval: recall@5 and MRR per method
python -m ragpdp.evaluate --answers  # LLM answers: correct citations? refuses out-of-scope questions?
"""
import json
import re
import sys
import time

from ragpdp.answer import CHAT_MODEL, answer, citations, client, invalid_citations
from ragpdp.retrieve import ROOT, Index

K = 8  # number of pasal sent to the LLM (Index.search)


def load_questions() -> list[dict]:
    with open(ROOT / "eval/questions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eval_retrieval(idx: Index, qs: list[dict]) -> None:
    qs = [q for q in qs if q["expected"]]  # out-of-scope questions have no correct answer
    print(f"{len(qs)} questions\n")
    print(f"| Method | recall@5 | recall@{K} | MRR |\n|---|---|---|---|")
    for method in ("bm25", "hybrid", "embed"):
        r5 = rk = mrr = 0.0
        misses = []
        for q in qs:
            ranked = [idx.chunks[i]["id"] for i in idx.rank(q["q"], method)]
            exp = set(q["expected"])
            r5 += len(exp & set(ranked[:5])) / len(exp)
            rk += len(exp & set(ranked[:K])) / len(exp)
            mrr += 1 / (min(ranked.index(e) for e in exp) + 1)
            if not exp & set(ranked[:K]):
                misses.append(q["q"])
        n = len(qs)
        print(f"| {method} | {r5 / n:.3f} | {rk / n:.3f} | {mrr / n:.3f} |")
    print(f"\nComplete misses with embed (no correct pasal in top-{K}):")
    for m in misses:
        print(" -", m)


def eval_answers(idx: Index, qs: list[dict]) -> None:
    out = ROOT / "eval" / f"answers-{re.sub(r'\W', '_', CHAT_MODEL)}.jsonl"
    cited = expected_cited = bad = refused = 0
    inscope = [q for q in qs if q["expected"]]
    with out.open("w", encoding="utf-8") as f:
        for i, q in enumerate(qs, 1):
            if "localhost" not in str(client.base_url):
                time.sleep(7)  # Gemini Flash-Lite free tier: max 10 requests/minute
            hits, stream = answer(idx, q["q"])
            text = "".join(stream)
            cites = set(citations(text))
            invalid = invalid_citations(text, hits)
            if q["expected"]:
                labels = {f"{'UU 27/2022' if e.startswith('uu27') else 'PP 33/2026'} Pasal {e.split(':')[1]}" for e in q["expected"]}
                cited += bool(cites)
                expected_cited += bool(cites & labels)
                bad += bool(invalid)
            else:
                refused += not cites
            row = {**q, "sources": [c["id"] for c in hits], "answer": text, "invalid": invalid}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"\r{i}/{len(qs)}", end="", flush=True)
    n_out = len(qs) - len(inscope)
    print(f"\n\nModel: {CHAT_MODEL}  (full answers: {out.relative_to(ROOT)})\n")
    print("| Metric | Value |\n|---|---|")
    print(f"| Answers with citations | {cited}/{len(inscope)} |")
    print(f"| Cites the correct pasal | {expected_cited}/{len(inscope)} |")
    print(f"| Has hallucinated citations | {bad}/{len(inscope)} |")
    print(f"| Refuses out-of-scope questions | {refused}/{n_out} |")


if __name__ == "__main__":
    idx, qs = Index.load(), load_questions()
    eval_answers(idx, qs) if "--answers" in sys.argv else eval_retrieval(idx, qs)
