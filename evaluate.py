"""Validate with the provided sample questions.

questions.txt: one question per line. Optionally add the expected PDF page(s) after a '|',
e.g.   What drives sovereign spreads in EMDEs? | 119,120
Then:  python evaluate.py [--rerank]
Prints answers + retrieved pages; if expected pages are given, reports a retrieval hit rate.
Writes results.md.
"""
import sys
import time

import config
from rag import RAG

rerank = "--rerank" in sys.argv
rows = []
for line in (config.BASE_DIR / "questions.txt").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    q, _, exp = line.partition("|")
    rows.append((q.strip(), {int(x) for x in exp.replace(" ", "").split(",") if x}))

rag = RAG()
out, hits, graded = ["# Evaluation results\n"], 0, 0
for q, expected in rows:
    r = rag.answer(q, rerank=rerank)
    pages = r["retrieved_pages"]
    mark = ""
    if expected:
        graded += 1
        ok = bool(expected & set(pages)) or any(s["page"] in expected for s in r["sources"])
        hits += ok
        mark = "  [HIT]" if ok else f"  [MISS, expected {sorted(expected)}]"
    print(f"\nQ: {q}\nA: {r['answer']}\n   retrieved PDF pages: {pages}{mark}")
    out.append(f"## {q}\n\n{r['answer']}\n\n*Retrieved PDF pages: {pages}*{mark}\n")
    time.sleep(4)   # stay inside free-tier rate limits
if graded:
    summary = f"Retrieval hit rate: {hits}/{graded}"
    print("\n" + summary); out.append(f"\n**{summary}**")
(config.BASE_DIR / "results.md").write_text("\n".join(out), encoding="utf-8")
print("Saved results.md")
