"""Advanced retrieval + context management.

Query  -> (optional) rewrite follow-ups into a standalone question
       -> hybrid retrieval: Gemini dense embeddings + BM25 keyword search, fused with RRF
       -> metadata boosts (explicit Figure/Table/Box/Chapter mentions, region names)
          and optional chapter filter
       -> optional LLM reranking (precision mode)
       -> context assembly: neighbour expansion inside the same section, de-duplication,
          document order, grouped by section breadcrumb, character budget
       -> grounded answer with page citations
"""
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np
from google import genai
from google.genai import types

import config

STOP = set("a an and are as at be by for from has have in is it its of on or that the their this to was were which with will what how does do did about into than then these those who whom whose can could should would".split())

REGION_ALIASES = {
    "east asia": "East Asia and Pacific", "china": "East Asia and Pacific",
    "europe and central asia": "Europe and Central Asia", "central asia": "Europe and Central Asia",
    "latin america": "Latin America and the Caribbean", "caribbean": "Latin America and the Caribbean",
    "middle east": "Middle East, North Africa", "north africa": "Middle East, North Africa",
    "pakistan": "Middle East, North Africa", "afghanistan": "Middle East, North Africa",
    "south asia": "South Asia", "india": "South Asia",
    "sub-saharan": "Sub-Saharan Africa", "africa": "Sub-Saharan Africa",
}

SYSTEM_PROMPT = """You answer questions about the World Bank report "Global Economic Prospects, June 2026".
Use ONLY the numbered context excerpts. Each excerpt header shows where it sits in the report (chapter > section) and its page.
Rules:
- If the excerpts do not contain the answer, say you could not find it in the report. Never guess or use outside knowledge.
- Quote figures exactly as written, with units, years and the country/region group they refer to.
- For comparisons or multi-part questions, cover every part and say which parts are missing from the excerpts.
- Charts are not available as data: if a number appears to live only in a figure or table that is not in the excerpts, say so.
- Be concise. End with the sources you used, formatted like: (Sources: PDF p.31; PDF p.45-46)."""


def tokenize(text):
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [t for t in toks if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype="float32")
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.inv = defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                self.inv[w].append(i)

    def scores(self, query):
        s = np.zeros(len(self.tf), dtype="float32")
        for w in set(tokenize(query)):
            if w not in self.idf:
                continue
            for i in self.inv[w]:
                f = self.tf[i][w]
                s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def rrf(rank_lists, k=60):
    out = defaultdict(float)
    for ranks in rank_lists:
        for r, i in enumerate(ranks):
            out[i] += 1.0 / (k + r + 1)
    return out


def page_label(c):
    a, b = c["page_start"], c["page_end"]
    return f"PDF p.{a}" if a == b else f"PDF p.{a}-{b}"


class RAG:
    def __init__(self):
        if not config.API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing. Create a .env file (see .env.example).")
        self.client = genai.Client(api_key=config.API_KEY)
        self.vectors = np.load(config.INDEX_DIR / "embeddings.npy")
        self.chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.bm25 = BM25([c["breadcrumb"] + " " + c["text"] for c in self.chunks])
        self.chapters = list(dict.fromkeys(c["chapter"] for c in self.chunks))

    # ---------- LLM helpers ----------
    def _generate(self, prompt, system=None, temperature=0.1):
        res = self.client.models.generate_content(
            model=config.CHAT_MODEL, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
        )
        return (res.text or "").strip()

    def embed_query(self, q):
        res = self.client.models.embed_content(
            model=config.EMBED_MODEL, contents=[q],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=config.EMBED_DIM),
        )
        v = np.array(res.embeddings[0].values, dtype="float32")
        return v / np.linalg.norm(v)

    def rewrite(self, question, history):
        """Turn a follow-up ('and for South Asia?') into a standalone question."""
        if not history:
            return question
        convo = "\n".join(f"{h['role']}: {h['content'][:600]}" for h in history[-4:])
        prompt = ("Rewrite the user's last question as a standalone search question, using the conversation for context. "
                  "Keep names, numbers and report terms. Output only the question.\n\n"
                  f"Conversation:\n{convo}\n\nLast question: {question}\nStandalone question:")
        try:
            return self._generate(prompt, temperature=0) or question
        except Exception:
            return question

    # ---------- retrieval ----------
    def _boosts(self, query):
        """Explicit mentions of Figure/Table/Box/Chapter ids and regions give a metadata boost."""
        boost = defaultdict(float)
        ids = {f"{m.group(1).upper()} {m.group(2).upper()}"
               for m in re.finditer(r"\b(figure|table|box|annex)\s+(b?\d+(?:\.\d+)*)", query, re.I)}
        chap = {m.group(1) for m in re.finditer(r"\bchapter\s+(\d)", query, re.I)}
        q = query.lower()
        regions = {v for k, v in REGION_ALIASES.items() if k in q}
        for c in self.chunks:
            i = c["id"]
            if ids and (ids & set(c["captions"]) or ids & set(c["refs"])):
                boost[i] += 0.05
            for x in ids:
                if x.startswith("BOX ") and c["box"] and c["box"].upper().startswith(x):
                    boost[i] += 0.05
            if chap and any(c["chapter"].startswith(f"Ch. {n}") for n in chap):
                boost[i] += 0.03
            if regions and c["region"] and any(c["region"].startswith(r) for r in regions):
                boost[i] += 0.03
        return boost

    def retrieve(self, query, chapter=None, k=config.FINAL_K, rerank=False):
        allowed = np.array([(chapter is None or c["chapter"] == chapter) for c in self.chunks])
        dense = self.vectors @ self.embed_query(query)
        sparse = self.bm25.scores(query)
        dense = np.where(allowed, dense, -1e9)
        sparse = np.where(allowed, sparse, -1e9)
        d_rank = [int(i) for i in np.argsort(-dense)[:config.CANDIDATES]]
        s_rank = [int(i) for i in np.argsort(-sparse)[:config.CANDIDATES] if sparse[i] > 0]
        fused = rrf([d_rank, s_rank])
        for i, b in self._boosts(query).items():
            if allowed[i]:
                fused[i] = fused.get(i, 0) + b
        ranked = sorted(fused, key=lambda i: -fused[i])
        pool = ranked[:config.RERANK_POOL if rerank else k]
        if rerank:
            pool = self.rerank(query, pool, k)
        return [{**self.chunks[i], "score": round(float(fused[i]), 4),
                 "dense": round(float(dense[i]), 3)} for i in pool]

    def rerank(self, query, ids, k):
        listing = "\n".join(f"[{i}] ({self.chunks[i]['breadcrumb'].split(' > ', 1)[-1]}) {self.chunks[i]['text'][:350]}" for i in ids)
        prompt = (f"Question: {query}\n\nPassages:\n{listing}\n\n"
                  f"Return a JSON list of the ids of the {k} passages most useful for answering the question, best first. "
                  "Output only the JSON list, e.g. [12, 4, 30].")
        try:
            out = self._generate(prompt, temperature=0)
            picked = [int(x) for x in re.findall(r"\d+", out) if int(x) in ids]
            picked = list(dict.fromkeys(picked))[:k]
            rest = [i for i in ids if i not in picked]
            return (picked + rest)[:k] if picked else ids[:k]
        except Exception:
            return ids[:k]

    # ---------- context management ----------
    def build_context(self, hits):
        """Neighbour expansion within the same section, merged ranges, document order, size budget."""
        by_id = self.chunks
        chosen = [h["id"] for h in hits]
        selected, size = [], 0

        def same_section(a, b):
            x, y = by_id[a], by_id[b]
            return (x["chapter"], x["section"], x["region"]) == (y["chapter"], y["section"], y["region"])

        def add(i):
            nonlocal size
            if i in selected or i < 0 or i >= len(by_id):
                return False
            cost = len(by_id[i]["text"]) + 150
            if size + cost > config.MAX_CONTEXT_CHARS:
                return False
            selected.append(i)
            size += cost
            return True

        for i in chosen:                           # primary hits first
            add(i)
        for i in chosen:                           # then neighbours, best hit first
            for j in (i - 1, i + 1):
                if 0 <= j < len(by_id) and same_section(i, j):
                    add(j)
        selected.sort()
        blocks, sources, prev = [], [], None
        for n, i in enumerate(selected, start=1):
            c = by_id[i]
            crumb = c["breadcrumb"].split(" > ", 1)[-1]
            blocks.append(f"[{n}] {crumb} | {page_label(c)} (report p.{c['printed_start']})\n{c['text']}")
            sources.append({"id": i, "page": c["page_start"], "label": page_label(c), "breadcrumb": crumb,
                            "primary": i in chosen, "text": c["text"][:320]})
        return "\n\n".join(blocks), sources

    # ---------- main entry ----------
    def answer(self, question, history=None, chapter=None, rerank=False):
        standalone = self.rewrite(question, history or [])
        hits = self.retrieve(standalone, chapter=chapter or None, rerank=rerank)
        context, sources = self.build_context(hits)
        prompt = f"Context excerpts:\n{context}\n\nQuestion: {standalone}\nAnswer:"
        text = self._generate(prompt, system=SYSTEM_PROMPT)
        return {"answer": text, "query_used": standalone, "sources": sources,
                "retrieved_pages": sorted({h["page_start"] for h in hits})}
