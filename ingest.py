"""Section-aware ingestion for the World Bank Global Economic Prospects report.

Pipeline
  1. Text per page with pypdf (keeps correct two-column reading order) + cleaning
  2. Document structure: PDF bookmarks (chapters, boxes, regions) + font-based
     headings found with pdfplumber (section / subsection / run-in topic)
  3. Structure-aware chunking: chunks never cross a section boundary, each chunk
     carries a breadcrumb + metadata (chapter, region, section, pages, figures)
  4. Gemini embeddings of "breadcrumb + text" (throttled, resumable)

Run once:  python ingest.py            (add --no-embed to only inspect chunking)
"""
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict

import numpy as np
import pdfplumber
from pypdf import PdfReader

import config

# ----------------------------------------------------------------------------
# 1. Text extraction + cleaning
# ----------------------------------------------------------------------------
def _despace(s):
    return re.sub(r"\s+", "", s).upper()


def clean_page_text(raw):
    t = unicodedata.normalize("NFKC", raw)            # ligatures: fi, ff ...
    t = t.replace("7'_", "T")                         # drop-cap glyph used for "T"
    t = re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", t)  # re-join hyphenated words
    lines = []
    for line in t.split("\n"):
        s = line.strip()
        d = _despace(s)
        if not s:
            continue
        if "GLOBALECONOMICPROSPECTS" in d and "JUNE2026" in d and len(s) < 110:
            continue                                   # running header
        if re.fullmatch(r"[\d\s%.,\-()–]+", s):
            continue                                   # chart axis numbers / stray numbers
        if len(s) <= 2:
            continue
        lines.append(s)
    return " ".join(lines)


# ----------------------------------------------------------------------------
# 2. Structure
# ----------------------------------------------------------------------------
def _clean_title(t):
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"(^|\s)--\s*", r"\1", t).strip()


CHAPTER_RE = re.compile(r"^(Ch\. \d|Executive Summary|Statistical Annex)")
REGION_KEYS = ("East Asia", "Europe and Central Asia", "Latin America", "Middle East", "South Asia", "Sub-Saharan")


def load_outline(reader):
    """Classify bookmarks by title (the PDF lists boxes at the same level as chapters)."""
    flat = []

    def walk(items):
        for it in items:
            if isinstance(it, list):
                walk(it)
            else:
                flat.append((_clean_title(it.title), reader.get_destination_page_number(it) + 1))

    walk(reader.outline)
    chapters, regions = [(1, "Front matter")], []
    for title, page in flat:
        if CHAPTER_RE.match(title):
            chapters.append((page, title))
        elif any(title.startswith(k) for k in REGION_KEYS):
            regions.append((page, title))
    chapters.sort()
    regions.sort()
    return chapters, regions


def _last_before(entries, page):
    cur = None
    for start, name in entries:
        if start <= page:
            cur = (start, name)
    return cur


def classify_heading(fontname, size, text):
    fn = fontname.split("+")[-1]
    if "Helvetica" in fn and "Bold" in fn and abs(size - 10.0) < 0.3:
        if re.match(r"(FIGURE|TABLE)\b", text):
            return "caption"
        if re.match(r"BOX\b", text):
            return "box"
        return None
    if 14 <= size <= 20 and ("Bold" in fn or "Medium" in fn):
        return "h1"
    if "Garamond" in fn and "BoldItalic" in fn.replace("-", "").replace("Bold Italic", "BoldItalic"):
        return "h2"
    if "Garamond" in fn and "Bold" in fn and "Italic" not in fn:
        return "h3"
    return None


def heading_runs(page):
    """Return [(level, text)] for headings on a pdfplumber page, top to bottom."""
    by_line = defaultdict(list)
    all_chars = page.chars
    for ch in all_chars:
        by_line[(round(ch["top"]), ch["fontname"], round(ch["size"], 1))].append(ch)
    runs = []
    for (top, fn, sz), cs in by_line.items():
        cs.sort(key=lambda c: c["x0"])
        text, prev = "", None
        for c in cs:
            if prev is not None and c["x0"] - prev["x1"] > c["size"] * 0.25:
                text += " "
            text += c["text"]
            prev = c
        text = text.strip()
        level = classify_heading(fn, sz, text)
        if not level or not (4 <= len(text) <= 160) or top < 40 or top > page.height - 35:
            continue
        x0 = cs[0]["x0"]
        # skip emphasis in the middle of a line (some other text sits right before it)
        if any(o["fontname"] != cs[0]["fontname"] and abs(o["top"] - top) <= 2
               and o["x1"] <= x0 and x0 - o["x1"] < 6 for o in all_chars):
            continue
        runs.append({"level": level, "text": text, "top": top, "x0": x0, "fn": fn, "size": sz})
    runs.sort(key=lambda r: (r["top"], r["x0"]))
    # merge consecutive lines of the same multi-line heading
    merged = []
    for r in runs:
        if merged:
            m = merged[-1]
            if m["level"] == r["level"] == "h1" or (m["level"] == r["level"] and m["fn"] == r["fn"]):
                if abs(m["x0"] - r["x0"]) < 4 and 0 < r["top"] - m["last_top"] <= r["size"] * 1.7 and r["level"] != "caption":
                    m["text"] += " " + r["text"]
                    m["last_top"] = r["top"]
                    continue
        r["last_top"] = r["top"]
        merged.append(r)
    return [(m["level"], m["text"]) for m in merged]


def _norm_with_map(s):
    chars, idx = [], []
    for i, ch in enumerate(s):
        if ch.isalnum():
            chars.append(ch.lower())
            idx.append(i)
    return "".join(chars), idx


def locate(flat_norm, idx_map, heading, start=0):
    h, _ = _norm_with_map(heading)
    if not h:
        return None
    p = flat_norm.find(h, start)
    return idx_map[p] if p >= 0 else None


BOX_RE = re.compile(r"^BOX\s+(\d+\.\d+)\s+(.*?)\s*(\(continued\s*\))?$", re.I)


def _split_long(text, limit=900):
    """Break run-on 'sentences' (table rows, lists) into <= limit-char pieces at word boundaries."""
    if len(text) <= limit:
        return [text] if text else []
    out, cur = [], ""
    for w in text.split(" "):
        if len(cur) + len(w) + 1 > limit and cur:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out


def build_units(config_pdf=config.PDF_PATH):
    """Walk the document and return text 'units' (sentences) with structural context."""
    reader = PdfReader(str(config_pdf))
    chapters, regions = load_outline(reader)
    units = []
    state = {"section": None, "subsection": None, "topic": None}
    cur_chapter = None
    stats = defaultdict(int)

    with pdfplumber.open(str(config_pdf)) as pdf:
        for pno, (rpage, ppage) in enumerate(zip(reader.pages, pdf.pages), start=1):
            text = clean_page_text(rpage.extract_text() or "")
            if len(text) < 40:
                continue
            ch = _last_before(chapters, pno)
            chapter = ch[1] if ch else "Front matter"
            if chapter != cur_chapter:
                state = {"section": None, "subsection": None, "topic": None}
                cur_chapter = chapter
            rg = _last_before(regions, pno)
            region = rg[1] if (rg and ch and rg[0] >= ch[0] and chapter.startswith("Ch. 2")) else None
            box = None

            try:
                heads = heading_runs(ppage)
            except Exception:
                heads = []
            norm, imap = _norm_with_map(text)
            markers, search_from = [], 0
            for level, htext in heads:
                pos = locate(norm, imap, htext if level != "box" else htext, 0)
                markers.append((pos if pos is not None else 0, level, htext))
            markers.sort(key=lambda m: m[0])

            segments, cursor = [], 0
            ctx = lambda: (chapter, region, state["section"], state["subsection"], state["topic"], box)
            for pos, level, htext in markers:
                if pos > cursor:
                    segments.append((ctx(), text[cursor:pos]))
                    cursor = pos
                if level == "h1":
                    state.update(section=htext, subsection=None, topic=None); stats["h1"] += 1
                elif level == "h2":
                    state.update(subsection=htext, topic=None); stats["h2"] += 1
                elif level == "h3":
                    state.update(topic=htext); stats["h3"] += 1
                elif level == "box":
                    m = BOX_RE.match(htext)
                    title = re.sub(r"\(\s*(continued)?\s*\)?\s*$", "", m.group(2)).strip() if m else htext
                    box = f"BOX {m.group(1)} {title}" if m else htext
                    stats["box"] += 1
            segments.append((ctx(), text[cursor:]))

            for c, seg in segments:
                for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[“\"])", seg.strip()):
                    for piece in _split_long(sent.strip()):
                        units.append({"ctx": c, "page": pno, "text": piece})
    print("headings detected:", dict(stats))
    return units


# ----------------------------------------------------------------------------
# 3. Structure-aware chunking
# ----------------------------------------------------------------------------
REF_RE = re.compile(r"\b(FIGURE|TABLE|BOX|Figure|Table|Box|figure|table|box)\s+(B?\d+(?:\.\d+)*)")


def breadcrumb(ctx):
    chapter, region, section, subsection, topic, box = ctx
    parts = ["World Bank Global Economic Prospects, June 2026", chapter]
    for p in (region, section, subsection, topic, box):
        if p and p not in parts:
            parts.append(p)
    return " > ".join(parts)


def build_chunks(units):
    chunks, group, group_ctx = [], [], None

    def flush():
        nonlocal group
        if not group:
            return
        sents = list(group)
        buf, pages, size = [], [], 0
        pieces = []
        for u in sents:
            if size + len(u["text"]) > config.CHUNK_SIZE and buf:
                pieces.append(buf)
                carry, n = [], 0
                for prev in reversed(buf):               # overlap
                    n += len(prev["text"])
                    if n > config.CHUNK_OVERLAP:
                        break
                    carry.insert(0, prev)
                buf, size = list(carry), sum(len(c["text"]) for c in carry)
            buf.append(u)
            size += len(u["text"]) + 1
        if buf:
            pieces.append(buf)
        # merge a tiny trailing piece into the previous one
        if len(pieces) > 1 and sum(len(u["text"]) for u in pieces[-1]) < 200:
            tail = pieces.pop()
            seen = {id(u) for u in pieces[-1]}
            pieces[-1].extend(u for u in tail if id(u) not in seen)
        for piece in pieces:
            text = " ".join(u["text"] for u in piece)
            if len(text) < 60:
                continue
            ctx = piece[0]["ctx"]
            refs = sorted({f"{m.group(1).upper()} {m.group(2)}" for m in REF_RE.finditer(text)})
            caps = sorted({f"{m.group(1).upper()} {m.group(2)}" for m in re.finditer(r"\b(FIGURE|TABLE|BOX)\s+(B?\d+(?:\.\d+)*)", text)})
            chapter, region, section, subsection, topic, box = ctx
            chunks.append({
                "id": len(chunks), "text": text, "breadcrumb": breadcrumb(ctx),
                "chapter": chapter, "region": region, "section": section,
                "subsection": subsection, "topic": topic, "box": box,
                "page_start": piece[0]["page"], "page_end": piece[-1]["page"],
                "printed_start": max(piece[0]["page"] - config.PAGE_OFFSET, 0),
                "captions": caps, "refs": refs,
            })
        group = []

    for u in units:
        if group_ctx is not None and u["ctx"] != group_ctx:
            flush()
        group_ctx = u["ctx"]
        group.append(u)
    flush()
    return chunks


# ----------------------------------------------------------------------------
# 4. Embeddings (throttled + resumable)
# ----------------------------------------------------------------------------
def _retry_delay(err, default):
    m = re.search(r"retry in ([\d.]+)s", str(err))
    return float(m.group(1)) + 2 if m else default


def embed_texts(client, texts, task_type):
    from google.genai import types
    config.INDEX_DIR.mkdir(exist_ok=True)
    cache = config.INDEX_DIR / "partial_embeddings.json"
    vecs = json.loads(cache.read_text()) if cache.exists() else []
    if vecs:
        print(f"  resuming from {len(vecs)}/{len(texts)}")
    for i in range(len(vecs), len(texts), config.EMBED_BATCH):
        batch = texts[i:i + config.EMBED_BATCH]
        for attempt in range(12):
            try:
                res = client.models.embed_content(
                    model=config.EMBED_MODEL, contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=config.EMBED_DIM),
                )
                vecs.extend(e.values for e in res.embeddings)
                break
            except Exception as e:
                msg = str(e)
                if attempt == 0:
                    print("\n  API error (full message):\n  " + msg + "\n")
                daily = "perday" in msg.lower().replace(" ", "").replace("_", "") or "per day" in msg.lower()
                no_hint = "retry in" not in msg
                if daily or (no_hint and attempt >= 2 and "429" in msg):
                    print(f"  Stopping: this looks like a DAILY quota limit. {len(vecs)}/{len(texts)} chunks are saved.")
                    print("  Re-run `python ingest.py` after the quota resets (or enable billing / use another key) to resume.")
                    sys.exit(1)
                wait = _retry_delay(e, min(2 ** attempt * 5, 120))
                print(f"  rate limit/error; waiting {wait:.0f}s")
                time.sleep(wait)
        else:
            raise RuntimeError("Embedding failed after retries (re-run to resume)")
        cache.write_text(json.dumps(vecs))
        print(f"  embedded {len(vecs)}/{len(texts)}")
        if len(vecs) < len(texts):
            time.sleep(config.EMBED_PAUSE)
    arr = np.array(vecs, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    cache.unlink(missing_ok=True)
    return arr


def main():
    embed = "--no-embed" not in sys.argv
    print("Reading structure and text (this takes ~1-2 minutes)...")
    units = build_units()
    chunks = build_chunks(units)
    sizes = [len(c["text"]) for c in chunks]
    print(f"{len(units)} sentences -> {len(chunks)} chunks "
          f"(avg {int(sum(sizes)/len(sizes))} chars, max {max(sizes)})")
    config.INDEX_DIR.mkdir(exist_ok=True)
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    if not embed:
        print("Skipped embeddings (--no-embed). Chunks saved to index/chunks.json")
        return
    if not config.API_KEY:
        sys.exit("GEMINI_API_KEY missing. Create a .env file (see .env.example).")
    from google import genai
    client = genai.Client(api_key=config.API_KEY)
    print("Embedding chunks (throttled for the free tier; safe to re-run to resume)...")
    vectors = embed_texts(client, [c["breadcrumb"] + "\n\n" + c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    np.save(config.INDEX_DIR / "embeddings.npy", vectors)
    print(f"Saved index to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
