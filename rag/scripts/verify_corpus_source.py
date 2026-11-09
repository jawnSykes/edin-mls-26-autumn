"""Verify that every course corpus record comes from BeIR/fiqa.

Also tries to recover how the r_/q_ IDs were derived, and whether the course
qrels are FiQA qrels restricted to the 24k-record corpus (so they can be reused
when upsampling).

Run:
  uv run --no-project --with huggingface_hub --with pyarrow \
      python rag/scripts/verify_corpus_source.py [--corpus PATH] [--zip PATH]
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import html
import io
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data/rag_corpus/corpus.jsonl"
DEFAULT_ZIP = ROOT.parent / "rag-assets.zip"
ZIP_PREFIX = "data/rag_corpus/"
FIQA_REPO = "BeIR/fiqa"
FIQA_QRELS_REPO = "BeIR/fiqa-qrels"
FIQA_SPLITS = ("train", "dev", "test")
ID_RE = re.compile(r"^([rq])_([0-9a-f]{16})$")


# ---------------------------------------------------------------- course data
class CourseSource:
    """Reads course files from an installed directory or straight from the zip."""

    def __init__(self, corpus: Path | None, zip_path: Path):
        if corpus is None and DEFAULT_CORPUS.is_file():
            corpus = DEFAULT_CORPUS
        self.dir = corpus.parent if corpus else None
        self.corpus_name = corpus.name if corpus else "corpus.jsonl"
        self.zip = None
        if self.dir is None:
            if not zip_path.is_file():
                sys.exit(f"No corpus at {DEFAULT_CORPUS} and no zip at {zip_path}")
            self.zip = zipfile.ZipFile(zip_path)
        self.where = str(self.dir) if self.dir else f"{zip_path}:{ZIP_PREFIX}"

    def read_text(self, name: str) -> str | None:
        if self.dir is not None:
            p = self.dir / name
            return p.read_text(encoding="utf-8-sig") if p.is_file() else None
        try:
            return self.zip.read(ZIP_PREFIX + name).decode("utf-8-sig")
        except KeyError:
            return None

    def corpus(self) -> list[dict]:
        rows = [json.loads(l) for l in self.read_text(self.corpus_name).splitlines() if l.strip()]
        ids = [r["record_id"] for r in rows]
        if len(set(ids)) != len(ids):
            sys.exit("duplicate record_id in course corpus")
        return rows

    def qrels(self) -> dict[str, dict[str, set[str]]]:
        """split -> query_id -> set(record_id) (score > 0), mirrors CourseData.load_qrels."""
        out = {}
        for split in ("practice", "dev"):
            text = self.read_text(f"qrels_{split}.tsv")
            if text is None:
                continue
            q: dict[str, set[str]] = defaultdict(set)
            for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
                if float(row.get("score", 1)) > 0:
                    q[row["query_id"]].add(row["record_id"])
            out[split] = dict(q)
        return out


# ---------------------------------------------------------------- FiQA data
def load_fiqa():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    def parquet(path_in_repo):
        p = hf_hub_download(FIQA_REPO, path_in_repo, repo_type="dataset")
        t = pq.read_table(p).to_pydict()
        return t

    c = parquet("corpus/corpus-00000-of-00001.parquet")
    corpus = {str(i): (ti or "", tx or "") for i, ti, tx in zip(c["_id"], c["title"], c["text"])}
    q = parquet("queries/queries-00000-of-00001.parquet")
    queries = {str(i): tx or "" for i, tx in zip(q["_id"], q["text"])}

    qrels: dict[str, dict[str, set[str]]] = {}
    for split in FIQA_SPLITS:
        p = hf_hub_download(FIQA_QRELS_REPO, f"{split}.tsv", repo_type="dataset")
        d: dict[str, set[str]] = defaultdict(set)
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if float(row["score"]) > 0:
                    d[str(row["query-id"])].add(str(row["corpus-id"]))
        qrels[split] = dict(d)
    return corpus, queries, qrels


# ---------------------------------------------------------------- matching
def course_clean(s: str) -> str:
    """Candidate cleaning recipe the course applied to FiQA text."""
    return " ".join(html.unescape(s).split())


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", html.unescape(s)).replace('""', '"')
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


def shingles(s: str, n: int = 5) -> set[str]:
    w = s.lower().split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def match_records(course, fiqa_corpus, threshold):
    exact, clean, norm = defaultdict(list), defaultdict(list), defaultdict(list)
    for fid, (_, text) in fiqa_corpus.items():
        exact[text].append(fid)
        clean[course_clean(text)].append(fid)
        norm[normalise(text)].append(fid)

    results = {}  # record_id -> (fiqa_ids, tier, score)
    leftovers = []
    for r in course:
        rid, text = r["record_id"], r["text"]
        if text in exact:
            results[rid] = (exact[text], "T1_exact", 1.0)
        elif text in clean:
            results[rid] = (clean[text], "T2a_clean_recipe", 1.0)
        elif (n := normalise(text)) in norm:
            results[rid] = (norm[n], "T2b_normalised", 1.0)
        else:
            leftovers.append(r)

    unmatched = []
    if leftovers:
        print(f"  fuzzy-matching {len(leftovers)} leftover records ...")
        index = defaultdict(set)
        normed = {fid: normalise(t) for fid, (_, t) in fiqa_corpus.items()}
        for fid, t in normed.items():
            for sh in shingles(t):
                index[sh].add(fid)
        for r in leftovers:
            t = normalise(r["text"])
            votes = Counter()
            for sh in shingles(t):
                for fid in index.get(sh, ()):
                    votes[fid] += 1
            best, best_score = None, 0.0
            for fid, _ in votes.most_common(5):
                s = difflib.SequenceMatcher(None, t, normed[fid], autojunk=False).ratio()
                if s > best_score:
                    best, best_score = fid, s
            if best is not None and best_score >= threshold:
                results[r["record_id"]] = ([best], "T3_fuzzy", round(best_score, 4))
            else:
                unmatched.append({**r, "best_fiqa_id": best, "best_score": round(best_score, 4)})
    return results, unmatched


# ---------------------------------------------------------------- ID derivation
HASHES = ("md5", "sha1", "sha224", "sha256", "sha384", "sha512", "blake2b", "blake2s", "sha3_256")
ID_PREFIXES = ("", "fiqa:", "fiqa-", "fiqa_", "fiqa/", "beir/fiqa:", "BeIR/fiqa:", "doc:", "doc_", "d_",
               "r:", "r_", "record:", "corpus:", "q:", "q_", "query:")


def candidate_inputs(kind, src_id, text, title=""):
    for p in ID_PREFIXES:
        yield f"id[{p!r}+_id]", p + src_id
    yield "text", text
    yield "normalised_text", normalise(text)
    yield "clean_text", course_clean(text)
    yield "_id+clean_text", src_id + course_clean(text)
    yield "_id\\tclean_text", src_id + "\t" + course_clean(text)
    yield "_id+text", src_id + text
    yield "_id\\ttext", src_id + "\t" + text
    yield "_id\\ntext", src_id + "\n" + text
    yield "_id:text", src_id + ":" + text
    if title:
        yield "title+text", title + text
    yield "json[_id,text]", json.dumps({"_id": src_id, "text": text})


def find_id_rule(pairs, kind):
    """pairs: list of (course_id, [(src_id, text, title), ...]). Returns rule string or None."""
    pairs = [(cid, cands) for cid, cands in pairs if ID_RE.match(cid)]
    if not pairs:
        return None, 0
    fmt = [f"{h}/{label}/{side}" for h in HASHES for label, _ in candidate_inputs(kind, "0", "x")
           for side in ("prefix", "suffix")]
    hits = Counter()
    probe = pairs[:300]
    for cid, cands in probe:
        target = cid[2:]
        found = set()
        for src_id, text, title in cands:
            for label, s in candidate_inputs(kind, src_id, text, title):
                b = s.encode("utf-8")
                for h in HASHES:
                    hx = hashlib.new(h, b).hexdigest()
                    if hx[:16] == target:
                        found.add(f"{h}/{label}/prefix")
                    if hx[-16:] == target:
                        found.add(f"{h}/{label}/suffix")
        hits.update(found)
    if not hits:
        return None, len(probe)
    rule, n = hits.most_common(1)[0]
    # Confirm on all pairs.
    ok = sum(1 for cid, cands in pairs if any(apply_rule(rule, kind, *c) == cid for c in cands))
    return f"{rule} ({ok}/{len(pairs)} records)", ok


def apply_rule(rule, kind, src_id, text, title=""):
    h, label, side = rule.split("/", 2) if rule.count("/") == 2 else (None, None, None)
    if h is None:  # labels may themselves contain '/', e.g. id['fiqa/'+_id]
        h, rest = rule.split("/", 1)
        label, side = rest.rsplit("/", 1)
    for lab, s in candidate_inputs(kind, src_id, text, title):
        if lab == label:
            hx = hashlib.new(h, s.encode("utf-8")).hexdigest()
            return f"{kind}_" + (hx[:16] if side == "prefix" else hx[-16:])
    return None


def proposed_namespace(fiqa_ids, existing_ids):
    """Fallback ID scheme for new records; check it collides with nothing."""
    new = {"r_" + hashlib.sha256(f"fiqa:{i}".encode()).hexdigest()[:16] for i in fiqa_ids}
    return {
        "rule": "r_ + sha256('fiqa:' + fiqa_id)[:16]",
        "unique": len(new) == len(set(fiqa_ids)),
        "collisions_with_existing": len(new & set(existing_ids)),
    }


# ---------------------------------------------------------------- qrels reuse
def match_queries(course_qrels, rec2fiqa, fiqa_qrels, fiqa_queries, corpus_fiqa_ids):
    doc2q = defaultdict(set)  # fiqa doc -> {(split, qid)}
    for split, qs in fiqa_qrels.items():
        for qid, docs in qs.items():
            for d in docs:
                doc2q[d].add((split, qid))

    rows = []
    for csplit, qs in course_qrels.items():
        for cq, recs in sorted(qs.items()):
            S = {rec2fiqa[r] for r in recs if r in rec2fiqa}
            cands = set().union(*(doc2q[d] for d in S)) if S else set()
            kinds = defaultdict(list)
            for fs, fq in cands:
                F = fiqa_qrels[fs][fq]
                if F == S:
                    kinds["exact"].append((fs, fq))
                elif (F & corpus_fiqa_ids) == S:
                    kinds["restricted_exact"].append((fs, fq))
                elif S < F:
                    kinds["subset"].append((fs, fq))
            for kind in ("exact", "restricted_exact", "subset"):
                if kinds[kind]:
                    m = sorted(kinds[kind])
                    kind_out = kind if len(m) == 1 else kind + "_ambiguous"
                    break
            else:
                m, kind_out = [], "none"
            rows.append({
                "query_id": cq, "split": csplit,
                "fiqa_query_id": ";".join(q for _, q in m),
                "fiqa_split": ";".join(sorted({s for s, _ in m})),
                "match_kind": kind_out,
                "n_relevant": len(recs),
                "fiqa_query_text": fiqa_queries.get(m[0][1], "") if len(m) == 1 else "",
            })
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, help="course corpus.jsonl (default: installed, else zip)")
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--out-dir", type=Path, default=ROOT / ".cache/provenance")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.9)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    src = CourseSource(args.corpus, args.zip)
    print(f"Course data: {src.where}")
    course = src.corpus()
    course_qrels = src.qrels()
    print(f"  {len(course)} records, " + ", ".join(f"{k}: {len(v)} queries" for k, v in course_qrels.items()))

    print(f"Fetching {FIQA_REPO} + {FIQA_QRELS_REPO} ...")
    fiqa_corpus, fiqa_queries, fiqa_qrels = load_fiqa()
    print(f"  {len(fiqa_corpus)} docs, {len(fiqa_queries)} queries, "
          + ", ".join(f"{s}: {len(q)} q" for s, q in fiqa_qrels.items()))

    # 1. Record membership
    print("Matching records ...")
    results, unmatched = match_records(course, fiqa_corpus, args.fuzzy_threshold)
    tiers = Counter(t for _, t, _ in results.values())
    ambiguous = sum(1 for ids, _, _ in results.values() if len(ids) > 1)
    all_from_fiqa = not unmatched

    # 2. ID derivation (records)
    text_of = {r["record_id"]: r["text"] for r in course}
    pairs = [(rid, [(f, fiqa_corpus[f][1], fiqa_corpus[f][0]) for f in ids])
             for rid, (ids, _, _) in results.items()]
    rec_rule, _ = find_id_rule(pairs, "r")

    # Pick one FiQA id per record; prefer the one the ID rule reproduces.
    rec2fiqa = {}
    for rid, (ids, _, _) in results.items():
        pick = ids[0]
        if rec_rule and len(ids) > 1:
            rule = rec_rule.split(" (")[0]
            pick = next((f for f in ids if apply_rule(rule, "r", f, fiqa_corpus[f][1], fiqa_corpus[f][0]) == rid), pick)
        rec2fiqa[rid] = pick
    corpus_fiqa_ids = set(rec2fiqa.values())
    recipe_ok = sum(1 for rid, f in rec2fiqa.items() if course_clean(fiqa_corpus[f][1]) == text_of[rid])

    # 3. Qrels reuse
    print("Matching queries via qrels ...")
    qrows = match_queries(course_qrels, rec2fiqa, fiqa_qrels, fiqa_queries, corpus_fiqa_ids)
    qkinds = Counter(r["match_kind"] for r in qrows)
    qsplits = Counter((r["split"], r["fiqa_split"]) for r in qrows if r["fiqa_split"])
    qpairs = [(r["query_id"], [(r["fiqa_query_id"], fiqa_queries.get(r["fiqa_query_id"], ""), "")])
              for r in qrows if r["fiqa_query_id"] and ";" not in r["fiqa_query_id"]]
    q_rule, _ = find_id_rule(qpairs, "q")

    # 4. Selection pattern
    judged = {s: set().union(*q.values()) for s, q in fiqa_qrels.items()}
    selection = {s: {"fiqa_judged_docs": len(d), "in_course_corpus": len(d & corpus_fiqa_ids)}
                 for s, d in judged.items()}
    any_judged = set().union(*judged.values())
    selection["unjudged_fill"] = len(corpus_fiqa_ids - any_judged)
    selection["fiqa_docs_not_in_corpus"] = len(set(fiqa_corpus) - corpus_fiqa_ids)

    namespace = None if rec_rule else proposed_namespace(list(fiqa_corpus), list(text_of))

    # ---- outputs
    with (args.out_dir / "record_map.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "fiqa_id", "tier", "score", "all_fiqa_ids"])
        for r in course:
            rid = r["record_id"]
            if rid in results:
                ids, tier, score = results[rid]
                w.writerow([rid, rec2fiqa[rid], tier, score, ";".join(ids)])
    with (args.out_dir / "unmatched.jsonl").open("w") as f:
        for u in unmatched:
            f.write(json.dumps(u) + "\n")
    with (args.out_dir / "query_map.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(qrows[0]) if qrows else ["query_id"])
        w.writeheader()
        w.writerows(qrows)

    summary = {
        "course_source": src.where,
        "n_records": len(course),
        "tiers": dict(tiers),
        "unmatched": len(unmatched),
        "records_with_duplicate_fiqa_text": ambiguous,
        "clean_recipe": "' '.join(html.unescape(fiqa_text).split())",
        "clean_recipe_exact_matches": recipe_ok,
        "all_records_from_fiqa": all_from_fiqa,
        "record_id_rule": rec_rule,
        "query_id_rule": q_rule,
        "proposed_namespace_if_no_rule": namespace,
        "query_match_kinds": dict(qkinds),
        "query_split_origin": {f"{a}<-{b}": n for (a, b), n in qsplits.items()},
        "selection": selection,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ SUMMARY ================")
    print(f"Records: {len(course)}   " + "  ".join(f"{k}={v}" for k, v in sorted(tiers.items()))
          + f"  unmatched={len(unmatched)}")
    print(f"Records whose text appears under >1 FiQA id: {ambiguous}")
    print(f"Cleaning recipe ' '.join(html.unescape(text).split()) reproduces {recipe_ok}/{len(course)} texts exactly")
    print(f"ALL {len(course):,} records from {FIQA_REPO}: {'YES' if all_from_fiqa else 'NO'}")
    print(f"record_id rule: {rec_rule or 'not found (likely salted/keyed hash)'}")
    print(f"query_id rule:  {q_rule or 'not found (likely salted/keyed hash)'}")
    if namespace:
        print(f"  fallback for new records: {namespace}")
    print(f"Query matches: {dict(qkinds)}")
    print(f"Query split origin (course<-fiqa): {summary['query_split_origin']}")
    print(f"Selection: {json.dumps(selection)}")
    print(f"Outputs in {args.out_dir}")
    return 0 if all_from_fiqa else 1


if __name__ == "__main__":
    sys.exit(main())
