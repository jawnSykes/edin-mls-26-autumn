"""Build the full-FiQA corpus + query set (course 24k + remaining BeIR/fiqa docs).

See rag/docs/corpus-provenance.md for why this is valid. Subcommands:

  fetch    (head node, needs internet)  download BeIR/fiqa + qrels into the HF cache
  prepare  (CPU)   mint IDs, dedupe, migrate qrels, write text files to --out-dir
  encode   (GPU)   sanity-gate enc_a, copy the original 24k vectors, encode new docs + queries
  package  (CPU)   write a checksummed, uncompressed zip to ~/mls-rag-assets/fiqa_full/
  install  (CPU)   extract a package zip into rag/data/fiqa_full/ and verify checksums

Typical cluster use (repo root):  see rag/scripts/build_fiqa_full.sbatch
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent  # rag/
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from verify_corpus_source import (  # noqa: E402
    DEFAULT_ZIP, FIQA_QRELS_REPO, FIQA_REPO, FIQA_SPLITS, ZIP_PREFIX,
    CourseSource, course_clean, load_fiqa, match_records,
)

DEFAULT_OUT = ROOT / "data/fiqa_full"
DEFAULT_COURSE_DIR = ROOT / "data/rag_corpus"
DEFAULT_MODEL = ROOT / "models/enc_a"
DEFAULT_PKG_DIR = Path.home() / "mls-rag-assets/fiqa_full"
PKG_NAME = "fiqa-full-enc_a"
PKG_ROOT = "fiqa_full"
SUMS = "SHA256SUMS"

RECORD_RULE = "r_ + sha256('fiqa:' + fiqa_id)[:16]"
QUERY_RULE = "q_ + sha256('fiqa-query:' + fiqa_query_id)[:16]"
CLEAN_RECIPE = "' '.join(html.unescape(text).split())"
MAX_SEQ_LENGTH = 512


# ================================================================ pure helpers
def mint_record_id(fiqa_id: str) -> str:
    return "r_" + hashlib.sha256(f"fiqa:{fiqa_id}".encode()).hexdigest()[:16]


def mint_query_id(fiqa_qid: str) -> str:
    return "q_" + hashlib.sha256(f"fiqa-query:{fiqa_qid}".encode()).hexdigest()[:16]


def fiqa_sort_key(s: str):
    return (0, int(s), "") if s.isdigit() else (1, 0, s)


def build_records(course, rec_matches, fiqa_corpus):
    """Map every FiQA doc id to a record_id, minting ids for docs not in the course corpus.

    course:      [{record_id, text}] in course order (text already cleaned by the course)
    rec_matches: record_id -> [fiqa ids whose (cleaned) text equals the record text]
    fiqa_corpus: fiqa_id -> (title, raw_text)

    Returns (records, fiqa2rid, stats). records is a list of dicts with keys
    record_id, fiqa_id (primary), source ('course'|'fiqa_ext'), aliases (other fiqa ids), text.
    """
    fiqa2rid: dict[str, str] = {}
    text2rid: dict[str, str] = {}
    records: list[dict] = []
    by_rid: dict[str, dict] = {}

    for r in course:
        rid, text = r["record_id"], r["text"]
        ids = rec_matches.get(rid)
        if not ids:
            raise ValueError(f"course record {rid} has no FiQA match")
        ids = sorted(set(ids), key=fiqa_sort_key)
        for f in ids:
            if f in fiqa2rid and fiqa2rid[f] != rid:
                raise ValueError(f"FiQA id {f} matches two course records: {fiqa2rid[f]}, {rid}")
            fiqa2rid[f] = rid
        key = course_clean(text)
        if key in text2rid:
            raise ValueError(f"course records {text2rid[key]} and {rid} have identical text")
        text2rid[key] = rid
        rec = {"record_id": rid, "fiqa_id": ids[0], "source": "course", "aliases": ids[1:], "text": text}
        records.append(rec)
        by_rid[rid] = rec

    existing = set(by_rid)
    empty, dup_course, dup_new = [], 0, 0
    for f in sorted(fiqa_corpus, key=fiqa_sort_key):
        if f in fiqa2rid:
            continue
        text = course_clean(fiqa_corpus[f][1] or "")
        if not text:
            empty.append(f)
            continue
        if text in text2rid:  # duplicate text: alias onto the earlier record
            rid = text2rid[text]
            fiqa2rid[f] = rid
            by_rid[rid]["aliases"].append(f)
            if by_rid[rid]["source"] == "course":
                dup_course += 1
            else:
                dup_new += 1
            continue
        rid = mint_record_id(f)
        if rid in existing:
            raise ValueError(f"minted id {rid} for FiQA {f} collides with an existing id")
        existing.add(rid)
        fiqa2rid[f] = rid
        text2rid[text] = rid
        rec = {"record_id": rid, "fiqa_id": f, "source": "fiqa_ext", "aliases": [], "text": text}
        records.append(rec)
        by_rid[rid] = rec

    stats = {
        "course_records": len(course),
        "new_records": sum(1 for r in records if r["source"] == "fiqa_ext"),
        "total_records": len(records),
        "fiqa_docs": len(fiqa_corpus),
        "dropped_empty": len(empty),
        "dropped_empty_ids": empty[:50],
        "aliased_dup_of_course": dup_course,
        "aliased_dup_of_new": dup_new,
        "course_records_with_aliases": sum(1 for r in records if r["source"] == "course" and r["aliases"]),
    }
    return records, fiqa2rid, stats


def match_course_queries(course_qrels, fiqa_qrels, fiqa2rid, original_ids):
    """Find the FiQA query behind each course query, comparing at record level.

    A FiQA query's judged docs are mapped through fiqa2rid (so aliases resolve) and compared
    with the course qrels: 'exact' if equal, 'restricted_exact' if equal once limited to the
    original corpus. Returns rows shaped like verify_corpus_source.match_queries.
    """
    rid2fq = defaultdict(set)  # record_id -> {fiqa qid}
    fq_rids: dict[str, set[str]] = defaultdict(set)
    for split in FIQA_SPLITS:
        for fq, docs in fiqa_qrels.get(split, {}).items():
            for d in docs:
                if d in fiqa2rid:
                    fq_rids[fq].add(fiqa2rid[d])
                    rid2fq[fiqa2rid[d]].add(fq)
    rows = []
    for csplit, qs in course_qrels.items():
        for cq, recs in sorted(qs.items()):
            recs = set(recs)
            cands = set().union(*(rid2fq[r] for r in recs)) if recs else set()
            exact = sorted(fq for fq in cands if fq_rids[fq] == recs)
            restricted = sorted(fq for fq in cands if (fq_rids[fq] & original_ids) == recs)
            if exact:
                m, kind = exact, "exact"
            elif restricted:
                m, kind = restricted, "restricted_exact"
            else:
                m, kind = [], "none"
            if len(m) > 1:
                kind += "_ambiguous"
            rows.append({"query_id": cq, "split": csplit, "fiqa_query_id": ";".join(m), "match_kind": kind})
    return rows


def fiqa_query_splits(fiqa_qrels) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for split in FIQA_SPLITS:
        for q in fiqa_qrels.get(split, {}):
            out[q].append(split)
    return out


def build_queries(course_qrows, fiqa_queries, fiqa_qrels):
    """Map every FiQA query id to a query_id (course ids kept, others minted).

    course_qrows: rows from verify_corpus_source.match_queries (query_id, split, fiqa_query_id, match_kind)
    Returns (queries, fq2qid, stats). queries: list of dicts
    query_id, fiqa_query_id, fiqa_split, course_split, text.
    """
    fq2qid: dict[str, str] = {}
    course_split_of: dict[str, str] = {}
    for row in course_qrows:
        cq, fq, kind = row["query_id"], row["fiqa_query_id"], row["match_kind"]
        if kind not in ("exact", "restricted_exact") or not fq or ";" in fq:
            raise ValueError(f"course query {cq} has no unique FiQA match ({kind}: {fq!r})")
        if fq in fq2qid:
            raise ValueError(f"FiQA query {fq} matches two course queries: {fq2qid[fq]}, {cq}")
        if cq in course_split_of:
            raise ValueError(f"course query {cq} appears in two course splits")
        fq2qid[fq] = cq
        course_split_of[cq] = row["split"]

    splits = fiqa_query_splits(fiqa_qrels)
    existing = set(fq2qid.values())
    missing_course = [fq for fq in fq2qid if fq not in fiqa_queries]
    if missing_course:
        raise ValueError(f"course queries map to FiQA ids with no query text: {missing_course[:5]}")

    queries, empty = [], []
    for fq in sorted(fiqa_queries, key=fiqa_sort_key):
        text = course_clean(fiqa_queries[fq] or "")
        if fq in fq2qid:
            qid = fq2qid[fq]
        else:
            if not text:
                empty.append(fq)
                continue
            qid = mint_query_id(fq)
            if qid in existing:
                raise ValueError(f"minted id {qid} for FiQA query {fq} collides with an existing id")
            existing.add(qid)
            fq2qid[fq] = qid
        queries.append({
            "query_id": qid,
            "fiqa_query_id": fq,
            "fiqa_split": ";".join(splits.get(fq, [])),
            "course_split": course_split_of.get(qid),
            "text": text,
        })

    stats = {
        "fiqa_queries": len(fiqa_queries),
        "course_queries": len(course_split_of),
        "new_queries": len(queries) - len(course_split_of),
        "total_queries": len(queries),
        "dropped_empty": len(empty),
        "queries_without_qrels": sum(1 for q in queries if not q["fiqa_split"]),
        "queries_in_multiple_fiqa_splits": sum(1 for q in queries if ";" in q["fiqa_split"]),
    }
    return queries, fq2qid, stats


def migrate_qrels(fiqa_qrels, fq2qid, fiqa2rid):
    """FiQA qrels (split -> fiqa_qid -> {fiqa_doc}) -> split -> qid -> {record_id}."""
    out: dict[str, dict[str, set[str]]] = {}
    dropped = Counter()
    for split in FIQA_SPLITS:
        d: dict[str, set[str]] = defaultdict(set)
        for fq, docs in fiqa_qrels.get(split, {}).items():
            qid = fq2qid.get(fq)
            if qid is None:
                dropped["unknown_query"] += len(docs)
                continue
            for doc in docs:
                rid = fiqa2rid.get(doc)
                if rid is None:
                    dropped["unknown_or_empty_doc"] += 1
                    continue
                d[qid].add(rid)
        out[split] = dict(d)
    return out, dict(dropped)


def course_split_qrels(migrated, queries):
    """Union of migrated FiQA qrels for the course practice/dev queries."""
    split_of = {q["query_id"]: q["course_split"] for q in queries if q["course_split"]}
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for qs in migrated.values():
        for qid, rids in qs.items():
            if qid in split_of:
                out[split_of[qid]][qid] |= rids
    return {s: dict(v) for s, v in out.items()}


def check_course_invariant(full_course_qrels, course_qrels, original_ids):
    """Restricted to the original corpus, the migrated qrels must equal the course qrels."""
    bad = []
    for split, qs in course_qrels.items():
        got = full_course_qrels.get(split, {})
        for qid, rids in qs.items():
            restricted = got.get(qid, set()) & original_ids
            if restricted != set(rids):
                bad.append((split, qid, sorted(rids), sorted(restricted)))
        extra = set(got) - set(qs)
        bad.extend((split, qid, [], "unexpected query") for qid in extra)
    if bad:
        raise AssertionError(f"{len(bad)} course queries fail the qrels invariant, e.g. {bad[:3]}")
    n_extended = sum(1 for split, qs in full_course_qrels.items()
                     for qid, rids in qs.items() if rids - original_ids)
    return n_extended


# ================================================================ io helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _umask() -> int:
    old = os.umask(0)
    os.umask(old)
    return old


def atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o666 & ~_umask())  # mkstemp creates 0600
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str):
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_save_npy(path: Path, arr):
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    atomic_write_bytes(path, buf.getvalue())


def write_qrels_tsv(path: Path, qrels: dict[str, set[str]]):
    lines = ["query_id\trecord_id\tscore"]
    for qid in sorted(qrels):
        for rid in sorted(qrels[qid]):
            lines.append(f"{qid}\t{rid}\t1.0")
    atomic_write_text(path, "\n".join(lines) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fiqa_file_hashes() -> dict[str, str]:
    """sha256 of the cached FiQA files (same files load_fiqa reads)."""
    from huggingface_hub import hf_hub_download
    files = [(FIQA_REPO, "corpus/corpus-00000-of-00001.parquet"),
             (FIQA_REPO, "queries/queries-00000-of-00001.parquet")]
    files += [(FIQA_QRELS_REPO, f"{s}.tsv") for s in FIQA_SPLITS]
    return {f"{repo}/{p}": sha256_file(Path(hf_hub_download(repo, p, repo_type="dataset")))
            for repo, p in files}


# ================================================================ fetch
def cmd_fetch(args):
    corpus, queries, qrels = load_fiqa()
    print(f"FiQA cached: {len(corpus)} docs, {len(queries)} queries, "
          + ", ".join(f"{s}: {len(q)} q" for s, q in qrels.items()))
    for k, v in fiqa_file_hashes().items():
        print(f"  {v}  {k}")


# ================================================================ prepare
def cmd_prepare(args):
    out: Path = args.out_dir
    src = CourseSource(args.corpus, args.zip)
    print(f"Course data: {src.where}")
    course = src.corpus()
    course_qrels = src.qrels()
    if set(course_qrels) != {"practice", "dev"}:
        sys.exit(f"expected practice + dev course qrels, found {sorted(course_qrels)}")
    print(f"  {len(course)} records, " + ", ".join(f"{k}: {len(v)} queries" for k, v in course_qrels.items()))

    print("Loading FiQA from the HF cache ...")
    try:
        fiqa_corpus, fiqa_queries, fiqa_qrels = load_fiqa()
    except Exception as e:
        if type(e).__name__ != "LocalEntryNotFoundError":
            raise
        sys.exit("Could not download BeIR/fiqa from the Hugging Face Hub and it is not cached in "
                 f"{os.environ.get('HF_HUB_CACHE') or os.environ.get('HF_HOME') or '~/.cache/huggingface'}.\n"
                 "Check that this node has internet access and HF_HUB_OFFLINE is not set.")
    print(f"  {len(fiqa_corpus)} docs, {len(fiqa_queries)} queries")

    print("Matching course records ...")
    results, unmatched = match_records(course, fiqa_corpus, threshold=1.01)  # disable fuzzy acceptance
    if unmatched:
        sys.exit(f"{len(unmatched)} course records did not match FiQA exactly; refusing to continue")
    bad_tier = Counter(t for _, t, _ in results.values() if t not in ("T1_exact", "T2a_clean_recipe"))
    if bad_tier:
        sys.exit(f"records matched only by loose tiers {dict(bad_tier)}; refusing to continue")
    rec_matches = {rid: ids for rid, (ids, _, _) in results.items()}

    records, fiqa2rid, rec_stats = build_records(course, rec_matches, fiqa_corpus)
    print(f"  records: {rec_stats['course_records']} course + {rec_stats['new_records']} new "
          f"= {rec_stats['total_records']} (empty dropped {rec_stats['dropped_empty']}, "
          f"aliased to course {rec_stats['aliased_dup_of_course']}, to new {rec_stats['aliased_dup_of_new']})")

    print("Matching course queries ...")
    original_ids = {r["record_id"] for r in records if r["source"] == "course"}
    qrows = match_course_queries(course_qrels, fiqa_qrels, fiqa2rid, original_ids)
    print(f"  match kinds: {dict(Counter(r['match_kind'] for r in qrows))}")
    queries, fq2qid, q_stats = build_queries(qrows, fiqa_queries, fiqa_qrels)
    print(f"  queries: {q_stats['course_queries']} course + {q_stats['new_queries']} new = {q_stats['total_queries']}")

    print("Migrating qrels ...")
    migrated, dropped = migrate_qrels(fiqa_qrels, fq2qid, fiqa2rid)
    full_course = course_split_qrels(migrated, queries)
    n_extended = check_course_invariant(full_course, course_qrels, original_ids)
    print(f"  invariant OK: migrated qrels restricted to the 24k == course qrels for all "
          f"{sum(len(v) for v in course_qrels.values())} course queries "
          f"({n_extended} gain relevant docs outside the 24k); dropped {dropped}")

    # ---- write
    out.mkdir(parents=True, exist_ok=True)
    stale = [p for p in (out / "embeddings").glob("*") if p.is_file()]
    for p in stale:  # embeddings from an older prepare would no longer line up
        p.unlink()

    atomic_write_text(out / "corpus.jsonl", "".join(
        json.dumps({"record_id": r["record_id"], "text": r["text"]}, ensure_ascii=False) + "\n"
        for r in records))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["record_id", "fiqa_id", "source", "alias_fiqa_ids"])
    for r in records:
        w.writerow([r["record_id"], r["fiqa_id"], r["source"], ";".join(r["aliases"])])
    atomic_write_text(out / "records.csv", buf.getvalue())
    atomic_write_text(out / "queries.jsonl", "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries))
    for split, qs in migrated.items():
        write_qrels_tsv(out / f"qrels_fiqa_{split}.tsv", qs)
    for split in ("practice", "dev"):
        write_qrels_tsv(out / f"qrels_{split}.tsv", full_course.get(split, {}))

    corpus_text = src.read_text(src.corpus_name)
    manifest = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "course_source": src.where,
        "course_corpus_sha256": hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
        "fiqa_files_sha256": fiqa_file_hashes(),
        "clean_recipe": CLEAN_RECIPE,
        "record_id_rule_new": RECORD_RULE,
        "query_id_rule_new": QUERY_RULE,
        "corpus_order": "original course records in course order, then new FiQA docs by numeric FiQA id",
        "records": rec_stats,
        "queries": q_stats,
        "qrels": {
            "fiqa_pairs": {s: sum(len(v) for v in qs.values()) for s, qs in migrated.items()},
            "dropped": dropped,
            "course_queries_extended_beyond_24k": n_extended,
            "note": ("qrels_practice/dev.tsv are FiQA qrels for the course queries mapped onto this corpus "
                     "(correct at this scale); restricted to the original 24k they equal the course files. "
                     "FiQA's extra judgements for the 3 restricted course queries point to empty docs, which are dropped. "
                     "All scores are 1.0 (FiQA judgements are binary)."),
        },
        "encode": None,
    }
    atomic_write_text(out / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out}")


# ================================================================ encode
def load_original_embeddings(course_dir: Path, zip_path: Path):
    import numpy as np
    emb_dir = course_dir / "embeddings"
    if (emb_dir / "corpus_enc_a.npy").is_file():
        vecs = np.load(emb_dir / "corpus_enc_a.npy")
        ids = json.loads((emb_dir / "corpus_enc_a_ids.json").read_text(encoding="utf-8"))
        where = str(emb_dir)
    else:
        if not zip_path.is_file():
            sys.exit(f"no embeddings at {emb_dir} and no zip at {zip_path}")
        with zipfile.ZipFile(zip_path) as z:
            vecs = np.load(io.BytesIO(z.read(ZIP_PREFIX + "embeddings/corpus_enc_a.npy")))
            ids = json.loads(z.read(ZIP_PREFIX + "embeddings/corpus_enc_a_ids.json").decode("utf-8"))
        where = f"{zip_path}:{ZIP_PREFIX}embeddings/"
    vecs = np.asarray(vecs, dtype=np.float32)
    if len(ids) != vecs.shape[0] or len(set(ids)) != len(ids):
        sys.exit("original embeddings: row count / id list mismatch or duplicate ids")
    return vecs, ids, where


def cmd_encode(args):
    import numpy as np
    import torch
    from src.encoder import LocalEncoder

    data: Path = args.data_dir
    manifest_path = data / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"{manifest_path} missing; run prepare first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA requested but torch.cuda.is_available() is False")

    corpus = read_jsonl(data / "corpus.jsonl")
    with (data / "records.csv").open(encoding="utf-8", newline="") as f:
        meta = list(csv.DictReader(f))
    if [m["record_id"] for m in meta] != [r["record_id"] for r in corpus]:
        sys.exit("records.csv and corpus.jsonl are out of sync; re-run prepare")
    queries = read_jsonl(data / "queries.jsonl")

    orig_vecs, orig_ids, orig_where = load_original_embeddings(args.course_dir, args.zip)
    row_of = {rid: i for i, rid in enumerate(orig_ids)}
    course_rows = [r for r, m in zip(corpus, meta) if m["source"] == "course"]
    new_rows = [r for r, m in zip(corpus, meta) if m["source"] == "fiqa_ext"]
    n_course = len(course_rows)
    if corpus[:n_course] != course_rows:
        sys.exit("corpus.jsonl must list course records first")
    if {r["record_id"] for r in course_rows} != set(orig_ids):
        sys.exit("course records in corpus.jsonl differ from the original embedding ids")
    course_vecs = orig_vecs[[row_of[r["record_id"]] for r in course_rows]]

    print(f"Loading encoder from {args.model_dir} on {args.device} ...")
    enc = LocalEncoder(args.model_dir, device=args.device)
    msl = enc.model.max_seq_length
    if msl != MAX_SEQ_LENGTH:
        sys.exit(f"encoder max_seq_length is {msl}, expected {MAX_SEQ_LENGTH} (the stored vectors use 512)")
    enc.model.float()  # fp32, matching the stored vectors

    def encode(texts):
        if not texts:
            return np.zeros((0, orig_vecs.shape[1]), dtype=np.float32), 0.0
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        v = enc.encode_many(texts, batch_size=args.batch_size)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        return v, time.perf_counter() - t0

    # ---- sanity gate
    rng = np.random.default_rng(0)
    gate_n = min(args.gate_n, n_course)
    pick = rng.choice(n_course, size=gate_n, replace=False)
    gate_vecs, _ = encode([course_rows[i]["text"] for i in pick])
    cos = np.sum(gate_vecs * course_vecs[pick], axis=1)
    gate = {"n": int(gate_n), "seed": 0, "min_cos": float(cos.min()), "median_cos": float(np.median(cos)),
            "threshold": args.gate_min}
    print(f"Sanity gate: re-encoded {gate_n} original records, cosine min {gate['min_cos']:.6f} "
          f"median {gate['median_cos']:.6f} (need >= {args.gate_min})")
    if gate["min_cos"] < args.gate_min:
        sys.exit("SANITY GATE FAILED: re-encoded vectors do not match the stored corpus vectors; nothing written")

    partial = args.limit is not None
    if partial:
        new_rows = new_rows[:args.limit]
        queries = queries[:args.limit]
        print(f"--limit {args.limit}: PARTIAL run, row counts will not match corpus.jsonl")

    print(f"Encoding {len(new_rows)} new docs ...")
    new_vecs, t_docs = encode([r["text"] for r in new_rows])
    print(f"Encoding {len(queries)} queries ...")
    q_vecs, t_q = encode([q["text"] for q in queries])

    all_vecs = np.ascontiguousarray(np.concatenate([course_vecs, new_vecs]).astype(np.float32))
    all_ids = [r["record_id"] for r in course_rows] + [r["record_id"] for r in new_rows]
    q_vecs = np.ascontiguousarray(q_vecs.astype(np.float32))
    q_ids = [q["query_id"] for q in queries]
    for name, v in (("corpus", all_vecs), ("queries", q_vecs)):
        if not np.isfinite(v).all():
            sys.exit(f"{name} vectors contain non-finite values")
        norms = np.linalg.norm(v, axis=1)
        if v.shape[0] and not np.allclose(norms, 1.0, atol=1e-4):
            sys.exit(f"{name} vectors are not unit length (min {norms.min()}, max {norms.max()})")
    if not partial and all_ids != [r["record_id"] for r in corpus]:
        sys.exit("internal error: embedding ids do not follow corpus.jsonl order")
    if not np.array_equal(all_vecs[:n_course], course_vecs):
        sys.exit("internal error: original vectors were altered")

    emb = data / "embeddings"
    manifest["encode"] = None  # a crash mid-write must leave the build unpackageable
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    atomic_save_npy(emb / "corpus_enc_a.npy", all_vecs)
    atomic_write_text(emb / "corpus_enc_a_ids.json", json.dumps(all_ids))
    atomic_save_npy(emb / "queries_enc_a.npy", q_vecs)
    atomic_write_text(emb / "queries_enc_a_ids.json", json.dumps(q_ids))

    import sentence_transformers
    manifest["encode"] = {
        "partial": partial,
        "limit": args.limit,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "model_dir": str(args.model_dir),
        "max_seq_length": msl,
        "dtype": "float32",
        "batch_size": args.batch_size,
        "normalize_embeddings": True,
        "original_vectors_from": orig_where,
        "original_vectors_copied": n_course,
        "sanity_gate": gate,
        "new_docs_encoded": len(new_rows),
        "new_docs_seconds": round(t_docs, 2),
        "new_docs_per_second": round(len(new_rows) / t_docs, 1) if t_docs else None,
        "queries_encoded": len(queries),
        "queries_seconds": round(t_q, 2),
        "corpus_shape": list(all_vecs.shape),
        "queries_shape": list(q_vecs.shape),
        "query_note": "query vectors are from clean FiQA text, not ASR; use for systems / recall-vs-exact only",
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {emb}: corpus {all_vecs.shape}, queries {q_vecs.shape}; "
          f"{manifest['encode']['new_docs_per_second']} docs/s")


# ================================================================ package / install
def data_files(d: Path) -> list[str]:
    return sorted(p.relative_to(d).as_posix() for p in d.rglob("*")
                  if p.is_file() and p.name != SUMS and not p.name.startswith("."))


def next_version(dest: Path) -> int:
    pat = re.compile(rf"^{re.escape(PKG_NAME)}-v(\d+)\.zip$")
    vs = [int(m.group(1)) for p in dest.glob(f"{PKG_NAME}-v*.zip") if (m := pat.match(p.name))]
    return max(vs, default=0) + 1


def check_alignment(data: Path, partial: bool):
    """Each .npy must match its ids file; full builds must follow corpus/queries jsonl order."""
    import numpy as np
    for kind, jsonl, key in (("corpus", "corpus.jsonl", "record_id"), ("queries", "queries.jsonl", "query_id")):
        v = np.load(data / "embeddings" / f"{kind}_enc_a.npy", mmap_mode="r")
        ids = json.loads((data / "embeddings" / f"{kind}_enc_a_ids.json").read_text(encoding="utf-8"))
        if v.shape[0] != len(ids):
            sys.exit(f"{kind}: {v.shape[0]} vectors but {len(ids)} ids; re-run encode")
        if not partial and ids != [r[key] for r in read_jsonl(data / jsonl)]:
            sys.exit(f"{kind}: embedding ids do not follow {jsonl}; re-run encode")


def cmd_package(args):
    data: Path = args.data_dir
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    enc = manifest.get("encode")
    if not enc:
        sys.exit("manifest has no encode section; run encode first")
    if enc.get("partial") and not args.allow_partial:
        sys.exit("this is a partial (--limit) build; pass --allow-partial to package it anyway")
    for name in ("corpus_enc_a.npy", "corpus_enc_a_ids.json", "queries_enc_a.npy", "queries_enc_a_ids.json"):
        if not (data / "embeddings" / name).is_file():
            sys.exit(f"missing embeddings/{name}")
    check_alignment(data, partial=bool(enc.get("partial")))

    files = data_files(data)
    sums = "".join(f"{sha256_file(data / f)}  {f}\n" for f in files)
    atomic_write_text(data / SUMS, sums)

    dest: Path = args.dest_dir
    dest.mkdir(parents=True, exist_ok=True)
    v = next_version(dest)
    stem = f"{PKG_NAME}-v{v}"
    final = dest / f"{stem}.zip"
    fd, tmp = tempfile.mkstemp(dir=dest, prefix=f".{stem}.", suffix=".zip")
    os.close(fd)
    os.chmod(tmp, 0o666 & ~_umask())
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for f in files + [SUMS]:
                z.write(data / f, f"{PKG_ROOT}/{f}")
        if final.exists():  # never overwrite a version
            sys.exit(f"{final} already exists")
        os.replace(tmp, final)
    finally:
        Path(tmp).unlink(missing_ok=True)
    shutil.copyfile(data / SUMS, dest / f"{stem}.{SUMS}")
    shutil.copyfile(data / "manifest.json", dest / f"{stem}.manifest.json")
    print(f"Packaged {len(files)} files -> {final}")
    print(f"  {final.stat().st_size / 1e6:.1f} MB, sha256 {sha256_file(final)}")


def verify_sums(d: Path) -> int:
    sums_path = d / SUMS
    if not sums_path.is_file():
        raise ValueError(f"no {SUMS} in {d}")
    listed = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, name = line.split("  ", 1)
            listed[name] = h
    present = set(data_files(d))
    if present != set(listed):
        raise ValueError(f"file set mismatch: missing {sorted(set(listed) - present)[:5]}, "
                         f"extra {sorted(present - set(listed))[:5]}")
    bad = [n for n, h in listed.items() if sha256_file(d / n) != h]
    if bad:
        raise ValueError(f"checksum mismatch: {bad[:5]}")
    return len(listed)


def cmd_install(args):
    to: Path = args.to.resolve()
    if to.exists() and any(to.iterdir()):
        if not args.force:
            sys.exit(f"{to} exists and is not empty; pass --force to replace it")
    to.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=to.parent, prefix=f".{to.name}.install."))
    try:
        with zipfile.ZipFile(args.from_zip) as z:
            for info in z.infolist():
                name = info.filename
                parts = Path(name).parts
                if (name.startswith("/") or ".." in parts or not parts or parts[0] != PKG_ROOT):
                    raise ValueError(f"unsafe or unexpected path in zip: {name!r}")
                if info.is_dir():
                    continue
                target = staging.joinpath(*parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        n = verify_sums(staging)
        os.chmod(staging, 0o777 & ~_umask())  # mkdtemp creates 0700
        if to.exists():
            old = to.with_name(f".{to.name}.old.{os.getpid()}")
            to.rename(old)
            try:
                staging.rename(to)
            except BaseException:
                old.rename(to)  # put the previous install back
                raise
            shutil.rmtree(old)
        else:
            staging.rename(to)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Installed {n} files into {to}; all checksums OK")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch", help="download FiQA into the HF cache (head node)")

    p = sub.add_parser("prepare", help="mint ids, dedupe, migrate qrels, write text files")
    p.add_argument("--corpus", type=Path, help="course corpus.jsonl (default: installed, else zip)")
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)

    p = sub.add_parser("encode", help="encode new docs + queries with enc_a")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--course-dir", type=Path, default=DEFAULT_COURSE_DIR)
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gate-n", type=int, default=1000)
    p.add_argument("--gate-min", type=float, default=0.9999)
    p.add_argument("--limit", type=int, help="smoke test: encode only the first N new docs / queries")

    p = sub.add_parser("package", help="zip the built directory with checksums")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dest-dir", type=Path, default=DEFAULT_PKG_DIR)
    p.add_argument("--allow-partial", action="store_true")

    p = sub.add_parser("install", help="extract a package zip and verify checksums")
    p.add_argument("--from", dest="from_zip", type=Path, required=True)
    p.add_argument("--to", type=Path, default=DEFAULT_OUT)
    p.add_argument("--force", action="store_true", help="replace an existing non-empty --to")

    args = ap.parse_args()
    {"fetch": cmd_fetch, "prepare": cmd_prepare, "encode": cmd_encode,
     "package": cmd_package, "install": cmd_install}[args.cmd](args)


if __name__ == "__main__":
    main()
