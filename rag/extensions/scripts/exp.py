"""Everything in the experiment sequence that isn't a timed bench.py cell.

  prep      step 1: course query set, exact top-100 ground truth, FAISS-vs-numpy exact check
  quality   step 9: qrels recall/MRR/NDCG@10 on course dev queries, exact vs HNSW
  pipeline  step 11: per-stage pipeline timings (ASR, encode, retrieve, generate) -> retrieval share

Run via laptop_rerun.py (prep, quality) or run_experiments.sbatch (prep, pipeline).
Corpora are given as name=path/to/corpus_enc_a.npy.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json

import numpy as np

GT_K = 100


# ----------------------------------------------------------------------------- shared helpers

def corpora(specs):
    """['24k=/p/corpus_enc_a.npy', ...] -> {'24k': Path}."""
    return {s.split("=", 1)[0]: Path(s.split("=", 1)[1]).expanduser() for s in specs}


def ids_for(npy):
    return json.loads(Path(npy).with_name(Path(npy).stem + "_ids.json").read_text(encoding="utf-8"))


def exact_topk(Q, X, k, chunk=512):
    """Exact inner-product top-k (sorted), in query chunks to bound the score matrix."""
    out = np.empty((len(Q), k), dtype=np.int64)
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ X.T
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1, kind="stable")
        out[s:s + chunk] = np.take_along_axis(idx, order, 1)
    return out


def load_settings(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def hnsw_path(cache, name, M, efc):
    hits = sorted((Path(cache) / name).glob(f"hnsw-*-{M}-{efc}.faiss"))
    if not hits:
        raise SystemExit(f"no saved HNSW index for {name} M={M} efc={efc} in {Path(cache) / name}")
    return hits[0]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


# ----------------------------------------------------------------------------- prep (step 1)

def cmd_prep(a):
    import faiss
    cache = Path(a.cache)
    qdir = cache / "queries"
    qdir.mkdir(parents=True, exist_ok=True)

    # Query set: the 475 course queries (vectors of the cached ASR transcripts).
    sets = {"course": (np.load(a.course_queries).astype(np.float32), ids_for(a.course_queries))}
    for name, (Q, ids) in sets.items():
        np.save(qdir / f"{name}.npy", Q)
        (qdir / f"{name}_ids.json").write_text(json.dumps(ids))
        print(f"[prep] queries/{name}.npy: {Q.shape}")

    check = {"faiss_version": faiss.__version__, "simd_level": faiss.SIMDConfig.get_level_name(),
             "queries": {n: len(ids) for n, (Q, ids) in sets.items()}, "exact_agreement": {}}
    for cname, path in corpora(a.corpus).items():
        X = np.ascontiguousarray(np.load(path), dtype=np.float32)
        (cache / cname).mkdir(parents=True, exist_ok=True)
        flat = faiss.IndexFlatIP(X.shape[1])
        flat.add(X)
        for qname, (Q, _) in sets.items():
            gt = exact_topk(Q, X, GT_K)
            np.save(cache / cname / f"gt-{qname}-k{GT_K}.npy", gt)
            D, I = flat.search(Q, GT_K)
            # numpy's scores at its own top-100, rank by rank (chunked), to compare with FAISS's D
            S = np.concatenate([np.einsum("qd,qkd->qk", Q[c:c + 256], X[gt[c:c + 256]]) for c in range(0, len(Q), 256)])
            score_diff = float(np.abs(D - S).max())
            agree = {}
            for kk in (10, GT_K):
                agree[f"top{kk}_rows_identical"] = float((I[:, :kk] == gt[:, :kk]).all(axis=1).mean())
                agree[f"top{kk}_rows_same_set"] = float(np.mean([set(I[i, :kk]) == set(gt[i, :kk]) for i in range(len(Q))]))
            check["exact_agreement"][f"{cname}/{qname}"] = {
                "n": int(X.shape[0]), "nq": len(Q), **agree, "max_abs_rankwise_score_diff": score_diff,
                # Different summation order (BLAS vs FAISS) moves scores by ~1e-7, which reorders
                # near-ties. Identical score lists rank by rank mean the two agree up to that.
                "verdict": "agree up to float rounding" if score_diff < 1e-5 else "DISAGREE: check"}
            same_row = (I == gt).all(axis=1)
            same_set = np.array([set(I[i]) == set(gt[i]) for i in range(len(Q))])
            print(f"[prep] {cname}/{qname}: gt saved; faiss vs numpy top-100 rows {same_row.mean():.4f}, sets {same_set.mean():.4f}, "
                  f"top-10 sets {agree['top10_rows_same_set']:.4f}, max score diff {score_diff:.1e}")
    out = Path(a.out) / "check.json"
    out.write_text(json.dumps(check, indent=2))
    print(f"wrote {out}")


# ----------------------------------------------------------------------------- quality (step 9)

def load_qrels(path):
    qrels = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if float(r.get("score", 1)) > 0:
                qrels.setdefault(r["query_id"], {})[r["record_id"]] = float(r.get("score", 1))
    return qrels


def cmd_quality(a):
    import faiss
    from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
    s = load_settings(a.settings)
    qrels = load_qrels(a.qrels)
    qids = ids_for(a.queries)
    sel = [i for i, q in enumerate(qids) if q in qrels]
    Q = np.ascontiguousarray(np.load(a.queries)[sel], dtype=np.float32)
    print(f"[quality] {len(sel)} queries with qrels in {a.qrels}")
    rows = []
    for name, path in corpora(a.corpus).items():
        rec_ids = ids_for(path)
        X = np.ascontiguousarray(np.load(path), dtype=np.float32)
        exact = exact_topk(Q, X, a.k)
        index = faiss.read_index(str(hnsw_path(a.cache, name, s[name]["M"], s[name]["efc"])))
        runs = [("exact", "", exact)]
        for ef in dict.fromkeys((s[name]["ef95"], s[name]["ef99"])):
            index.hnsw.efSearch = ef
            runs.append(("hnsw", ef, index.search(Q, a.k)[1]))
        for method, ef, I in runs:
            m = {"recall": [], "mrr": [], "ndcg": [], "recall_vs_exact": []}
            for row, qi in enumerate(sel):
                ranked = [rec_ids[j] for j in I[row] if j >= 0]
                rel = qrels[qids[qi]]
                m["recall"].append(recall_at_k(ranked, list(rel), a.k))
                m["mrr"].append(mrr_at_k(ranked, list(rel), a.k))
                m["ndcg"].append(ndcg_at_k(ranked, rel, a.k))
                m["recall_vs_exact"].append(len(set(I[row]) & set(exact[row])) / a.k)
            rows.append(dict(corpus=name, method=method, M=s[name]["M"] if method == "hnsw" else "",
                             efc=s[name]["efc"] if method == "hnsw" else "", ef=ef, nq=len(sel), k=a.k,
                             **{f"{k_}@{a.k}": float(np.mean(v)) for k_, v in m.items()}))
            print(f"[quality] {name} {method} ef={ef}: " + " ".join(f"{k_}={np.mean(v):.4f}" for k_, v in m.items()))
    write_rows(Path(a.out) / "rows.csv", rows)


# ----------------------------------------------------------------------------- pipeline (step 11)

def cmd_pipeline(a):
    import os
    import time
    from src.asr_adapter import ASRAdapter
    from src.context_builder import SimpleContextBuilder
    from src.data import CourseData
    from src.encoder import LocalEncoder
    from src.generator import ELMGenerator
    from src.rag import RAGPipeline
    from src.retriever import ExactDenseRetriever
    from extensions.hnsw import HNSWRetriever

    from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k

    class Timed:
        """Wraps a retriever to also time the whole retrieve() call. The pipeline's retrieval_ms
        is the retriever's own figure (the search alone); call_ms adds its input checks and
        id lookup, i.e. what the pipeline actually waits for."""
        def __init__(self, r):
            self.r = r

        def retrieve(self, q, k=5):
            t = time.perf_counter()
            res = self.r.retrieve(q, k=k)
            res["call_ms"] = (time.perf_counter() - t) * 1000.0
            return res

    # HNSW at the tuned setting that is fastest while reaching 0.95 recall@10 against exact
    # search (step 2): the most favourable case for HNSW's latency, so if even this saves
    # nothing visible end to end, no slower HNSW setting will.
    s = load_settings(a.settings)[a.name]
    data_dir = Path(a.data_dir)
    emb, ids = data_dir / "embeddings" / "corpus_enc_a.npy", data_dir / "embeddings" / "corpus_enc_a_ids.json"
    os.environ["PERSIST_INDEX_PATH"] = str(Path(a.cache) / a.name)
    retrievers = {"exact_dense": Timed(ExactDenseRetriever(emb, ids)),
                  "hnsw": Timed(HNSWRetriever(emb, ids, M=s["M"], efConstruction=s["efc"], efSearch=s["ef95"]))}
    qrels = load_qrels(a.qrels or data_dir / "qrels_dev.tsv")
    data = CourseData(data_dir)
    asr = ASRAdapter()
    asr.prepare()
    pipe = RAGPipeline(asr=asr, encoder=LocalEncoder(ROOT / "models" / "enc_a", device=a.device),
                       retriever=None, data=data, context_builder=SimpleContextBuilder(), generator=ELMGenerator())
    items = data.audio_items("dev")
    if a.n:
        items = items[:a.n]
    print(f"[pipeline] {len(items)} dev queries; hnsw M={s['M']} efc={s['efc']} ef={s['ef95']}; k={a.k}")
    for r in retrievers.values():  # warm-up, discarded: model loading, first CUDA call
        pipe.retriever = r
        pipe.answer_audio(data_dir / items[0]["audio_file"], k=a.k)

    rows = []
    for i, item in enumerate(items):
        # Alternate which retriever goes first, so per-query warm-up (e.g. the encoder's first
        # call on new input) doesn't land on the same one every time.
        order = list(retrievers.items())
        for rname, r in (order if i % 2 == 0 else order[::-1]):
            pipe.retriever = r
            res = pipe.answer_audio(data_dir / item["audio_file"], k=a.k)
            t = res["timing"]
            ranked, rel = res["retrieval"]["record_ids"], qrels.get(item["query_id"], {})
            rows.append(dict(query_id=item["query_id"], retriever=rname, **t,
                             retrieval_call_ms=res["retrieval"]["call_ms"],
                             retrieval_share=res["retrieval"]["call_ms"] / t["end_to_end_ms"],
                             has_qrels=bool(rel),
                             **{f"recall@{a.k}": recall_at_k(ranked, list(rel), a.k) if rel else "",
                                f"mrr@{a.k}": mrr_at_k(ranked, list(rel), a.k) if rel else "",
                                f"ndcg@{a.k}": ndcg_at_k(ranked, rel, a.k) if rel else ""}))
            print(f"[pipeline] {item['query_id']} {rname}: " + " ".join(f"{k_}={v:.1f}" for k_, v in t.items()))
    asr.close()
    write_rows(Path(a.out) / "rows.csv", rows)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    p.add_argument("--cache", required=True)
    p.add_argument("--corpus", nargs="+", required=True, help="name=path/to/corpus_enc_a.npy")
    p.add_argument("--course-queries", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("quality")
    p.add_argument("--cache", required=True)
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--queries", required=True, help="course query vectors (ids json next to it)")
    p.add_argument("--qrels", required=True, help="qrels_dev.tsv")
    p.add_argument("--settings", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", required=True)

    p = sub.add_parser("pipeline")
    p.add_argument("--cache", required=True)
    p.add_argument("--name", default="24k", help="corpus label of the course corpus in the cache")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "rag_corpus"))
    p.add_argument("--settings", required=True)
    p.add_argument("--n", type=int, default=None, help="first N dev queries only (default: all dev queries)")
    p.add_argument("--qrels", default=None, help="default: <data-dir>/qrels_dev.tsv")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)

    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
