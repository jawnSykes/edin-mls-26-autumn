"""Everything in the experiment sequence that isn't a timed bench.py cell.

  prep      step 1: FiQA-extra query set, exact top-100 ground truth, FAISS-vs-numpy exact check
  pick      step 2: choose (M, efC), efSearch for 0.95/0.99 recall and Gorder's w -> settings.json
  diag      graph diagnostics: degree, hubness, reachability, memory split, link id gap per ordering
  quality   step 9: qrels recall/MRR/NDCG@10 on course dev queries, exact vs HNSW
  pipeline  step 11: per-stage pipeline timings (ASR, encode, retrieve, generate) -> retrieval share
  report    steps 6-7 (+ 2, 5, 9, 10, 11): tables and plots from every rows*.csv under $RESULTS_DIR

Normally run via run_experiments.sbatch. Corpora are given as name=path/to/corpus_enc_a.npy.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import re
import statistics
import subprocess

import numpy as np

GT_K = 100


# ----------------------------------------------------------------------------- shared helpers

def corpora(specs):
    """['24k=/p/corpus_enc_a.npy', ...] -> {'24k': Path}."""
    return {s.split("=", 1)[0]: Path(s.split("=", 1)[1]).expanduser() for s in specs}


def ids_for(npy):
    return json.loads(Path(npy).with_name(Path(npy).stem + "_ids.json").read_text(encoding="utf-8"))


def exact_topk(Q, X, k, chunk=512):
    """Exact inner-product top-k (sorted), in query chunks so 6.6k x 57.6k scores fit in memory."""
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


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def med(rows, col):
    return statistics.median(fnum(r[col]) for r in rows)


def group(rows, *cols):
    out = {}
    for r in rows:
        out.setdefault(tuple(r.get(c, "") for c in cols), []).append(r)
    return out


# ----------------------------------------------------------------------------- prep (step 1)

def cmd_prep(a):
    import faiss
    cache = Path(a.cache)
    qdir = cache / "queries"
    qdir.mkdir(parents=True, exist_ok=True)

    # Query sets: course (475, from cached ASR transcripts) and FiQA-extra (FiQA minus course queries).
    course_ids = ids_for(a.course_queries)
    course = np.load(a.course_queries).astype(np.float32)
    fiqa_ids = ids_for(a.fiqa_queries)
    fiqa = np.load(a.fiqa_queries).astype(np.float32)
    course_set = set(course_ids)
    keep = np.array([q not in course_set for q in fiqa_ids])
    if (~keep).sum() != len(course_set):
        raise SystemExit(f"only {(~keep).sum()} of {len(course_set)} course query ids found in the FiQA query set")
    sets = {"course": (course, course_ids), "fiqa_extra": (fiqa[keep], [q for q, k in zip(fiqa_ids, keep) if k])}
    for name, (Q, ids) in sets.items():
        np.save(qdir / f"{name}.npy", Q)
        (qdir / f"{name}_ids.json").write_text(json.dumps(ids))
        print(f"[prep] queries/{name}.npy: {Q.shape}")

    check = {"faiss_version": faiss.__version__, "simd_level": faiss.SIMDConfig.get_level_name(),
             "queries": {n: len(ids) for n, (Q, ids) in sets.items()},
             "fiqa_removed_as_course": int((~keep).sum()), "exact_agreement": {}}
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


# ----------------------------------------------------------------------------- pick (step 2)

def read_csvs(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def cmd_pick(a):
    rows = read_csvs(a.rows)
    s = load_settings(a.settings)

    t = [r for r in rows if r["mode"] == "time" and r["index"] == "hnsw" and r["order"] == "identity"]
    for (corpus,), rs in group(t, "corpus").items():
        cells = {}  # (M, efc) -> [(ef, recall, p50)]
        for (M, efc, ef), g in group(rs, "M", "efc", "ef").items():
            cells.setdefault((int(M), int(efc)), []).append((int(ef), med(g, "recall_at_k"), med(g, "p50_us")))
        best = None
        for (M, efc), pts in cells.items():
            ok = sorted(p for p in pts if p[1] >= a.select_recall)
            if ok and (best is None or ok[0][2] < best[2]):
                best = (M, efc, ok[0][2])
        if best is None:
            print(f"!! {corpus}: no (M, efC) reaches recall {a.select_recall}; widen the ef sweep")
            continue
        M, efc, _ = best
        pts = sorted(cells[(M, efc)])
        entry = s.setdefault(corpus, {})
        entry.update(M=M, efc=efc)
        for target, key in [(0.95, "ef95"), (0.99, "ef99")]:
            ok = [ef for ef, rec, _ in pts if rec >= target]
            entry[key] = ok[0] if ok else pts[-1][0]
            if not ok:
                print(f"!! {corpus}: recall {target} not reached; using largest ef {pts[-1][0]}")
        print(f"[pick] {corpus}: M={M} efc={efc} ef95={entry['ef95']} ef99={entry['ef99']}")

    g = [r for r in rows if r["mode"] == "cachegrind" and r["order"].startswith("gorder:")]
    for (corpus,), rs in group(g, "corpus").items():
        by_w = {int(r["order"].split(":")[1]): fnum(r["LL_miss_per_q"]) for r in rs}
        w = min(by_w, key=by_w.get)
        s.setdefault(corpus, {})["gorder_w"] = w
        print(f"[pick] {corpus}: gorder LL misses/q by w {by_w} -> w={w}")

    Path(a.settings).write_text(json.dumps(s, indent=2))
    print(f"wrote {a.settings}")


# ----------------------------------------------------------------------------- diag

def cmd_diag(a):
    import faiss
    from extensions.reorder import level0_csr, inverse
    from scipy.sparse.csgraph import breadth_first_order
    s = load_settings(a.settings)
    graph_rows, gap_rows = [], []
    for name in a.names:
        M, efc = s[name]["M"], s[name]["efc"]
        path = hnsw_path(a.cache, name, M, efc)
        index = faiss.read_index(str(path))
        n, d, h = index.ntotal, index.d, index.hnsw
        A = level0_csr(index)
        out_deg = np.diff(A.indptr)
        in_deg = np.asarray(A.sum(axis=0)).ravel().astype(float)  # k-occurrence
        top = np.sort(in_deg)[::-1][:max(1, n // 100)]
        reach = len(breadth_first_order(A, int(h.entry_point), directed=True, return_predecessors=False))
        n_links_total = faiss.vector_to_array(h.neighbors).size
        graph_rows.append(dict(
            corpus=name, M=M, efc=efc, n=n, max_level=h.max_level,
            out_deg_mean=out_deg.mean(), out_deg_min=out_deg.min(), out_deg_max=out_deg.max(),
            in_deg_mean=in_deg.mean(), in_deg_std=in_deg.std(), in_deg_max=in_deg.max(),
            in_deg_skew=float(((in_deg - in_deg.mean()) ** 3).mean() / in_deg.std() ** 3),
            frac_no_in_links=float((in_deg == 0).mean()), top1pct_share_of_in_links=top.sum() / in_deg.sum(),
            reachable_frac=reach / n,
            bytes_per_vec_vectors=d * 4, bytes_per_vec_level0_links=2 * M * 4,
            bytes_per_vec_upper_links=(n_links_total - n * 2 * M) * 4 / n,
            index_file_bytes=path.stat().st_size))
        # Locality: mean |id gap| along level-0 links under each cached ordering (lower = more local).
        coo = A.tocoo()
        layouts = [("identity", None)] + [
            (p.name[len(path.stem) + 1:-len(".perm.npz")].replace("-", ":", 1), p)
            for p in sorted(path.parent.glob(f"{path.stem}.*.perm.npz"))]
        for order, p in layouts:
            pos = np.arange(n) if p is None else inverse(np.load(p)["perm"])
            gap = np.abs(pos[coo.row] - pos[coo.col])
            gap_rows.append(dict(corpus=name, M=M, efc=efc, order=order, mean_link_id_gap=gap.mean(),
                                 median_link_id_gap=float(np.median(gap)),
                                 frac_links_within_64=float((gap <= 64).mean())))
        print(f"[diag] {name}: reachable {reach / n:.4f}, in-degree max {in_deg.max():.0f}, {len(layouts)} layouts")
    write_rows(Path(a.out) / "rows.csv", graph_rows)
    write_rows(Path(a.out) / "rows-gaps.csv", gap_rows)


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
    from src.asr_adapter import ASRAdapter
    from src.context_builder import SimpleContextBuilder
    from src.data import CourseData
    from src.encoder import LocalEncoder
    from src.generator import ELMGenerator
    from src.rag import RAGPipeline
    from src.retriever import ExactDenseRetriever
    from extensions.hnsw import HNSWRetriever

    s = load_settings(a.settings)[a.name]
    data_dir = Path(a.data_dir)
    emb, ids = data_dir / "embeddings" / "corpus_enc_a.npy", data_dir / "embeddings" / "corpus_enc_a_ids.json"
    os.environ["PERSIST_INDEX_PATH"] = str(Path(a.cache) / a.name)
    retrievers = {"exact": ExactDenseRetriever(emb, ids),
                  "hnsw": HNSWRetriever(emb, ids, M=s["M"], efConstruction=s["efc"], efSearch=s["ef95"])}
    data = CourseData(data_dir)
    asr = ASRAdapter()
    asr.prepare()
    pipe = RAGPipeline(asr=asr, encoder=LocalEncoder(ROOT / "models" / "enc_a", device=a.device),
                       retriever=None, data=data, context_builder=SimpleContextBuilder(), generator=ELMGenerator())
    items = data.audio_items("dev")[:a.n + 1]
    rows = []
    for i, item in enumerate(items):
        for rname, r in retrievers.items():
            pipe.retriever = r
            res = pipe.answer_audio(data_dir / item["audio_file"], k=a.k)
            if i == 0:
                continue  # warm-up: model loading, first CUDA call
            t = res["timing"]
            rows.append(dict(query_id=item["query_id"], retriever=rname, **t,
                             retrieval_share=t["retrieval_ms"] / t["end_to_end_ms"]))
            print(f"[pipeline] {item['query_id']} {rname}: " + " ".join(f"{k_}={v:.1f}" for k_, v in t.items()))
    asr.close()
    write_rows(Path(a.out) / "rows.csv", rows)


# ----------------------------------------------------------------------------- report (steps 6, 7)

def md_table(rows, cols):
    if not rows:
        return "_no data_\n"
    def fmt(v):
        if isinstance(v, str) and re.fullmatch(r"-?\d+\.\d{5,}(e-?\d+)?|nan", v):
            v = float(v)  # long decimals read back from CSV
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def size_bytes(text):
    """lscpu '12 MiB (1 instance)' -> bytes per instance."""
    m = re.match(r"([\d.]+)\s*([KMG])i?B?(?:\s*\((\d+) instance)?", text or "")
    if not m:
        return float("nan")
    return float(m.group(1)) * {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m.group(2)] / int(m.group(3) or 1)


def working_set(r):
    """Raw vectors + level-0 links (the brief's definition); exact search: vectors only."""
    n, d = fnum(r["n"]), fnum(r["d"])
    links = 2 * fnum(r["M"]) * 4 if r["index"] == "hnsw" else 0
    return n * (d * 4 + links)


def load_results(results):
    rows = []
    for p in sorted(Path(results).glob("*/*/rows*.csv")):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                r["step"], r["run"], r["file"] = p.parts[-3], p.parts[-2], p.name
                rows.append(r)
    return rows


def cmd_report(a):
    rows = load_results(a.results)
    out = Path(a.results) / "report"
    out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    md = [f"# ANN experiment report\n\nGenerated from {len({r['run'] for r in rows})} runs under `{a.results}`.\n"]
    step = lambda name: [r for r in rows if r["step"] == name]

    # --- Step 2: tuning
    tune = step("2-tune")
    md.append("## Step 2: settings (FiQA-extra queries)\n")
    md.append("```\n" + json.dumps(load_settings(Path(a.results) / "settings.json"), indent=2) + "\n```\n")
    gw = [dict(corpus=c, order=o, LL_miss_per_q=med(g, "LL_miss_per_q"), reorder_s=med(g, "reorder_s"))
          for (c, o), g in group([r for r in tune if r["mode"] == "cachegrind"], "corpus", "order").items()]
    md.append("Gorder window sweep (Cachegrind, net LL misses per query):\n\n" + md_table(gw, ["corpus", "order", "LL_miss_per_q", "reorder_s"]))

    # --- Step 3 + 6: stage 1 and the matched-recall headline
    s1 = [r for r in step("3-stage1") if r["mode"] == "time"]
    md.append("## Step 3: Stage 1 (course queries)\n")
    curve = [dict(host=h, corpus=c, index=i, M=M, efc=e, ef=ef, recall=med(g, "recall_at_k"), p50_us=med(g, "p50_us"),
                  p95_us=med(g, "p95_us"), batch_qps=med(g, "batch_qps"), ndis_per_q=med(g, "ndis_per_q"))
             for (h, c, i, M, e, ef), g in group(s1, "host", "corpus", "index", "M", "efc", "ef").items()]
    md.append(md_table(curve, ["host", "corpus", "index", "M", "efc", "ef", "recall", "p50_us", "p95_us", "batch_qps", "ndis_per_q"]))
    write_rows(out / "stage1_curve.csv", curve) if curve else None

    headline = []
    for (h, c), g in group(curve, "host", "corpus").items():
        ex = {r["index"]: r for r in g if r["index"] != "hnsw"}
        hn = sorted((r for r in g if r["index"] == "hnsw"), key=lambda r: int(r["ef"]))
        src = {r["index"]: r for r in s1 if r["host"] == h and r["corpus"] == c}
        for t in (0.90, 0.95, 0.99):
            hit = next((r for r in hn if r["recall"] >= t), None)
            row = dict(host=h, corpus=c, target=t, hnsw_ef=hit["ef"] if hit else "not reached",
                       hnsw_p50_us=hit["p50_us"] if hit else "", hnsw_batch_qps=hit["batch_qps"] if hit else "")
            for kind in ("flat", "numpy"):
                if kind in ex:
                    row[f"{kind}_p50_us"] = ex[kind]["p50_us"]
                    row[f"{kind}_batch_qps"] = ex[kind]["batch_qps"]
            if hit and "flat" in ex:
                row["speedup_vs_flat"] = ex["flat"]["p50_us"] / hit["p50_us"]
            if "hnsw" in src:
                r0 = src["hnsw"]
                row.update(hnsw_bytes_per_vec=fnum(r0["index_bytes"]) / fnum(r0["n"]), build_s=fnum(r0["build_s"]),
                           build_peak_rss_mb=fnum(r0["build_peak_rss_mb"]))
            headline.append(row)
    md.append("## Step 6: matched-recall headline (recall@10 vs exact)\n")
    md.append(md_table(headline, ["host", "corpus", "target", "hnsw_ef", "hnsw_p50_us", "flat_p50_us", "numpy_p50_us",
                                  "speedup_vs_flat", "hnsw_batch_qps", "flat_batch_qps", "hnsw_bytes_per_vec",
                                  "build_s", "build_peak_rss_mb"]))

    # --- Step 4 + 7: reordering wall clock, random-seed spread, break-even
    s4 = [r for r in step("4-reorder") if r["mode"] == "time"]
    reo = []
    for (h, c, ef), g in group(s4, "host", "corpus", "ef").items():
        by = {o: med(gg, "p50_us") for (o,), gg in group(g, "order").items()}
        spread = {o: (min(fnum(r["p50_us"]) for r in gg), max(fnum(r["p50_us"]) for r in gg)) for (o,), gg in group(g, "order").items()}
        rnd = [v for o, v in by.items() if o.startswith("random:")]
        base_id = by.get("identity")
        rmean = float(np.mean(rnd)) if rnd else float("nan")
        for o, v in sorted(by.items()):
            rs = next(fnum(r["reorder_s"]) for r in g if r["order"] == o)
            save_us = (base_id - v) if base_id else float("nan")
            reo.append(dict(host=h, corpus=c, ef=ef, order=o, p50_us=v, rounds_min=spread[o][0], rounds_max=spread[o][1],
                            gain_vs_identity=1 - v / base_id if base_id else "", gain_vs_random_mean=1 - v / rmean if rnd else "",
                            reorder_s=rs, breakeven_queries=rs / (save_us * 1e-6) if save_us and save_us > 0 and rs else ""))
        if rnd:
            reo.append(dict(host=h, corpus=c, ef=ef, order=f"random (mean of {len(rnd)} seeds)", p50_us=rmean,
                            rounds_min=min(rnd), rounds_max=max(rnd),
                            gain_vs_identity=1 - rmean / base_id if base_id else ""))
    md.append("## Steps 4 + 7: reordering (wall clock, course queries)\n\nFor random seeds, min/max are over seeds; otherwise over rounds. "
              "Break-even = reorder time / per-query saving vs identity.\n\n")
    md.append(md_table(reo, ["host", "corpus", "ef", "order", "p50_us", "rounds_min", "rounds_max", "gain_vs_identity",
                             "gain_vs_random_mean", "reorder_s", "breakeven_queries"]))

    # --- Step 5: cache-size sweep
    s5 = [r for r in step("5-cachesweep") if r["mode"] == "cachegrind"]
    sweep = []
    for (c, i, ef, LL), g in group(s5, "corpus", "index", "ef", "LL_bytes").items():
        ident = next((fnum(r["LL_miss_per_q"]) for r in g if r["order"] == "identity"), float("nan"))
        for r in g:
            v = fnum(r["LL_miss_per_q"])
            sweep.append(dict(corpus=c, index=i, ef=ef, order=r["order"], LL_MiB=fnum(LL) / 2**20,
                              ws_over_LL=working_set(r) / fnum(LL), D1_miss_per_q=fnum(r["D1_miss_per_q"]),
                              LL_miss_per_q=v, LL_miss_per_ndis=fnum(r["LL_miss_per_ndis"]),
                              LL_reduction_vs_identity=1 - v / ident if ident else float("nan")))
    sweep.sort(key=lambda r: (r["corpus"], r["index"], str(r["ef"]), r["order"], r["LL_MiB"]))
    if sweep:
        write_rows(out / "cachesweep.csv", sweep)
    # Real machines' points: wall-clock gain at working set / real L3.
    real = [dict(host=r["host"], corpus=r["corpus"], ef=r["ef"], order=r["order"], gain=r["gain_vs_identity"],
                 ws_over_L3=working_set(next(x for x in s4 if x["host"] == r["host"] and x["corpus"] == r["corpus"]))
                 / size_bytes(next(x for x in s4 if x["host"] == r["host"])["l3"]))
            for r in reo if r["order"] in ("bfs",) or r["order"].startswith("gorder:")]
    md.append("## Step 5: cache-size sweep (Cachegrind, course queries)\n\nFull table: `cachesweep.csv`. "
              "Real-machine wall-clock gains at their working set / L3:\n\n")
    md.append(md_table(real, ["host", "corpus", "ef", "order", "ws_over_L3", "gain"]))
    if plt and sweep:
        for ef in sorted({r["ef"] for r in sweep if r["index"] == "hnsw"}):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            for (c, o), g in group([r for r in sweep if r["index"] == "hnsw" and r["ef"] == ef], "corpus", "order").items():
                g = sorted(g, key=lambda r: r["ws_over_LL"])
                x = [r["ws_over_LL"] for r in g]
                ls = "-" if c == sorted({r["corpus"] for r in sweep})[0] else "--"
                axes[0].plot(x, [r["LL_miss_per_q"] for r in g], ls, marker="o", label=f"{c} {o}")
                if o != "identity":
                    axes[1].plot(x, [r["LL_reduction_vs_identity"] for r in g], ls, marker="o", label=f"{c} {o}")
            for r in real:
                if str(r["ef"]) == str(ef) and r["gain"] != "":
                    axes[1].scatter([r["ws_over_L3"]], [r["gain"]], marker="*", s=150, zorder=5)
                    axes[1].annotate(f"{r['host']} {r['corpus']} {r['order']} (wall clock)", (r["ws_over_L3"], r["gain"]), fontsize=7)
            for ax, yl in zip(axes, ["LL misses per query", "reduction vs original order"]):
                ax.set_xscale("log"); ax.set_xlabel("working set / simulated LL"); ax.set_ylabel(yl)
            axes[0].legend(fontsize=7)
            fig.suptitle(f"Cache-size sweep, efSearch={ef}")
            fig.tight_layout()
            fig.savefig(out / f"cachesweep_ef{ef}.png", dpi=120)
            plt.close(fig)
            md.append(f"![cache sweep ef={ef}](cachesweep_ef{ef}.png)\n")
    if plt and curve:
        for (h,), g in group(curve, "host").items():
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for (c, i), gg in group(g, "corpus", "index").items():
                gg = sorted(gg, key=lambda r: r["recall"])
                ax.plot([r["recall"] for r in gg], [r["p50_us"] for r in gg], marker="o", label=f"{c} {i}")
            ax.set_yscale("log"); ax.set_xlabel("recall@10 vs exact"); ax.set_ylabel("p50 latency (us)")
            ax.legend(fontsize=7); ax.set_title(f"Recall vs latency, {h}")
            fig.tight_layout(); fig.savefig(out / f"recall_latency_{h}.png", dpi=120); plt.close(fig)
            md.append(f"![recall vs latency {h}](recall_latency_{h}.png)\n")
    if not plt:
        md.append("_matplotlib not installed: plots skipped; plot from the CSVs in this folder._\n")

    # --- Step 9, 10, 11, diag
    q = step("9-quality")
    md.append("## Step 9: qrels quality (course dev queries)\n\n" + md_table(q, [c for c in (q[0] if q else {}) if c not in ("step", "run", "file")]))
    s10 = [r for r in step("10-queryset") if r["mode"] == "time"]
    other = [r for r in s1 + s4]
    qs = []
    for (h, c, i, o, ef), g in group(s10, "host", "corpus", "index", "order", "ef").items():
        ref = [r for r in other if (r["host"], r["corpus"], r["index"], r["order"], r["ef"]) == (h, c, i, o, ef)]
        qs.append(dict(host=h, corpus=c, index=i, order=o, ef=ef, fiqa_recall=med(g, "recall_at_k"), fiqa_p50_us=med(g, "p50_us"),
                       course_recall=med(ref, "recall_at_k") if ref else "", course_p50_us=med(ref, "p50_us") if ref else ""))
    md.append("## Step 10: query-set robustness (FiQA-extra vs course)\n\n" + md_table(qs, ["host", "corpus", "index", "order", "ef", "course_recall", "fiqa_recall", "course_p50_us", "fiqa_p50_us"]))
    pl = step("11-pipeline")
    pt = [dict(retriever=rn, n=len(g), **{c: med(g, c) for c in ("asr_ms", "embedding_ms", "retrieval_ms", "generation_ms", "end_to_end_ms", "retrieval_share")})
          for (rn,), g in group(pl, "retriever").items()]
    md.append("## Step 11: share of end-to-end time (medians)\n\n" + md_table(pt, ["retriever", "n", "asr_ms", "embedding_ms", "retrieval_ms", "generation_ms", "end_to_end_ms", "retrieval_share"]))
    dg = [r for r in step("diag") if r["file"] == "rows.csv"]
    gp = [r for r in step("diag") if r["file"] == "rows-gaps.csv"]
    md.append("## Graph diagnostics\n\n" + md_table(dg, [c for c in (dg[0] if dg else {}) if c not in ("step", "run", "file")]))
    md.append("\nLocality per ordering (mean id gap along level-0 links):\n\n" + md_table(gp, ["corpus", "order", "mean_link_id_gap", "median_link_id_gap", "frac_links_within_64"]))

    (out / "summary.md").write_text("\n".join(md))
    print(f"wrote {out / 'summary.md'}")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    p.add_argument("--cache", required=True)
    p.add_argument("--corpus", nargs="+", required=True, help="name=path/to/corpus_enc_a.npy")
    p.add_argument("--course-queries", required=True)
    p.add_argument("--fiqa-queries", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("pick")
    p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--settings", required=True)
    p.add_argument("--select-recall", type=float, default=0.95)

    p = sub.add_parser("diag")
    p.add_argument("--cache", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--settings", required=True)
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
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)

    p = sub.add_parser("report")
    p.add_argument("--results", required=True)

    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
