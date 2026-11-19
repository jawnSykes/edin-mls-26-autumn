"""Configuration, then one query in, top-k out: ExactDenseRetriever vs HNSW at a few efSearch values.

First prints the setup (corpus size, graph settings, CPU caches, FAISS, cache status). Then shows
the retriever's input (a dev query: its ID, transcript if available, and its enc_a vector)
and output (ranked record IDs, scores, qrels relevance, a text snippet), plus how many corpus
vectors each method scored and its latency.

Run from rag/ inside the demo shell (extensions/scripts/demo-shell.sh sets PERSIST_INDEX_PATH):
    taskset -c 2 python extensions/scripts/demo_query.py [--qid Q] [--efs 32 128] [--k 5]
    python extensions/scripts/demo_query.py --config-only      # just the setup block

Notes
  - The query vector is the cached course vector (queries/course.npy), the same one every
    experiment used, so the ASR + encoder (GPU) step is not rerun here.
  - Default query: the first dev query that has qrels. Try a few with --qid before recording
    and pick one where the ef=32 list differs slightly from exact (shows the approximation)
    or where all agree (shows recall ~1) -- whichever you want to talk about.
  - Transcript: taken from the audio manifest if it carries one, else from --transcripts
    (a jsonl with query_id + transcript, e.g. the pipeline's transcript cache); otherwise
    only the ID and audio file name are shown.
  - Latency is the median of --reps calls on a warm index, single-threaded; pin the CPU with
    taskset for numbers comparable to the report.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os
import time

import numpy as np

BOLD, DIM, GREEN, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[0m"


def first_text(row, keys=("transcript", "text", "query_text", "question", "query")):
    for k in keys:
        if isinstance(row.get(k), str) and row[k].strip():
            return row[k].strip()
    return None


def median_ms(fn, reps):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qid", default=None, help="query_id (default: first dev query with qrels)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--efs", nargs="+", type=int, default=[32, 128])
    ap.add_argument("--name", default="24k", help="corpus label in the cache and settings.json")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "rag_corpus"))
    ap.add_argument("--settings", default=str(ROOT / "results/ann-exp-data/consolidated/settings.json"))
    ap.add_argument("--transcripts", default=None, help="optional jsonl with query_id + transcript")
    ap.add_argument("--reps", type=int, default=50, help="calls per method for the median latency")
    ap.add_argument("--snippet", type=int, default=70, help="characters of record text to show")
    ap.add_argument("--config-only", action="store_true", help="print the setup block and stop")
    a = ap.parse_args()

    if "PERSIST_INDEX_PATH" not in os.environ:
        raise SystemExit("set PERSIST_INDEX_PATH to the ANN cache (e.g. ~/mls-rag-assets/ann-cache-laptop)")
    cache = Path(os.environ["PERSIST_INDEX_PATH"]).expanduser()

    import faiss
    from src.data import CourseData, load_jsonl
    from src.retriever import ExactDenseRetriever
    from extensions.hnsw import HNSWRetriever

    faiss.omp_set_num_threads(1)
    data_dir = Path(a.data_dir)
    emb = data_dir / "embeddings" / "corpus_enc_a.npy"
    ids_path = data_dir / "embeddings" / "corpus_enc_a_ids.json"
    s = json.loads(Path(a.settings).read_text())[a.name]

    # ---- setup
    X = np.load(emb, mmap_mode="r")
    graph = cache / a.name / f"hnsw-{emb.stem}-{s['M']}-{s['efc']}.faiss"
    perm = graph.with_name(f"{graph.stem}.gorder-{s['gorder_w']}.perm.npz")
    print(f"{BOLD}SETUP{RESET}")
    print(f"  corpus   {X.shape[0]:,} docs x {X.shape[1]}-d {X.dtype} = {X.nbytes / 2**20:.1f} MiB (enc_a, L2-normalised)")
    print(f"  exact    ExactDenseRetriever: one matrix-vector product over every doc, then top-k")
    print(f"  HNSW     FAISS IndexHNSWFlat M={s['M']} efConstruction={s['efc']} (unquantised); graph "
          + ("cached" if graph.exists() else "MISSING: built on first use (~12 s)"))
    print(f"  orders   identity, random, bfs, gorder:{s['gorder_w']}; gorder perm "
          + ("cached" if perm.exists() else "MISSING: bench.py computes it (~80 s)"))
    lscpu = ""
    try:
        import subprocess
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        pass
    pick = lambda k: next((l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.strip().startswith(k + ":")), "?")
    print(f"  cpu      {pick('Model name')}   L2 {pick('L2 cache')}   L3 {pick('L3 cache')}")
    print(f"  faiss    {faiss.__version__}  SIMD={os.environ.get('FAISS_SIMD_LEVEL', 'default')}  threads=1"
          f"  pinned to cpu {sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'}\n")
    if a.config_only:
        return

    # ---- input: one query
    Q = np.load(cache / "queries" / "course.npy")
    qids = json.loads((cache / "queries" / "course_ids.json").read_text())
    data = CourseData(data_dir)
    qrels = data.load_qrels("dev")
    manifest = {r["query_id"]: r for r in data.load_audio_manifest()}
    if a.qid is None:
        a.qid = next(q for q in qids if manifest.get(q, {}).get("split") == "dev" and q in qrels)
    if a.qid not in qids:
        raise SystemExit(f"{a.qid} is not among the cached course queries")
    q = Q[qids.index(a.qid)]
    row = manifest.get(a.qid, {})
    text = first_text(row)
    if text is None and a.transcripts:
        text = next((first_text(r) for r in load_jsonl(a.transcripts) if r.get("query_id") == a.qid), None)
    rel = qrels.get(a.qid, {})

    print(f"{BOLD}INPUT{RESET}  query {a.qid}  ({row.get('split', '?')} split, audio {Path(str(row.get('audio_file', '?'))).name})")
    if text:
        print(f"  transcript: \"{text}\"")
    print(f"  enc_a vector: {q.shape[0]}-d float32, |q| = {np.linalg.norm(q):.3f}   relevant docs in qrels: {len(rel)}")

    # ---- retrievers
    exact = ExactDenseRetriever(emb, ids_path)
    n = len(exact.record_ids)
    os.environ["PERSIST_INDEX_PATH"] = str(cache / a.name)  # where bench.py keeps the 24k graph
    hnsw = HNSWRetriever(emb, ids_path, M=s["M"], efConstruction=s["efc"], efSearch=a.efs[0])
    if not hnsw.loaded:
        print(f"{DIM}(built a new HNSW graph: no cached index at {hnsw.index_path.name}){RESET}")
    stats = faiss.cvar.hnsw_stats

    runs = [("exact (ExactDenseRetriever)", exact, None)] + [
        (f"HNSW  M={s['M']} efC={s['efc']} efSearch={ef}", hnsw, ef) for ef in a.efs]
    exact_top = None
    for label, r, ef in runs:
        if ef is not None:
            r.set_ef_search(ef)
        ms = median_ms(lambda: r.retrieve(q, k=a.k), a.reps)
        stats.reset()
        res = r.retrieve(q, k=a.k)
        scored = n if ef is None else int(stats.ndis)
        top = res["record_ids"]
        exact_top = exact_top or top
        same = len(set(top) & set(exact_top))
        print(f"\n{BOLD}{label}{RESET}")
        print(f"  vectors scored: {scored:>6,} of {n:,}   latency (median of {a.reps}): {ms:.3f} ms"
              + ("" if ef is None else f"   same top-{a.k} as exact: {same}/{a.k}"))
        for i, (rid, sc) in enumerate(zip(top, res["scores"]), 1):
            mark = f"{GREEN}✓{RESET}" if rid in rel else " "
            snip = " ".join(data.record_text(rid).split())[:a.snippet]
            print(f"  {i}. {mark} {rid:<14} {sc:.4f}  {DIM}{snip}{RESET}")
    print(f"\n{DIM}✓ = relevant in qrels_dev.tsv{RESET}")


if __name__ == "__main__":
    main()
