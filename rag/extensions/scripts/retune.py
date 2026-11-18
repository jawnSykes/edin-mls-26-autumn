"""Derive the 24k operating points (ef95/ef99 in settings.json) on the course *practice* queries.

Uses only the course corpus (24k), the 350 practice-split course queries and their exact ground
truth, so the 125 dev queries (the qrels quality set) stay unseen.

Standard HNSW practice, nothing chosen for being "best":
  - Build parameters are FIXED at the common hnswlib defaults, M = 16, efConstruction = 200
    (the graph every experiment measured). They are not tuned.
  - efSearch is the query-time knob tuned to a recall target on held-out queries: ef95 = the
    smallest ef in the sweep with recall@10 >= 0.95 on the practice queries. The smallest ef
    reaching 0.99 is reported as well, so the high-recall point can be labelled honestly.
  - Gorder's window stays FIXED at w = 10. A wall-clock sensitivity check against the other
    windows (same graph, same ef, interleaved rounds) shows how much that choice matters.

  1. prep     practice subset of the course query vectors (the same ASR-transcript vectors every
              step measured); exact top-100 ground truth on the 24k corpus, cross-checked against
              the rows of the existing course ground truth.
  2. sweep    bench.py --mode time, fixed graph, every ef.
  3. gorder   bench.py --mode time, identity + gorder:w for each w, at the reported ef values.
  4. report   summary.md + settings-retune.json (same format as settings.json, never overwrites it).

Everything is cached under a separate PERSIST_INDEX_PATH (default ~/mls-rag-assets/ann-cache-retune);
the measured graph and its permutations are symlinked in read-only, so nothing is rebuilt.

Run from rag/ on the laptop, pinned to the same P-core as every other laptop measurement:
    BENCH_CPU=2 extensions/scripts/with_pinned_freq.sh \\
        ~/mls-rag-env/bin/python extensions/scripts/retune.py
Quick plumbing check (1 round, unpinned, throwaway output):
    ~/mls-rag-env/bin/python extensions/scripts/retune.py --smoke
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # rag/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import csv
import json
import os
import shutil
import socket
import statistics
import subprocess
import time

import numpy as np

from exp import exact_topk, GT_K  # same exact top-k as step 1

HOME = Path.home()
ASSETS = HOME / "mls-rag-assets"
NAME = "24k"
TARGETS = [(0.95, "ef95"), (0.99, "ef99")]


def sh(cmd, env, log):
    line = "$ " + " ".join(map(str, cmd))
    print(line, flush=True)
    with open(log, "a") as f:
        f.write(line + "\n")
    rc = subprocess.run(list(map(str, cmd)), env=env).returncode
    if rc != 0:
        raise SystemExit(f"command failed (rc={rc})")


def read_rows(path):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["mode"] == "time"]


def cells(rows, *keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def stats(rs, col):
    v = [float(r[col]) for r in rs]
    return statistics.median(v), min(v), max(v)


# ----------------------------------------------------------------------------- 1. prep

def prep(a, cache):
    qdir, cdir = cache / "queries", cache / NAME
    qdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)

    ids = json.loads(a.course_queries.with_name(a.course_queries.stem + "_ids.json").read_text())
    Q = np.load(a.course_queries).astype(np.float32)
    split = {}
    for line in a.manifest.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            split[r["query_id"]] = r["split"]
    missing = [q for q in ids if q not in split]
    if missing:
        raise SystemExit(f"{len(missing)} course query ids have no split in {a.manifest}")
    rows = np.array([i for i, q in enumerate(ids) if split[q] == "practice"])
    n_dev = sum(split[q] == "dev" for q in ids)
    print(f"[prep] course queries: {len(ids)} = {len(rows)} practice + {n_dev} dev; tuning on practice only")
    if len(rows) + n_dev != len(ids):
        raise SystemExit("course queries are not exactly practice + dev")

    Qp = np.ascontiguousarray(Q[rows])
    qpath = qdir / "course_practice.npy"
    np.save(qpath, Qp)
    (qdir / "course_practice_ids.json").write_text(json.dumps([ids[i] for i in rows]))

    X = np.ascontiguousarray(np.load(a.embeddings), dtype=np.float32)
    gt = exact_topk(Qp, X, GT_K)
    np.save(cdir / f"gt-course_practice-k{GT_K}.npy", gt)
    # Cross-check: the same queries' rows in the ground truth every other step used.
    old = a.measured_cache / NAME / f"gt-course-k{GT_K}.npy"
    if old.exists():
        same10 = np.mean([set(gt[i, :10]) == set(g) for i, g in enumerate(np.load(old)[rows, :10])])
        print(f"[prep] practice ground truth vs existing course ground truth: top-10 sets equal for {same10:.4f}")
        if same10 < 1.0:
            raise SystemExit("practice ground truth disagrees with the existing course ground truth")

    # Reuse the measured graph and permutations read-only, via symlinks: nothing is rebuilt.
    stem = f"hnsw-{a.embeddings.stem}-{a.M}-{a.efc}"
    srcs = sorted((a.measured_cache / NAME).glob(f"{stem}.*"))
    if not any(p.suffix == ".faiss" for p in srcs):
        raise SystemExit(f"no measured graph {stem}.faiss in {a.measured_cache / NAME}")
    for p in srcs:
        link = cdir / p.name
        if not link.exists():
            link.symlink_to(p)
    print(f"[prep] linked {len(srcs)} measured graph/permutation files ({stem}.*) into {cdir}")
    return qpath, len(rows)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", type=Path, default=ROOT / "data/rag_corpus/embeddings/corpus_enc_a.npy")
    ap.add_argument("--course-queries", type=Path, default=ASSETS / "ann-cache/queries/course.npy",
                    help="the course query vectors every step measured (ids json next to it)")
    ap.add_argument("--manifest", type=Path, default=ASSETS / "data/rag_corpus/student_audio_manifest.jsonl",
                    help="course audio manifest: query_id -> split")
    ap.add_argument("--measured-cache", type=Path, default=ASSETS / "ann-cache",
                    help="cache every experiment used; read only (graph symlinked)")
    ap.add_argument("--cache", type=Path, default=ASSETS / "ann-cache-retune")
    ap.add_argument("--settings", type=Path, default=ROOT / "results/ann-exp-data/consolidated/settings.json",
                    help="settings every experiment used, for the comparison")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/ann-exp-data/laptop/ann/retune/<host>-<time>/")
    ap.add_argument("--M", type=int, default=16, help="fixed (hnswlib default)")
    ap.add_argument("--efc", type=int, default=200, help="fixed (hnswlib default)")
    ap.add_argument("--ef", nargs="+", default=["16", "32", "64", "128", "256", "512"])
    ap.add_argument("--gorder-w", type=int, default=10, help="fixed window; the one every experiment used")
    ap.add_argument("--gorder-alt", nargs="+", default=["3", "5"], help="windows for the sensitivity check")
    ap.add_argument("--rounds", type=int, default=3, help="ef-sweep rounds")
    ap.add_argument("--rounds-gorder", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3, help="timed passes over the practice queries per cell per round")
    ap.add_argument("--smoke", action="store_true", help="1 round, throwaway output in /tmp")
    a = ap.parse_args()

    ts = time.strftime("%Y%m%dT%H%M%S")
    host = socket.gethostname().split(".")[0]
    user = os.environ.get("USER", "u")
    if a.smoke:
        a.rounds = a.rounds_gorder = a.reps = 1
        a.cache = Path(f"/tmp/{user}/ann-retune-smoke-cache")
        a.out = Path(f"/tmp/{user}/ann-retune-smoke/{host}-{ts}")
    out = a.out or ROOT / "results/ann-exp-data/laptop/ann/retune" / f"{host}-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(f"/tmp/{user}/ann-retune-logs/{ts}")
    scratch.mkdir(parents=True, exist_ok=True)
    log = out / "commands.log"

    if not a.smoke and os.environ.get("BENCH_CPU") is None:
        print("!! BENCH_CPU is not set: not running under with_pinned_freq.sh? "
              "Every other laptop measurement used BENCH_CPU=2.", flush=True)

    qpath, nq = prep(a, a.cache)

    env = dict(os.environ, PERSIST_INDEX_PATH=str(a.cache), PYTHONUNBUFFERED="1",
               FAISS_SIMD_LEVEL=os.environ.get("FAISS_SIMD_LEVEL", "AVX2"),
               OPENBLAS_CORETYPE=os.environ.get("OPENBLAS_CORETYPE", "Haswell"))
    bench = [sys.executable, Path(__file__).resolve().parent / "bench.py", "--name", NAME,
             "--embeddings", a.embeddings, "--queries", qpath, "--index", "hnsw",
             "--M", a.M, "--efc", a.efc, "--log-dir", scratch]

    def cap(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
        except FileNotFoundError:
            return ""
    (out / "env.txt").write_text("\n".join([
        "step: retune (practice queries; fixed graph, efSearch selection, Gorder w sensitivity)",
        f"date: {ts}", f"host: {host}",
        f"git: {cap(['git', 'rev-parse', 'HEAD'])} on {cap(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])}",
        cap([sys.executable, "-c", "import sys, faiss, numpy; print('python', sys.executable, '| faiss', "
             "faiss.__version__, faiss.SIMDConfig.get_level_name(), '| numpy', numpy.__version__)"]),
        f"FAISS_SIMD_LEVEL={env['FAISS_SIMD_LEVEL']} OPENBLAS_CORETYPE={env['OPENBLAS_CORETYPE']} "
        f"BENCH_CPU={os.environ.get('BENCH_CPU', '')}",
        f"queries: {qpath} (nq={nq}, practice split)  cache: {a.cache}",
        f"args: {vars(a)}", "--- lscpu", cap(["lscpu"])]) + "\n")

    old = json.loads(a.settings.read_text()).get(NAME, {}) if a.settings.exists() else {}

    # 2. ef sweep on the fixed graph
    sweep_csv = out / "rows-sweep.csv"
    sh(bench + ["--ef", *a.ef, "--order", "identity", "--rounds", a.rounds, "--reps", a.reps,
                "--out", sweep_csv], env, log)
    curve = sorted((int(ef), stats(rs, "recall_at_k")[0], *stats(rs, "p50_us"))
                   for (ef,), rs in cells(read_rows(sweep_csv), "ef").items())
    pick = dict(M=a.M, efc=a.efc)
    for t, key in TARGETS:
        ok = [ef for ef, rec, *_ in curve if rec >= t]
        pick[key] = ok[0] if ok else None

    # 3. Gorder sensitivity at the ef values the experiments report (old settings), plus ef95 if new
    report_efs = sorted({int(e) for e in (old.get("ef95"), old.get("ef99"), pick["ef95"]) if e})
    ws = [a.gorder_w] + [int(w) for w in a.gorder_alt if int(w) != a.gorder_w]
    g_csv = out / "rows-gorder.csv"
    sh(bench + ["--ef", *report_efs, "--order", "identity", *[f"gorder:{w}" for w in ws],
                "--rounds", a.rounds_gorder, "--reps", a.reps, "--out", g_csv], env, log)
    res = {(o, int(ef)): stats(rs, "p50_us") for (o, ef), rs in cells(read_rows(g_csv), "order", "ef").items()}
    gains = []
    for ef in report_efs:
        ip, ilo, ihi = res[("identity", ef)]
        for w in ws:
            p, lo, hi = res[(f"gorder:{w}", ef)]
            gains.append(dict(w=w, ef=ef, p50=p, gain=1 - p / ip, noise=max((hi - lo) / p, (ihi - ilo) / ip)))

    # 4. report
    pick["gorder_w"] = a.gorder_w
    (out / "settings-retune.json").write_text(json.dumps({NAME: pick}, indent=2) + "\n")
    md = [f"# Re-derived operating points on {nq} course practice queries ({host}, {ts})\n",
          f"Graph fixed at M = {a.M}, efConstruction = {a.efc} (hnswlib defaults; the graph every experiment "
          f"measured). efSearch: smallest ef in the sweep reaching each recall target. "
          f"{a.rounds} rounds × {a.reps} reps (sweep), {a.rounds_gorder} rounds (Gorder).\n",
          "## Recall@10 and p50 by ef (practice queries)\n",
          "| ef | recall@10 | p50 µs (median) | min–max over rounds |", "|---|---|---|---|"]
    for ef, rec, p50, lo, hi in curve:
        md.append(f"| {ef} | {rec:.4f} | {p50:.1f} | {lo:.1f}–{hi:.1f} |")
    md.append("")
    for t, key in TARGETS:
        new, was = pick[key], old.get(key)
        rec = next((r for e, r, *_ in curve if e == new), None)
        line = f"- **{key}** (recall ≥ {t}): ef {new}" + (f" (recall {rec:.4f})" if rec is not None else " (not reached)")
        line += f"; experiments used ef {was}: " + ("**same**." if new == was else "**differs**.")
        if new != was and was is not None:
            wrec = next((r for e, r, *_ in curve if e == int(was)), None)
            line += (f" ef {was} has recall {wrec:.4f} here, so report it by its measured recall "
                     f"rather than as the {t} target." if wrec is not None else "")
        md.append(line)
    md.append("\n## Gorder window sensitivity (fixed w = %d; wall-clock p50 gain vs original order)\n" % a.gorder_w)
    md.append("| ef | w | p50 µs | gain | noise (max of spreads) | beats noise |")
    md.append("|---|---|---|---|---|---|")
    for g in gains:
        md.append(f"| {g['ef']} | {g['w']} | {g['p50']:.1f} | {100 * g['gain']:.1f}% | {100 * g['noise']:.1f}% | "
                  f"{'yes' if g['gain'] > g['noise'] else 'no'} |")
    md.append("")
    for ef in report_efs:
        gs = {g["w"]: g for g in gains if g["ef"] == ef}
        f = gs[a.gorder_w]
        for w in ws[1:]:
            d = gs[w]["gain"] - f["gain"]
            tol = max(f["noise"], gs[w]["noise"])
            md.append(f"- ef {ef}: w = {w} vs w = {a.gorder_w}: {100 * d:+.1f} points "
                      + ("(within noise)." if abs(d) <= tol else
                         f"(**beyond noise**: w = {a.gorder_w} {'under' if d > 0 else 'over'}states Gorder's best gain)."))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nwrote {out}/summary.md, settings-retune.json (settings.json not touched)")
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
