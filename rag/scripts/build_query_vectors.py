"""Encode every audio query once and save the vectors, so benchmarks never run ASR or the encoder.

Writes <out-dir>/queries_enc_a.npy (n, 384) float32 and queries_enc_a_ids.json (row -> query_id).
Transcripts come from the cache; any missing ones are transcribed with ASR and appended to it.

    python scripts/build_query_vectors.py --transcript-cache cache/transcripts.jsonl --out-dir cache
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import numpy as np

from src.data import CourseData, load_jsonl
from src.encoder import LocalEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "rag_corpus")
    ap.add_argument("--splits", nargs="+", default=["practice", "dev"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data = CourseData(args.data_dir)
    items = [it for split in args.splits for it in data.audio_items(split)]

    cache = {}
    if args.transcript_cache.exists():
        cache = {r["query_id"]: r["transcript"] for r in load_jsonl(args.transcript_cache)}

    # Only start ASR if something is missing from the cache
    missing = [it for it in items if it["query_id"] not in cache]
    if missing:
        from src.asr_adapter import ASRAdapter
        asr = ASRAdapter()
        args.transcript_cache.parent.mkdir(parents=True, exist_ok=True)
        with args.transcript_cache.open("a", encoding="utf-8") as f:
            for i, it in enumerate(missing, start=1):
                text = asr.transcribe(args.data_dir / it["audio_file"])
                cache[it["query_id"]] = text
                f.write(json.dumps({"query_id": it["query_id"], "transcript": text}, ensure_ascii=False) + "\n")
                print(f"ASR [{i}/{len(missing)}] {it['query_id']}")

    qids = [it["query_id"] for it in items]
    encoder = LocalEncoder(ROOT / "models" / "enc_a", device=args.device)
    # One at a time, as evaluate_rag.py does (batching pads, which can shift vectors slightly)
    vecs = np.stack([encoder.encode(cache[q]) for q in qids])  # already L2-normalised

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "queries_enc_a.npy", vecs)
    (args.out_dir / "queries_enc_a_ids.json").write_text(json.dumps(qids), encoding="utf-8")
    print(f"saved {vecs.shape} query vectors to {args.out_dir}")


if __name__ == "__main__":
    main()
