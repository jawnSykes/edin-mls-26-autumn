"""Exact-search metrics from the saved query vectors, printed in evaluate_rag.py's "Aggregate:" format.

If the saved vectors match what evaluate_rag.py encodes, the two Aggregate blocks are identical.

    python scripts/check_query_vectors.py --vectors-dir ~/mls-rag-assets/query_vectors --split practice
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import numpy as np

from src.data import CourseData
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from src.retriever import ExactDenseRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "rag_corpus")
    args = ap.parse_args()

    vecs = np.load(args.vectors_dir / "queries_enc_a.npy")
    qids = json.loads((args.vectors_dir / "queries_enc_a_ids.json").read_text(encoding="utf-8"))
    row_of = {q: i for i, q in enumerate(qids)}

    data = CourseData(args.data_dir)
    qrels = data.load_qrels(args.split)
    retriever = ExactDenseRetriever(
        args.data_dir / "embeddings" / "corpus_enc_a.npy",
        args.data_dir / "embeddings" / "corpus_enc_a_ids.json",
    )

    # Same query order and metrics as evaluate_rag.py
    rows = []
    for item in data.audio_items(args.split):
        qid = item["query_id"]
        ranked = retriever.retrieve(vecs[row_of[qid]], k=100)["record_ids"]
        rel = qrels[qid]
        rows.append({
            "Recall@5": recall_at_k(ranked, rel, 5),
            "Recall@10": recall_at_k(ranked, rel, 10),
            "Recall@20": recall_at_k(ranked, rel, 20),
            "Recall@100": recall_at_k(ranked, rel, 100),
            "MRR@10": mrr_at_k(ranked, rel, 10),
            "NDCG@10": ndcg_at_k(ranked, rel, 10),
        })

    print("\nAggregate:")
    for key in rows[0]:
        print(f"  {key:<10} {np.mean([r[key] for r in rows]):.6f}")


if __name__ == "__main__":
    main()
