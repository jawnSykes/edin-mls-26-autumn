import faiss
from pathlib import Path
import time
import numpy as np
import json
import os


def name_index(path: str | Path, M: int, efConstruction: int) -> str:
    stem = Path(path).stem
    return f"hnsw-{stem}-{M}-{efConstruction}.faiss"


class HNSWRetriever:
    def __init__(self, embeddings_path: str | Path, ids_path: str | Path,
                 # HNSW settings
                 M: int,  # links per node
                 efConstruction: int,  # graph building effort
                 efSearch: int):  # query time search effort
        # We need to persist indices if we are to re-use them for experiments -- just incase its non-det
        persist_dir = os.getenv("PERSIST_INDEX_PATH")
        if not persist_dir:
            raise EnvironmentError("Path for persisting indices is not set, set PERSIST_INDEX_PATH to an "
                                   "appropriate path as an environment variable")
        self.persist_dir = Path(persist_dir)
        self.name = name_index(embeddings_path, M, efConstruction)
        self.index_path = self.persist_dir / self.name

        # faiss requires C-contiguous float32
        embeddings = np.ascontiguousarray(np.load(embeddings_path), dtype=np.float32)
        self.record_ids = json.loads(Path(ids_path).read_text(encoding='utf-8'))

        if len(self.record_ids) != embeddings.shape[0]:
            raise ValueError("embedding row count does not match ID list")

        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("corpus embeddings are expected to be L2-normalised")

        d = embeddings.shape[1]

        # Seconds spent on graph construction (index.add); None when a saved graph was loaded
        self.build_s = None
        self.loaded = self.index_path.exists()
        if self.loaded:
            # There is already an index at this config -- reuse that exact graph
            self.index = faiss.read_index(str(self.index_path))
            if self.index.ntotal != len(self.record_ids) or self.index.d != d:
                raise ValueError(f"persisted index {self.index_path} does not match the corpus")
        else:
            self.index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = efConstruction
            t = time.perf_counter()
            self.index.add(embeddings)
            self.build_s = time.perf_counter() - t
            self.persist_index()

        self.set_ef_search(efSearch)

    def set_ef_search(self, efSearch: int) -> None:
        # Query-time only, so one loaded graph can be swept over efSearch
        self.index.hnsw.efSearch = efSearch

    def retrieve(self, query_vector: np.ndarray, k: int = 5):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.size != self.index.d or not np.isfinite(q).all():
            raise ValueError("query vector must be finite and match embedding dimension")
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            raise ValueError("zero-norm query vector")
        q = q / q_norm

        k = min(k, len(self.record_ids))

        start = time.perf_counter()
        # faiss expects a (n_queries, d) batch and returns (n_queries, k) arrays,
        # already sorted by descending inner product
        D, I = self.index.search(q.reshape(1, -1), k)  # D=similarities, I=row indices
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        scores, idx = D[0], I[0]
        keep = idx >= 0 # Filter out padded entries from faiss
        return {
            "record_ids": [self.record_ids[int(i)] for i in idx[keep]],
            "scores": [float(s) for s in scores[keep]],
            "latency_ms": elapsed_ms
        }

    def persist_index(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
