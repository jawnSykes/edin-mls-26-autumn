#!/bin/bash
# Build the full-FiQA corpus (course 24k + remaining BeIR/fiqa docs) with enc_a vectors,
# then package it to ~/mls-rag-assets/fiqa_full/. See rag/docs/corpus-provenance.md.
#
# One-time, on the head node, from the repo root:
#     source ~/mls-rag-env/bin/activate
#     pip install pyarrow huggingface_hub
#     # course assets installed per rag/README.md step 2 (rag/models/enc_a, rag/data/rag_corpus)
# The job downloads BeIR/fiqa itself (needs internet on the compute node) into a
# Hugging Face cache in node-local scratch, so nothing is cached in $HOME.
# Disk: ~300 MB in $HOME (rag/data/fiqa_full working copy + the package zip).
#
# Submit from the repo root:
#     sbatch rag/scripts/build_fiqa_full.sbatch
#     squeue -u "$USER"          # wait until the job is gone
#     cat fiqa-full-JOB_ID.out
# Smoke test instead (no packaging; output in node-local /disk/$USER/fiqa-full/JOB_ID, deleted at job end):
#     sbatch --export=ALL,SMOKE=200 rag/scripts/build_fiqa_full.sbatch
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=fiqa-full-%j.out

set -euo pipefail

export PATH=/opt/cuda-12.8.0/bin:$PATH
# shellcheck disable=SC1091
source "${RAG_VENV:-$HOME/mls-rag-env}/bin/activate"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

if [[ ! -d rag/scripts || ! -f rag/models/enc_a/model.safetensors ]]; then
    echo "Run from the repo root with the course assets installed (rag/models/enc_a missing)" >&2
    exit 1
fi

hostname
nvidia-smi -L
python - <<'EOF'
import importlib.util, sys
missing = [m for m in ("torch", "numpy", "sentence_transformers", "pyarrow", "huggingface_hub")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"Missing in {sys.prefix}: {', '.join(missing)}\n"
             "On the head node: source ~/mls-rag-env/bin/activate && pip install pyarrow huggingface_hub")
import torch
assert torch.cuda.is_available(), "no CUDA"
print("torch", torch.__version__, torch.cuda.get_device_name(0))
EOF
command -v quota >/dev/null && quota -s || true

# Node-local scratch: /disk/$USER/fiqa-full/<job id>; created here, removed when the job exits.
SCRATCH="/disk/scratch/$USER/fiqa-full/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"; rmdir "/disk/scratch/$USER/fiqa-full" 2>/dev/null || true' EXIT
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$TMPDIR"
echo "Node-local scratch: $SCRATCH"
# FiQA download cache, node-local; `prepare` fetches into it (online), later steps stay offline.
export HF_HOME="$SCRATCH/hf"
OFFLINE=(env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)

S=rag/scripts/build_fiqa_full.py
if [[ -n "${SMOKE:-}" ]]; then
    OUT="$SCRATCH/fiqa_full_smoke"
    python "$S" prepare --out-dir "$OUT"
    "${OFFLINE[@]}" python "$S" encode --data-dir "$OUT" --limit "$SMOKE" --gate-n 200
    ls -la "$OUT" "$OUT/embeddings"
    python -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['encode'], indent=2))" "$OUT/manifest.json"
    echo "Smoke run OK (output was node-local and is removed at job exit; not packaged)"
    exit 0
fi

python "$S" prepare
"${OFFLINE[@]}" python "$S" encode
"${OFFLINE[@]}" python "$S" package
