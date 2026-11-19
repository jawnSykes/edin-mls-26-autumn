#!/usr/bin/env bash
# A throwaway shell for screen recording that hides your username and hostname.
#
# Usage (Ubuntu laptop):
#     bash rag/extensions/scripts/demo-shell.sh      # opens the masked shell in rag/
#     ...run the demo commands listed in video-guide.md...
#     exit                                           # back to your normal shell
#
# Overrides (env, set before launching): PY_ENV (venv, default ~/mls-rag-env),
#   ASSETS (~/mls-rag-assets), CACHE ($ASSETS/ann-cache-laptop).
#
# What it changes (only inside this shell; nothing persistent):
#   - prompt:        demo@laptop:rag$
#   - window title:  "ANN demo" (the title bar normally shows user@host:path)
#   - working dir:   /tmp/ann-demo/rag, a symlink to the repo's rag/, so pwd shows no home path
#   - `mask`:        pipe any command through it to replace $HOME, $USER and the hostname,
#                    e.g.  ls -l | mask   (the demo scripts print no home paths;
#                    bench.py prints only file names and $OUT paths)
#   - HISTFILE off, so demo commands don't land in your history
#   - environment for the demo commands, so they stay short on screen:
#       venv activated (`python` = mls-rag-env, without the venv's prompt prefix)
#       PERSIST_INDEX_PATH = the ANN cache;  $EMB = corpus vectors;  $Q = the 475 course queries
#       $OUT = /tmp/ann-demo/out (bench.py --out/--log-dir; nothing written to the repo)
#       FAISS_SIMD_LEVEL=AVX2 + single-threaded BLAS/OMP, as in the report runs (laptop_rerun.py)
#
# Before recording
#   - AC power; pin the clock as for the report runs (performance governor, turbo off --
#     with_pinned_freq.sh, or by hand). bench.py warns if the clock settings change mid-run.
#   - perf counters: sudo sysctl kernel.perf_event_paranoid=1 (revert afterwards).
#   - The Gorder permutation must be cached next to the graph, or bench.py computes it (~80 s):
#     ls $PERSIST_INDEX_PATH/24k/*.perm.npz. If you record the computation, cut it and say so.
#   - Each bench.py cell is a fresh process, so most of a run is start-up; fine to speed up in
#     the edit (mark the cut).
#   - bench.py appends to --out: use a new file name per take (or rm $OUT/*.csv).
#
# Not covered -- check before recording:
#   - GNOME top bar / dock / other windows: record only the terminal window (or a region).
#   - `ls -l` and similar show the file owner: pipe through mask or avoid on camera.
#   - Full GUI masking: temporarily rename the host and restore it afterwards:
#         old=$(hostname); sudo hostnamectl set-hostname laptop   # before
#         sudo hostnamectl set-hostname "$old"                     # after
#     (open a new terminal after renaming; some apps cache the old name until re-login).
#
# Readability tips: make a dedicated terminal profile (Preferences -> Profiles -> +) with an
# 18-20 pt monospace font, a high-contrast theme and ~110 columns; maximise the window and
# record at 1080p. The widest demo lines (bench.py cells) fit in ~110 columns.
set -euo pipefail

RAG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DEMO_RAG="$RAG" DEMO_PY_ENV="${PY_ENV:-$HOME/mls-rag-env}" DEMO_ASSETS="${ASSETS:-$HOME/mls-rag-assets}"
export DEMO_CACHE="${CACHE:-${ASSETS:-$HOME/mls-rag-assets}/ann-cache-laptop}"
mkdir -p /tmp/ann-demo/out
ln -sfn "$RAG" /tmp/ann-demo/rag

RC="$(mktemp /tmp/ann-demo/rc.XXXXXX)"
cat >"$RC" <<'EOF'
export MASK_HOME="$HOME" MASK_USER="${USER:-$(id -un)}" MASK_HOST="$(hostname -s 2>/dev/null || hostname)"
mask() {
    perl -pe 'BEGIN{$|=1} s/\Q$ENV{MASK_HOME}\E/~/g; s/\b\Q$ENV{MASK_USER}\E\b/user/g; s/\b\Q$ENV{MASK_HOST}\E\b/laptop/gi'
}
export -f mask
unset HISTFILE
export VIRTUAL_ENV_DISABLE_PROMPT=1
[[ -f "$DEMO_PY_ENV/bin/activate" ]] && source "$DEMO_PY_ENV/bin/activate" || echo "warning: no venv at $DEMO_PY_ENV" | mask
export PERSIST_INDEX_PATH="$DEMO_CACHE" EMB="$DEMO_RAG/data/rag_corpus/embeddings/corpus_enc_a.npy"
export Q="$DEMO_CACHE/queries/course.npy" OUT=/tmp/ann-demo/out
export FAISS_SIMD_LEVEL=AVX2 OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PS1='\[\e[1;32m\]demo@laptop\[\e[0m\]:\[\e[1;34m\]\W\[\e[0m\]\$ '
PROMPT_COMMAND='printf "\033]0;ANN demo\007"'
alias ls='ls --color=auto'
cd /tmp/ann-demo/rag
clear
EOF
trap 'rm -f "$RC"' EXIT
bash --rcfile "$RC" -i
