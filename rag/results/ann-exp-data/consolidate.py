"""Collapse the per-run result dirs into one CSV per step (one-off; already applied).

Course corpus (24k) and course queries only: full-FiQA rows are dropped.

The experiment scripts write  <root>/<step>/<host>-<ts>/rows*.csv  plus an env.txt per run.
This script keeps exactly the rows claims_analysis_original.ipynb uses (corpus 24k, and the
same newest-run selection the notebook applied), and writes:

  consolidated/<step>.csv           rows + provenance columns (step, run, file, run_ts, run_host)
  consolidated/env/<step>/<run>.txt env.txt of every run that contributed a row, byte-identical
  consolidated/1-setup-check.json   FAISS-vs-numpy exact agreement (from the 1-setup run)
  consolidated/settings.json        fixed graph configuration
  consolidated/locality.csv         offline locality metrics (offline_locality.py now writes here)

The raw run dirs were deleted after this ran; they remain in git history before the commit
that added this file. To fold in a new run, restore the raw layout and re-run:

  uv run --with pandas python consolidate.py      (from rag/results/ann-exp-data/)
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LAPTOP = HERE / "laptop" / "ann"
PIPELINE = HERE / "ann"
OUT = HERE / "consolidated"
CORPUS = "24k"
RUN_RE = re.compile(r"^(?P<host>.+)-(?P<ts>\d{8}T\d{6})$")

# step -> (root, file glob, selection). Selection mirrors the notebook's loader calls:
#   list of keys -> newest run per cell of those keys; "latest" -> newest run only; "all" -> every run.
STEPS = {
    "1-setup":         (LAPTOP, "rows-vs-cached.csv", "all"),
    "3-stage1":        (LAPTOP, "rows*.csv", ["host_short", "corpus", "index", "M", "efc", "ef", "round"]),
    "4-reorder":       (LAPTOP, "rows*.csv", ["host_short", "corpus", "ef", "order", "round"]),
    "perf":            (LAPTOP, "rows*.csv", ["host_short", "corpus", "index", "ef", "order", "round"]),
    "graph":           (LAPTOP, "rows.csv", "all"),
    "build":           (LAPTOP, "rows.csv", "latest"),
    "9-quality":       (LAPTOP, "rows*.csv", "latest"),
    "robust":          (LAPTOP, "rows*.csv", ["index", "M", "efc", "ef", "round"]),
    "robust-practice": (LAPTOP, "rows*.csv", "latest"),
    "reorder-time":    (LAPTOP, "rows.csv", "latest"),
    "retune":          (LAPTOP, "rows-gorder.csv", "latest"),
    "11-pipeline":     (PIPELINE, "rows.csv", "all"),
}


def load(root: Path, step: str, pattern: str) -> pd.DataFrame:
    frames = []
    for d in sorted(p for p in (root / step).glob("*") if p.is_dir()):
        m = RUN_RE.match(d.name)
        for f in sorted(d.glob(pattern)):
            df = pd.read_csv(f)
            df["step"], df["run"], df["file"] = step, d.name, f.stem
            df["run_ts"] = m["ts"] if m else ""
            df["run_host"] = m["host"] if m else d.name
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if "corpus" in df:
        df = df[df["corpus"] == CORPUS]
    if "item" in df:  # 1-setup: drop the full-FiQA ground-truth row
        df = df[~df["item"].str.contains("fiqa")]
    return df


def select(df: pd.DataFrame, how) -> pd.DataFrame:
    if how == "all":
        return df
    if how == "latest":
        return df[df["run_ts"] == df["run_ts"].max()]
    host = df["host"] if "host" in df else df["run_host"]
    df = df.assign(host_short=host.astype(str).str.split(".").str[0])
    keys = [k for k in how if k in df]
    newest = df.groupby(keys, dropna=False)["run_ts"].transform("max")
    return df[df["run_ts"] == newest].drop(columns="host_short")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for step, (root, pattern, how) in STEPS.items():
        df = select(load(root, step, pattern), how)
        df.to_csv(OUT / f"{step}.csv", index=False)
        env_dir = OUT / "env" / step
        env_dir.mkdir(parents=True, exist_ok=True)
        for run in sorted(df["run"].unique()):
            src = root / step / run / "env.txt"
            if src.exists():
                shutil.copyfile(src, env_dir / f"{run}.txt")
        print(f"{step:16s} {len(df):5d} rows from {df['run'].nunique()} run(s)")
    check = json.loads(sorted((LAPTOP / "1-setup").glob("*/check.json"))[-1].read_text())
    check["queries"] = {k: v for k, v in check["queries"].items() if "fiqa" not in k}
    check.pop("fiqa_removed_as_course", None)
    check["exact_agreement"] = {k: v for k, v in check["exact_agreement"].items() if "fiqa" not in k}
    (OUT / "1-setup-check.json").write_text(json.dumps(check, indent=2) + "\n")
    settings = json.loads((LAPTOP / "settings.json").read_text())
    (OUT / "settings.json").write_text(json.dumps({CORPUS: settings[CORPUS]}, indent=2) + "\n")
    loc = pd.read_csv(HERE / "offline" / "locality.csv")
    loc[loc["corpus"] == CORPUS].to_csv(OUT / "locality.csv", index=False)


if __name__ == "__main__":
    main()
