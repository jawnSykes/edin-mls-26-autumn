# Re-derived operating points on 350 course practice queries (jetsykes-hp-ubuntu, 20261117T135545)

Graph fixed at M = 16, efConstruction = 200 (hnswlib defaults; the graph every experiment measured). efSearch: smallest ef in the sweep reaching each recall target. 3 rounds × 3 reps (sweep), 5 rounds (Gorder).

## Recall@10 and p50 by ef (practice queries)

| ef | recall@10 | p50 µs (median) | min–max over rounds |
|---|---|---|---|
| 16 | 0.9011 | 68.8 | 68.6–69.4 |
| 32 | 0.9649 | 114.1 | 113.6–114.1 |
| 64 | 0.9906 | 199.8 | 199.7–200.7 |
| 128 | 0.9974 | 357.4 | 356.7–358.2 |
| 256 | 0.9994 | 655.4 | 654.6–658.1 |
| 512 | 1.0000 | 1269.9 | 1269.5–1272.0 |

- **ef95** (recall ≥ 0.95): ef 32 (recall 0.9649); experiments used ef 32: **same**.
- **ef99** (recall ≥ 0.99): ef 64 (recall 0.9906); experiments used ef 128: **differs**. ef 128 has recall 0.9974 here, so report it by its measured recall rather than as the 0.99 target.

## Gorder window sensitivity (fixed w = 10; wall-clock p50 gain vs original order)

| ef | w | p50 µs | gain | noise (max of spreads) | beats noise |
|---|---|---|---|---|---|
| 32 | 10 | 102.7 | 9.7% | 1.3% | yes |
| 32 | 3 | 102.3 | 10.0% | 2.2% | yes |
| 32 | 5 | 102.6 | 9.8% | 1.2% | yes |
| 128 | 10 | 316.9 | 11.3% | 0.7% | yes |
| 128 | 3 | 316.3 | 11.5% | 0.8% | yes |
| 128 | 5 | 316.8 | 11.3% | 0.4% | yes |

- ef 32: w = 3 vs w = 10: +0.3 points (within noise).
- ef 32: w = 5 vs w = 10: +0.1 points (within noise).
- ef 128: w = 3 vs w = 10: +0.2 points (within noise).
- ef 128: w = 5 vs w = 10: +0.0 points (within noise).
