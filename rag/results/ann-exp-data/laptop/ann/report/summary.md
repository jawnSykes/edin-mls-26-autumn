# ANN experiment report

Generated from 1 runs under `/home/jetsykes/code/edin-mls-26-autumn/rag/results/ann`.

## Step 2: settings (FiQA-extra queries)

```
{
  "24k": {
    "M": 16,
    "efc": 200,
    "ef95": 32,
    "ef99": 128,
    "gorder_w": 10
  },
  "58k": {
    "M": 16,
    "efc": 200,
    "ef95": 32,
    "ef99": 128,
    "gorder_w": 3
  }
}
```

Gorder window sweep (Cachegrind, net LL misses per query):

_no data_

## Step 3: Stage 1 (course queries)

recall = recall@10 against exact search; recall_at_1/5 from the same results. search_peak_rss_mb is the search process's peak memory (index + Python + libraries).


| host | corpus | index | M | efc | ef | recall | recall_at_1 | recall_at_5 | p50_us | p95_us | p99_us | batch_qps | ndis_per_q | search_peak_rss_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jetsykes-hp-ubuntu | 24k | hnsw | 16 | 200 | 16 | 0.9017 | 0.9347 | 0.9238 | 68.27 | 88.32 | 100.5 | 1.637e+04 | 408.1 | 126.6 |
| jetsykes-hp-ubuntu | 24k | hnsw | 16 | 200 | 64 | 0.9893 | 0.9958 | 0.9933 | 196.6 | 234.4 | 259.7 | 5402 | 1063 | 126.6 |
| jetsykes-hp-ubuntu | 24k | hnsw | 16 | 200 | 32 | 0.9644 | 0.9832 | 0.9777 | 112.4 | 137.8 | 153.7 | 9321 | 641.4 | 126.6 |
| jetsykes-hp-ubuntu | 24k | hnsw | 16 | 200 | 256 | 0.9992 | 1 | 0.9996 | 649.5 | 773.8 | 833.2 | 1550 | 2951 | 126.6 |
| jetsykes-hp-ubuntu | 24k | hnsw | 16 | 200 | 128 | 0.9971 | 0.9979 | 0.9979 | 352.9 | 419.9 | 464.5 | 2951 | 1780 | 126.6 |
| jetsykes-hp-ubuntu | 24k | exact_dense |  |  |  | 1 | 1 | 1 | 2167 | 2287 | 2639 | nan | 0 | 135.7 |
| jetsykes-hp-ubuntu | 24k | faiss_exact |  |  |  | 1 | 1 | 1 | 1991 | 2102 | 2415 | 2773 | 0 | 126.6 |
| jetsykes-hp-ubuntu | 24k | hnsw | 16 | 200 | 512 | 1 | 1 | 1 | 1250 | 1478 | 1586 | 802.6 | 4796 | 126.6 |
| jetsykes-hp-ubuntu | 58k | hnsw | 16 | 200 | 16 | 0.8855 | 0.9221 | 0.9141 | 80.69 | 108.7 | 128.8 | 1.257e+04 | 442.9 | 231.5 |
| jetsykes-hp-ubuntu | 58k | hnsw | 16 | 200 | 64 | 0.9853 | 0.9937 | 0.992 | 229.7 | 275.2 | 309.8 | 4561 | 1141 | 231.5 |
| jetsykes-hp-ubuntu | 58k | hnsw | 16 | 200 | 32 | 0.9545 | 0.9768 | 0.9705 | 133.1 | 164.7 | 190.5 | 8091 | 687.2 | 231.5 |
| jetsykes-hp-ubuntu | 58k | hnsw | 16 | 200 | 256 | 0.9981 | 0.9979 | 0.9992 | 783 | 1001 | 1166 | 1262 | 3297 | 231.5 |
| jetsykes-hp-ubuntu | 58k | hnsw | 16 | 200 | 128 | 0.9952 | 0.9958 | 0.9979 | 419.7 | 529.6 | 630.6 | 2494 | 1936 | 231.5 |
| jetsykes-hp-ubuntu | 58k | exact_dense |  |  |  | 1 | 1 | 1 | 5409 | 5782 | 6589 | nan | 0 | 236.2 |
| jetsykes-hp-ubuntu | 58k | faiss_exact |  |  |  | 1 | 1 | 1 | 5009 | 5380 | 6221 | 1173 | 0 | 231.5 |
| jetsykes-hp-ubuntu | 58k | hnsw | 16 | 200 | 512 | 0.9996 | 1 | 1 | 1508 | 1857 | 2129 | 671.7 | 5548 | 231.5 |

## Step 6: matched-recall headline (recall@10 vs exact)

p50 = single-query latency. ExactDenseRetriever has no batched call, so only FAISS exact has a batched figure. build_s = HNSW graph construction only (index.add); build_total_s adds reading the vectors and saving the index. Exact search has no build.


| host | corpus | target | hnsw_ef | hnsw_p50_us | exact_dense_p50_us | faiss_exact_p50_us | speedup_vs_exact_dense | speedup_vs_faiss_exact | hnsw_batch_qps | faiss_exact_batch_qps | hnsw_bytes_per_vec | exact_bytes_per_vec | build_s | build_total_s | build_threads | build_peak_rss_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jetsykes-hp-ubuntu | 24k | 0.9 | 16 | 68.27 | 2167 | 1991 | 31.74 | 29.16 | 1.637e+04 | 2773 | 1680 | 1536 | 2.78 | nan | 8 | 119 |
| jetsykes-hp-ubuntu | 24k | 0.95 | 32 | 112.4 | 2167 | 1991 | 19.28 | 17.72 | 9321 | 2773 | 1680 | 1536 | 2.78 | nan | 8 | 119 |
| jetsykes-hp-ubuntu | 24k | 0.99 | 128 | 352.9 | 2167 | 1991 | 6.139 | 5.641 | 2951 | 2773 | 1680 | 1536 | 2.78 | nan | 8 | 119 |
| jetsykes-hp-ubuntu | 58k | 0.9 | 32 | 133.1 | 5409 | 5009 | 40.64 | 37.64 | 8091 | 1173 | 1680 | 1536 | 8.81 | nan | 8 | 227.3 |
| jetsykes-hp-ubuntu | 58k | 0.95 | 32 | 133.1 | 5409 | 5009 | 40.64 | 37.64 | 8091 | 1173 | 1680 | 1536 | 8.81 | nan | 8 | 227.3 |
| jetsykes-hp-ubuntu | 58k | 0.99 | 128 | 419.7 | 5409 | 5009 | 12.89 | 11.94 | 2494 | 1173 | 1680 | 1536 | 8.81 | nan | 8 | 227.3 |

## Steps 4 + 7: reordering (wall clock, course queries)

For random seeds, min/max are over seeds; otherwise over rounds. Break-even = reorder time / per-query saving vs identity.


| host | corpus | ef | order | p50_us | rounds_min | rounds_max | gain_vs_identity | gain_vs_random_mean | reorder_s | breakeven_queries |
|---|---|---|---|---|---|---|---|---|---|---|
| jetsykes-hp-ubuntu | 24k | 32 | bfs | 105.9 | 104.9 | 106.8 | 0.06109 | 0.06335 | 0.5584 | 8.107e+04 |
| jetsykes-hp-ubuntu | 24k | 32 | gorder:10 | 100.4 | 99.94 | 100.6 | 0.1092 | 0.1113 | 110.8 | 9.002e+06 |
| jetsykes-hp-ubuntu | 24k | 32 | identity | 112.7 | 112.4 | 113.4 | 0 | 0.002401 | 0 |  |
| jetsykes-hp-ubuntu | 24k | 32 | random:0 | 113 | 111.9 | 115.8 | -0.002519 | -0.0001115 | 0.002 |  |
| jetsykes-hp-ubuntu | 24k | 32 | random:1 | 113.4 | 113.2 | 113.5 | -0.00604 | -0.003624 | 0.0014 |  |
| jetsykes-hp-ubuntu | 24k | 32 | random:2 | 113.1 | 112.4 | 113.5 | -0.002812 | -0.0004035 | 0.0014 |  |
| jetsykes-hp-ubuntu | 24k | 32 | random:3 | 112.7 | 112.3 | 113.4 | 0.0007184 | 0.003118 | 0.0014 | 1.728e+04 |
| jetsykes-hp-ubuntu | 24k | 32 | random:4 | 112.9 | 112.7 | 113.2 | -0.001384 | 0.001021 | 0.0014 |  |
| jetsykes-hp-ubuntu | 24k | 32 | random (mean of 5 seeds) | 113 | 112.7 | 113.4 | -0.002407 |  |  |  |
| jetsykes-hp-ubuntu | 24k | 128 | bfs | 331.2 | 326.7 | 331.6 | 0.06508 | 0.06554 | 0.5584 | 2.422e+04 |
| jetsykes-hp-ubuntu | 24k | 128 | gorder:10 | 313.5 | 312.1 | 315.3 | 0.1151 | 0.1156 | 110.8 | 2.717e+06 |
| jetsykes-hp-ubuntu | 24k | 128 | identity | 354.2 | 353.7 | 354.9 | 0 | 0.0004847 | 0 |  |
| jetsykes-hp-ubuntu | 24k | 128 | random:0 | 354.3 | 352.1 | 354.9 | -0.0001948 | 0.0002901 | 0.002 |  |
| jetsykes-hp-ubuntu | 24k | 128 | random:1 | 355 | 353.3 | 355.7 | -0.00212 | -0.001634 | 0.0014 |  |
| jetsykes-hp-ubuntu | 24k | 128 | random:2 | 354.9 | 353.6 | 355.9 | -0.001872 | -0.001386 | 0.0014 |  |
| jetsykes-hp-ubuntu | 24k | 128 | random:3 | 353.5 | 350.6 | 355 | 0.002013 | 0.002496 | 0.0014 | 1964 |
| jetsykes-hp-ubuntu | 24k | 128 | random:4 | 354.3 | 350.7 | 355.6 | -0.0002512 | 0.0002336 | 0.0014 |  |
| jetsykes-hp-ubuntu | 24k | 128 | random (mean of 5 seeds) | 354.4 | 353.5 | 355 | -0.000485 |  |  |  |
| jetsykes-hp-ubuntu | 58k | 32 | bfs | 124.6 | 124.4 | 126 | 0.0498 | 0.05346 | 0.1183 | 1.811e+04 |
| jetsykes-hp-ubuntu | 58k | 32 | gorder:3 | 118.2 | 118.1 | 118.3 | 0.0992 | 0.1027 | 311.8 | 2.396e+07 |
| jetsykes-hp-ubuntu | 58k | 32 | identity | 131.2 | 131 | 133.1 | 0 | 0.003852 | 0 |  |
| jetsykes-hp-ubuntu | 58k | 32 | random:0 | 131.4 | 130.9 | 132.5 | -0.001708 | 0.002151 | 0.0045 |  |
| jetsykes-hp-ubuntu | 58k | 32 | random:1 | 131.8 | 131.2 | 132.7 | -0.004856 | -0.0009857 | 0.0031 |  |
| jetsykes-hp-ubuntu | 58k | 32 | random:2 | 131.7 | 131.4 | 133.3 | -0.004109 | -0.0002415 | 0.0031 |  |
| jetsykes-hp-ubuntu | 58k | 32 | random:3 | 131.7 | 131.5 | 138.5 | -0.003827 | 3.949e-05 | 0.0031 |  |
| jetsykes-hp-ubuntu | 58k | 32 | random:4 | 131.8 | 131.5 | 132.5 | -0.004833 | -0.0009629 | 0.0031 |  |
| jetsykes-hp-ubuntu | 58k | 32 | random (mean of 5 seeds) | 131.7 | 131.4 | 131.8 | -0.003867 |  |  |  |
| jetsykes-hp-ubuntu | 58k | 128 | bfs | 386.1 | 385.4 | 390.6 | 0.064 | 0.06599 | 0.1183 | 4481 |
| jetsykes-hp-ubuntu | 58k | 128 | gorder:3 | 366.5 | 365.2 | 368.7 | 0.1116 | 0.1135 | 311.8 | 6.774e+06 |
| jetsykes-hp-ubuntu | 58k | 128 | identity | 412.5 | 412.2 | 413.5 | 0 | 0.002126 | 0 |  |
| jetsykes-hp-ubuntu | 58k | 128 | random:0 | 413.2 | 412.8 | 416.4 | -0.001733 | 0.0003967 | 0.0045 |  |
| jetsykes-hp-ubuntu | 58k | 128 | random:1 | 413.2 | 412.3 | 413.9 | -0.001544 | 0.0005854 | 0.0031 |  |
| jetsykes-hp-ubuntu | 58k | 128 | random:2 | 413.5 | 412.6 | 417.7 | -0.00247 | -0.0003387 | 0.0031 |  |
| jetsykes-hp-ubuntu | 58k | 128 | random:3 | 414 | 412.5 | 414.3 | -0.003532 | -0.001398 | 0.0031 |  |
| jetsykes-hp-ubuntu | 58k | 128 | random:4 | 413.1 | 413 | 415.8 | -0.001374 | 0.0007547 | 0.0031 |  |
| jetsykes-hp-ubuntu | 58k | 128 | random (mean of 5 seeds) | 413.4 | 413.1 | 414 | -0.002131 |  |  |  |

## Step 5: cache-size sweep (Cachegrind, course queries)

Full table: `cachesweep.csv`. Real-machine wall-clock gains at their working set / L3:


| host | corpus | ef | order | ws_over_L3 | gain |
|---|---|---|---|---|---|
| jetsykes-hp-ubuntu | 24k | 32 | bfs | 3.174 | 0.06109 |
| jetsykes-hp-ubuntu | 24k | 32 | gorder:10 | 3.174 | 0.1092 |
| jetsykes-hp-ubuntu | 24k | 128 | bfs | 3.174 | 0.06508 |
| jetsykes-hp-ubuntu | 24k | 128 | gorder:10 | 3.174 | 0.1151 |
| jetsykes-hp-ubuntu | 58k | 32 | bfs | 7.617 | 0.0498 |
| jetsykes-hp-ubuntu | 58k | 32 | gorder:3 | 7.617 | 0.0992 |
| jetsykes-hp-ubuntu | 58k | 128 | bfs | 7.617 | 0.064 |
| jetsykes-hp-ubuntu | 58k | 128 | gorder:3 | 7.617 | 0.1116 |

_matplotlib not installed: plots skipped; plot from the CSVs in this folder._

## Step 9: qrels quality (course dev queries)

_no data_

## Step 10: query-set robustness (FiQA-extra vs course)

| host | corpus | index | order | ef | course_recall | fiqa_recall | course_p50_us | fiqa_p50_us |
|---|---|---|---|---|---|---|---|---|
| jetsykes-hp-ubuntu | 24k | hnsw | identity | 32 | 0.9644 | 0.9588 | 112.6 | 108.5 |
| jetsykes-hp-ubuntu | 24k | hnsw | bfs | 32 | 0.9644 | 0.9588 | 105.9 | 101.3 |
| jetsykes-hp-ubuntu | 24k | hnsw | identity | 128 | 0.9971 | 0.9966 | 354.2 | 343 |
| jetsykes-hp-ubuntu | 24k | hnsw | gorder:10 | 32 | 0.9644 | 0.9588 | 100.4 | 95.32 |
| jetsykes-hp-ubuntu | 24k | hnsw | bfs | 128 | 0.9971 | 0.9966 | 331.2 | 317.8 |
| jetsykes-hp-ubuntu | 24k | exact_dense | identity |  | 1 | 1 | 2167 | 2155 |
| jetsykes-hp-ubuntu | 24k | faiss_exact | identity |  | 1 | 1 | 1991 | 1964 |
| jetsykes-hp-ubuntu | 24k | hnsw | gorder:10 | 128 | 0.9971 | 0.9966 | 313.5 | 304.1 |
| jetsykes-hp-ubuntu | 58k | hnsw | identity | 32 | 0.9545 | 0.9519 | 131.9 | 127.3 |
| jetsykes-hp-ubuntu | 58k | hnsw | bfs | 32 | 0.9545 | 0.9519 | 124.6 | 119.7 |
| jetsykes-hp-ubuntu | 58k | hnsw | identity | 128 | 0.9952 | 0.9944 | 413.2 | 404.5 |
| jetsykes-hp-ubuntu | 58k | hnsw | gorder:3 | 32 | 0.9545 | 0.9519 | 118.2 | 113 |
| jetsykes-hp-ubuntu | 58k | hnsw | bfs | 128 | 0.9952 | 0.9944 | 386.1 | 375.5 |
| jetsykes-hp-ubuntu | 58k | exact_dense | identity |  | 1 | 1 | 5409 | 5370 |
| jetsykes-hp-ubuntu | 58k | faiss_exact | identity |  | 1 | 1 | 5009 | 4878 |
| jetsykes-hp-ubuntu | 58k | hnsw | gorder:3 | 128 | 0.9952 | 0.9944 | 366.5 | 353.8 |

## Step 11: the full pipeline, exact_dense vs HNSW (dev queries)

Times are medians; qrels metrics are means over dev queries with qrels. retrieval_call_ms is the whole retrieve() call; retrieval_share = retrieval_call_ms / end_to_end_ms.

_no data_

## Hardware counters (perf stat, warm cache, per query, median over rounds)

| host | corpus | index | ef | order | rounds | cycles_per_q | instructions_per_q | ipc | LLC_loads_per_q | LLC_load_misses_per_q | L1_dcache_load_misses_per_q | dTLB_load_misses_per_q | ndis_per_q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jetsykes-hp-ubuntu | 24k | hnsw | 32 | identity | 3 | 2.216e+05 | 1.701e+05 | 0.7671 | 6304 | 4233 | 1.497e+04 | 594.6 | 641.4 |
| jetsykes-hp-ubuntu | 24k | hnsw | 32 | random:3 | 3 | 2.215e+05 | 1.701e+05 | 0.7679 | 6325 | 4252 | 1.499e+04 | 588.1 | 641.4 |
| jetsykes-hp-ubuntu | 24k | hnsw | 128 | identity | 3 | 6.936e+05 | 4.912e+05 | 0.7082 | 1.917e+04 | 1.327e+04 | 4.272e+04 | 1823 | 1780 |
| jetsykes-hp-ubuntu | 24k | hnsw | 32 | random:4 | 3 | 2.218e+05 | 1.702e+05 | 0.7677 | 6333 | 4262 | 1.506e+04 | 592.8 | 641.4 |
| jetsykes-hp-ubuntu | 24k | exact_dense |  | identity | 3 | 4.289e+06 | 3.735e+06 | 0.8708 | 8.653e+04 | 6.145e+04 | 2.221e+05 | 5.544 | 0 |
| jetsykes-hp-ubuntu | 24k | hnsw | 32 | random:0 | 3 | 2.216e+05 | 1.702e+05 | 0.7678 | 6314 | 4254 | 1.498e+04 | 589.3 | 641.4 |
| jetsykes-hp-ubuntu | 24k | hnsw | 128 | random:1 | 3 | 6.955e+05 | 4.913e+05 | 0.7063 | 1.916e+04 | 1.334e+04 | 4.26e+04 | 1819 | 1780 |
| jetsykes-hp-ubuntu | 24k | hnsw | 128 | random:0 | 3 | 6.965e+05 | 4.912e+05 | 0.7053 | 1.916e+04 | 1.332e+04 | 4.272e+04 | 1824 | 1780 |
| jetsykes-hp-ubuntu | 24k | hnsw | 32 | gorder:10 | 3 | 1.996e+05 | 1.702e+05 | 0.8529 | 5533 | 3420 | 1.356e+04 | 401.8 | 641.4 |
| jetsykes-hp-ubuntu | 24k | hnsw | 128 | random:4 | 3 | 6.97e+05 | 4.911e+05 | 0.7045 | 1.9e+04 | 1.323e+04 | 4.244e+04 | 1824 | 1780 |
| jetsykes-hp-ubuntu | 24k | hnsw | 32 | bfs | 3 | 2.087e+05 | 1.704e+05 | 0.8164 | 5964 | 3788 | 1.435e+04 | 492.6 | 641.4 |
| jetsykes-hp-ubuntu | 24k | hnsw | 128 | random:3 | 3 | 6.962e+05 | 4.913e+05 | 0.7058 | 1.912e+04 | 1.329e+04 | 4.264e+04 | 1810 | 1780 |
| jetsykes-hp-ubuntu | 24k | hnsw | 32 | random:1 | 3 | 2.211e+05 | 1.701e+05 | 0.769 | 6358 | 4280 | 1.511e+04 | 598.4 | 641.4 |
| jetsykes-hp-ubuntu | 24k | hnsw | 128 | random:2 | 3 | 6.957e+05 | 4.914e+05 | 0.7063 | 1.9e+04 | 1.323e+04 | 4.239e+04 | 1817 | 1780 |
| jetsykes-hp-ubuntu | 24k | hnsw | 128 | gorder:10 | 3 | 6.185e+05 | 4.912e+05 | 0.7943 | 1.692e+04 | 1.055e+04 | 3.864e+04 | 1216 | 1780 |
| jetsykes-hp-ubuntu | 24k | hnsw | 32 | random:2 | 3 | 2.214e+05 | 1.701e+05 | 0.7681 | 6359 | 4269 | 1.51e+04 | 593.6 | 641.4 |
| jetsykes-hp-ubuntu | 24k | faiss_exact |  | identity | 3 | 3.944e+06 | 4.884e+06 | 1.238 | 6.208e+04 | 5.27e+04 | 2.59e+05 | 181.1 | 0 |
| jetsykes-hp-ubuntu | 24k | hnsw | 128 | bfs | 3 | 6.519e+05 | 4.914e+05 | 0.7539 | 1.786e+04 | 1.159e+04 | 4.034e+04 | 1480 | 1780 |
| jetsykes-hp-ubuntu | 58k | hnsw | 32 | identity | 3 | 2.602e+05 | 1.777e+05 | 0.6824 | 7650 | 6119 | 1.682e+04 | 718.5 | 687.2 |
| jetsykes-hp-ubuntu | 58k | hnsw | 32 | random:3 | 3 | 2.603e+05 | 1.776e+05 | 0.682 | 7646 | 6127 | 1.682e+04 | 719.1 | 687.2 |
| jetsykes-hp-ubuntu | 58k | hnsw | 128 | identity | 3 | 8.165e+05 | 5.167e+05 | 0.6325 | 2.314e+04 | 1.894e+04 | 4.831e+04 | 2189 | 1936 |
| jetsykes-hp-ubuntu | 58k | hnsw | 32 | random:4 | 3 | 2.604e+05 | 1.777e+05 | 0.6822 | 7691 | 6117 | 1.689e+04 | 725.4 | 687.2 |
| jetsykes-hp-ubuntu | 58k | exact_dense |  | identity | 3 | 1.068e+07 | 9.011e+06 | 0.8434 | 2.163e+05 | 1.686e+05 | 5.533e+05 | 22.18 | 0 |
| jetsykes-hp-ubuntu | 58k | hnsw | 32 | random:0 | 3 | 2.601e+05 | 1.776e+05 | 0.6828 | 7670 | 6115 | 1.688e+04 | 718 | 687.2 |
| jetsykes-hp-ubuntu | 58k | hnsw | 128 | random:1 | 3 | 8.168e+05 | 5.166e+05 | 0.6326 | 2.324e+04 | 1.905e+04 | 4.849e+04 | 2189 | 1936 |
| jetsykes-hp-ubuntu | 58k | hnsw | 128 | random:0 | 3 | 8.169e+05 | 5.166e+05 | 0.6323 | 2.323e+04 | 1.901e+04 | 4.864e+04 | 2240 | 1936 |
| jetsykes-hp-ubuntu | 58k | hnsw | 32 | gorder:3 | 3 | 2.336e+05 | 1.777e+05 | 0.7599 | 6673 | 5122 | 1.554e+04 | 513.8 | 687.2 |
| jetsykes-hp-ubuntu | 58k | hnsw | 128 | random:4 | 3 | 8.178e+05 | 5.166e+05 | 0.6317 | 2.313e+04 | 1.893e+04 | 4.827e+04 | 2197 | 1936 |
| jetsykes-hp-ubuntu | 58k | hnsw | 32 | bfs | 3 | 2.463e+05 | 1.777e+05 | 0.7214 | 7158 | 5613 | 1.619e+04 | 609.8 | 687.2 |
| jetsykes-hp-ubuntu | 58k | hnsw | 128 | random:3 | 3 | 8.18e+05 | 5.165e+05 | 0.6316 | 2.314e+04 | 1.893e+04 | 4.834e+04 | 2201 | 1936 |
| jetsykes-hp-ubuntu | 58k | hnsw | 32 | random:1 | 3 | 2.608e+05 | 1.777e+05 | 0.6818 | 7725 | 6135 | 1.697e+04 | 720.9 | 687.2 |
| jetsykes-hp-ubuntu | 58k | hnsw | 128 | random:2 | 3 | 8.172e+05 | 5.164e+05 | 0.632 | 2.323e+04 | 1.899e+04 | 4.845e+04 | 2183 | 1936 |
| jetsykes-hp-ubuntu | 58k | hnsw | 128 | gorder:3 | 3 | 7.24e+05 | 5.167e+05 | 0.7134 | 2.019e+04 | 1.553e+04 | 4.431e+04 | 1586 | 1936 |
| jetsykes-hp-ubuntu | 58k | hnsw | 32 | random:2 | 3 | 2.599e+05 | 1.776e+05 | 0.6834 | 7651 | 6126 | 1.682e+04 | 718.4 | 687.2 |
| jetsykes-hp-ubuntu | 58k | faiss_exact |  | identity | 3 | 9.786e+06 | 1.167e+07 | 1.193 | 1.675e+05 | 1.455e+05 | 6.572e+05 | 227.2 | 0 |
| jetsykes-hp-ubuntu | 58k | hnsw | 128 | bfs | 3 | 7.629e+05 | 5.168e+05 | 0.6774 | 2.138e+04 | 1.693e+04 | 4.593e+04 | 1883 | 1936 |

## Graph diagnostics

_no data_


Locality per ordering (mean id gap along level-0 links):

_no data_
