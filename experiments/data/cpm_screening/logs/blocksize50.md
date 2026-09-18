# CPM screening -- blocksize 50, 43056 corpus rows, 2392 evaluations
snapshots [250, 300, 350, 400, 450, 500]   bins [8, 16, 32]   replicate seeds up to 48
cells at t=500: median 60, 10-90% 12-160; 19.0% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0755 | 0.2380 | **3.15** | 6.23 (scalars_only) | 3.8% |
| 8 | 1 | 2 | 0.0621 | 0.2664 | **4.29** | 8.31 (scalars_only) | 2.8% |
| 8 | 1 | 4 | 0.0981 | 0.2308 | **2.35** | 14.43 (scalars_only) | 0.0% |
| 8 | 3 | 1 | 0.0908 | 0.2430 | **2.68** | 4.88 (scalars_only) | 2.8% |
| 8 | 3 | 2 | 0.0861 | 0.2525 | **2.93** | 7.68 (scalars_only) | 2.4% |
| 8 | 3 | 4 | 0.0708 | 0.2531 | **3.58** | 10.76 (scalars_only) | 0.2% |
| 8 | 6 | 1 | 0.0960 | 0.2447 | **2.55** | 4.69 (scalars_only) | 4.5% |
| 8 | 6 | 2 | 0.0967 | 0.3094 | **3.20** | 6.66 (scalars_only) | 3.4% |
| 8 | 6 | 4 | 0.0838 | 0.2699 | **3.22** | 10.88 (scalars_only) | 0.4% |
| 16 | 1 | 1 | 0.0692 | 0.2622 | **3.79** | 6.23 (scalars_only) | 3.6% |
| 16 | 1 | 2 | 0.0628 | 0.2881 | **4.58** | 8.31 (scalars_only) | 3.2% |
| 16 | 1 | 4 | 0.0781 | 0.2481 | **3.18** | 14.43 (scalars_only) | 0.0% |
| 16 | 3 | 1 | 0.0678 | 0.2327 | **3.43** | 4.88 (scalars_only) | 3.6% |
| 16 | 3 | 2 | 0.0805 | 0.2503 | **3.11** | 7.68 (scalars_only) | 3.2% |
| 16 | 3 | 4 | 0.0789 | 0.2073 | **2.63** | 10.76 (scalars_only) | 0.7% |
| 16 | 6 | 1 | 0.0910 | 0.2236 | **2.46** | 4.69 (scalars_only) | 3.9% |
| 16 | 6 | 2 | 0.0951 | 0.2732 | **2.87** | 6.66 (scalars_only) | 3.6% |
| 16 | 6 | 4 | 0.1150 | 0.2164 | **1.88** | 10.88 (scalars_only) | 0.7% |
| 32 | 1 | 1 | 0.0637 | 0.2225 | **3.49** | 6.23 (scalars_only) | 4.0% |
| 32 | 1 | 2 | 0.0648 | 0.2539 | **3.92** | 8.31 (scalars_only) | 2.4% |
| 32 | 1 | 4 | 0.0395 | 0.2214 | **5.60** | 14.43 (scalars_only) | 0.0% |
| 32 | 3 | 1 | 0.1006 | 0.1891 | **1.88** | 4.88 (scalars_only) | 2.3% |
| 32 | 3 | 2 | 0.0896 | 0.2039 | **2.28** | 7.68 (scalars_only) | 3.6% |
| 32 | 3 | 4 | 0.0598 | 0.1984 | **3.32** | 10.76 (scalars_only) | 0.7% |
| 32 | 6 | 1 | 0.1018 | 0.1990 | **1.95** | 4.69 (scalars_only) | 3.8% |
| 32 | 6 | 2 | 0.1085 | 0.2359 | **2.17** | 6.66 (scalars_only) | 3.5% |
| 32 | 6 | 4 | 0.0994 | 0.2597 | **2.61** | 10.88 (scalars_only) | 0.4% |

Best protocol on the shipped (equal-weight) metric: **b32_s1_k4** -- 32 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 5.60

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0308 | 1.0051 | 32.65 | 1.196 |
| log_r95 | 1 | 0.0509 | 0.9971 | 19.58 | 1.252 |
| pair_correlation_gofr | 10 | 1.9419 | 2.5391 | 1.31 | 63.590 |
| radial_density_profile_equal_volume | 10 | 1.9511 | 2.4595 | 1.26 | 49.132 |
| radial_fa_equal_volume | 10 | 1.9912 | 2.3225 | 1.17 | 31.421 |
| radial_s2_equal_volume | 10 | 2.3056 | 2.1577 | 0.94 | 20.938 |
| dbscan_gaslike_fraction  (dead) | 1 | 0.0000 | 1.0013 | n/a | 1.970 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0395 | 0.2214 | **5.60** |
| drop_dead | 6 | 0.0461 | 0.2583 | **5.60** |
| scalars_only | 2 | 0.0305 | 0.4394 | **14.43** |
| snr_pruned | 5 | 0.0504 | 0.3532 | **7.01** |

## Parameter screen under that protocol

Summary space: 42 live coordinates (1 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.002, 0.2] log | **17.69** | 50 | log_n 56%, log_r95 21% | [0.002358, 0.03276] | 57% |
| persistence | [0, 0.99] lin | **0.00** | 28 | pair_correlation_gofr 90%, radial_fa_equal_volume 5% | [0.03536, 0.1061] | 7% |
| surface_lambda | [0.25, 8] log | **8.36** | 22 | log_n 53%, log_r95 21% | [0.4642, 5.519] | 71% |
| motility | [100, 4000] log | **3.50** | 20 | log_n 58%, log_r95 18% | [722, 2694] | 36% |
| recalc_time | [2, 60] log | **3.70** | 12 | log_n 28%, pair_correlation_gofr 21% | [8, 33] | 42% |
| adhesion_cc | [40, 200] lin | **0.00** | 2 | pair_correlation_gofr 76%, radial_fa_equal_volume 8% | [57, 91] | 21% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | persistence | surface_lambda | motility | recalc_time | adhesion_cc |
|---|---|---|---|---|---|---|
| **division_rate** | 1.00 | 0.22 | 0.96 | 0.87 | 0.76 | 0.20 |
| **persistence** | 0.22 | 1.00 | 0.26 | 0.23 | 0.12 | 0.09 |
| **surface_lambda** | 0.96 | 0.26 | 1.00 | 0.88 | 0.81 | 0.15 |
| **motility** | 0.87 | 0.23 | 0.88 | 1.00 | 0.70 | 0.02 |
| **recalc_time** | 0.76 | 0.12 | 0.81 | 0.70 | 1.00 | 0.13 |
| **adhesion_cc** | 0.20 | 0.09 | 0.15 | 0.02 | 0.13 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/blocksize50/screening_report.json
